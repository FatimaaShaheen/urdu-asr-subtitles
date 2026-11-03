"""Whisper + LoRA fine-tuning (Step 7)."""
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.data.augment import augment_waveform
from src.evaluation.metrics import compute_metrics

SR = 16000


@dataclass
class WhisperCollator:
    """Batch banata hai: audio -> log-mel features, text -> token ids."""
    processor: object
    decoder_start_token_id: int
    augment: bool = True
    aug_cfg: dict = None
    max_label_len: int = 448
    _rng: object = None

    def __call__(self, features):
        audios = []
        for f in features:
            y = np.asarray(f["audio"]["array"], dtype=np.float32)
            # augmentation sirf train examples par (is_train column se pata chalta hai)
            if self.augment and f.get("is_train", False):
                if self._rng is None:
                    self._rng = np.random.default_rng()
                y = augment_waveform(y, self._rng, **(self.aug_cfg or {}))
            audios.append(y)

        batch = self.processor.feature_extractor(
            audios, sampling_rate=SR, return_tensors="pt")

        tok = self.processor.tokenizer(
            [f["text"] for f in features], padding=True, truncation=True,
            max_length=self.max_label_len, return_tensors="pt")
        labels = tok["input_ids"].masked_fill(tok["attention_mask"].ne(1), -100)
        # <|startoftranscript|> hata do, model khud lagata hai
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_compute_metrics(tokenizer):
    def fn(pred):
        pad = tokenizer.pad_token_id
        pred_ids = np.where(pred.predictions == -100, pad, pred.predictions)
        label_ids = np.where(pred.label_ids == -100, pad, pred.label_ids)
        hyps = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        m = compute_metrics(refs, hyps)
        return {"wer": m["wer"], "cer": m["cer"]}
    return fn


def run_experiment(name, train_ds, val_ds, out_dir, cfg, resume=False):
    """Ek experiment chalao. (log_history, info) wapas deta hai."""
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    t, lc = cfg["train"], cfg["lora"]

    processor = WhisperProcessor.from_pretrained(
        cfg["model_name"], language=cfg["language"], task=cfg["task"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_name"])
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.apply_spec_augment = True            # SpecAugment (sirf training mein)
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None

    lora = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      target_modules=lc["target_modules"], bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())

    # train/val ka farq collator ko batane ke liye
    train_ds = train_ds.add_column("is_train", [True] * len(train_ds))
    val_ds = val_ds.add_column("is_train", [False] * len(val_ds))

    collator = WhisperCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        aug_cfg=cfg["augment"], max_label_len=t["max_label_len"])

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        max_steps=t["max_steps"],
        fp16=torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=t["eval_steps"],
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=2,
        logging_steps=t["logging_steps"],
        predict_with_generate=True, generation_max_length=225,
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=2, report_to="none", seed=cfg["seed"])

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=make_compute_metrics(processor.tokenizer),
        processing_class=processor.feature_extractor)

    ckpt = None
    if resume and os.path.isdir(out_dir):
        from transformers.trainer_utils import get_last_checkpoint
        ckpt = get_last_checkpoint(out_dir)
        print("Resume checkpoint:", ckpt)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=ckpt)
    seconds = time.time() - t0

    model.save_pretrained(f"{out_dir}/adapter")      # sirf LoRA adapter (chhoti file)
    processor.save_pretrained(f"{out_dir}/adapter")

    info = {"experiment": name, "trainable_params": n_train, "all_params": n_all,
            "seconds": round(seconds), "n_train": len(train_ds), "n_val": len(val_ds)}
    return trainer.state.log_history, info