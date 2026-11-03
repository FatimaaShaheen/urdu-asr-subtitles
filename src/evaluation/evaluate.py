"""Model se transcription lena: baseline ya LoRA adapter ke saath (Step 8)."""
import math

import numpy as np
import torch

SR = 16000
MAX_SEC = 29.0   # isse lambi audio barabar tukron mein kati jati hai


def load_model(model_name, adapter_dir=None, device="cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_name, language="urdu", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()           # adapter ko model mein jor do (tez inference)
    model.generation_config.forced_decoder_ids = None
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    return model.eval(), processor


@torch.no_grad()
def transcribe(model, processor, audios, batch_size=16, num_beams=1, device="cuda"):
    """Audio arrays ki list -> text ki list."""
    pieces, owner = [], []
    for i, y in enumerate(audios):
        y = np.asarray(y, dtype=np.float32)
        n = max(1, math.ceil(len(y) / (MAX_SEC * SR)))
        for part in np.array_split(y, n):
            pieces.append(part)
            owner.append(i)

    dtype = next(model.parameters()).dtype
    texts = []
    for s in range(0, len(pieces), batch_size):
        feats = processor.feature_extractor(
            pieces[s:s + batch_size], sampling_rate=SR, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        ids = model.generate(input_features=feats, language="urdu", task="transcribe",
                             num_beams=num_beams, max_new_tokens=225)
        texts += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)

    out = [""] * len(audios)
    for o, t in zip(owner, texts):
        out[o] = (out[o] + " " + t.strip()).strip()
    return out