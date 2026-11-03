# Urdu ASR aur Subtitle Generator

Whisper ko Urdu par fine-tune karke audio/video se Urdu subtitles (.srt) banane ka semester project.

## Maqsad

Audio/Video -> Speech Recognition -> Urdu text -> Timestamps -> Subtitle file (.srt), saath mein sab NLP/ML/DL terms ko project ke zariye seekhna.

## Tech stack (sab free)

Python, PyTorch, Hugging Face (transformers, datasets, peft), OpenAI Whisper (open-source), jiwer, librosa, FastAPI, Streamlit, SQLite, Google Colab, GitHub.

## Datasets

Tafseel `docs/datasets.md` mein.

| Dataset | Use | Examples | Ghante |
|---|---|---|---|
| FLEURS Urdu train | Training | 2109 (2103 after 30s filter) | 7.03 |
| FLEURS Urdu validation | Validation | 267 | 0.76 |
| FLEURS Urdu test | **Final evaluation** | 299 | 0.81 |
| UrduSpeech corpus subset (vlogs, podcast, roadside_interview) | Training (1700) + validation (60) | 1760 | 1.61 |
| UrduSpeech benchmark (insani labels) | **Final evaluation** | 851 | 1.01 |

Merged training set: 3803 examples, 8.53 ghante (fleurs 2103 + urduspeech 1700).

Notes:
- Hugging Face `datasets` ka version 4 purani loading scripts support nahi karta, is liye project mein `datasets<4` use hota hai.
- UrduSpeech (ASLP-lab, CC-BY-4.0) ke corpus labels model-generated hain (noisy labels), benchmark labels insani check shuda hain.
- FLEURS test aur UrduSpeech benchmark kabhi training mein shamil nahi.

## Folder structure

```
urdu-asr-subtitles/
├── backend/        FastAPI
├── frontend/       Streamlit
├── notebooks/      Colab notebooks (01 se 06b)
├── src/
│   ├── data/       audio_utils.py, load_dataset.py, clean_text.py,
│   │               filters.py, extra_datasets.py, augment.py
│   ├── evaluation/ metrics.py
│   ├── training/   lora_whisper.py
│   └── inference/
├── tests/          test_clean_text.py, test_filters.py, test_augment.py
├── configs/        train_config.yaml
├── docs/           glossary.md, datasets.md, results/
├── requirements.txt
└── README.md
```

## Chalane ka tareeqa

Local (project folder se, venv active):

```
python -m pytest tests
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

Colab: har notebook ke pehle cell mein repo clone/pull hota hai. Bara data aur trained adapters Google Drive (`MyDrive/urdu_asr`) mein rehte hain, GitHub par nahi.

## Progress

- [x] Step 1: Setup, GitHub, backend + frontend skeleton
- [x] Step 2: Audio basics (waveform, mel-spectrogram, MFCC)
- [x] Step 3: Dataset load aur exploration (FLEURS Urdu)
- [x] Step 4: Whisper baseline aur WER
- [x] Step 5: Urdu text cleaning aur tokenization
- [x] Step 6: Extra datasets (UrduSpeech subset + benchmark)
- [x] Step 7: Augmentation + LoRA fine-tuning
- [ ] Step 8: Evaluation aur error analysis
- [ ] Step 9: LLM post-processing
- [ ] Step 10: Subtitles (.srt) aur translation
- [ ] Step 11: Backend + database + frontend + deployment

## Results

### Baseline (FLEURS Urdu test, 299 jumle)

Norm v1 = Step 4 wali normalization, norm v2 = Step 5 wali (`clean_text.normalize_for_wer`). Alag normalization ke scores seedhe compare nahi hote.

| Model | WER | CER | Note |
|---|---|---|---|
| whisper-base (baseline, norm v1) | 0.5155 | 0.2153 | Step 4, 117 sec |
| whisper-small (baseline, norm v1) | 0.3672 | 0.1373 | Step 4, 280 sec |
| whisper-small (baseline, norm v2) | 0.3631 | 0.1362 | Step 5, wahi predictions |

### Fine-tuning (Step 7): validation set (100 clips: 60 FLEURS + 40 UrduSpeech)

LoRA (r=32, alpha=64), 600 steps, effective batch 16, lr 5e-4, augmentation sirf train par. Dono experiments ka validation set ek jaisa hai. Yeh **test score nahi** hai, final comparison Step 8 mein hoga.

| Experiment | Train clips | Epochs | Val loss (600) | Val WER (200 / 400 / 600) | Val CER (600) |
|---|---|---|---|---|---|
| A: FLEURS only | 2103 (6.98 h) | ~4.6 | 0.938 | 0.420 / 0.584 / 0.442 | 0.220 |
| B: FLEURS + UrduSpeech | 3803 (8.53 h) | ~2.5 | 0.678 | 0.441 / 0.474 / 0.377 | 0.175 |

Trainable parameters: 12.98M (5.09% of 254.7M). Training time: A 1670 sec, B 1700 sec (T4).

## Nateeje (ab tak)

### Step 4
- `small` ne `base` se WER taqriban 15 points kam kiya, lekin taqriban 2.4 guna slow hai (accuracy vs speed).

### Step 5
- Nayi normalization se WER sirf 0.41 points kam hui (0.3672 se 0.3631). Yani zyada tar ghaltiyan **asal** hain.
- Model ki asal kamzoriyan: adad (`403` ko alfaaz mein likhna), naam aur shehron ke naam, English/science ke alfaaz, lafz ki hadd.
- Whisper ka tokenizer Urdu ke liye taqriban 2.7 tokens per lafz leta hai (English ke liye 1.0).

### Step 6
- Extra data sirf taqriban 1.5 ghante (1700 clips) ka bana, yani training data mein ~20% izafa.
- `news` aur `interview` categories ke 993 clips ke labels jsonl se match nahi hue, shamil nahi.
- Benchmark ke 7 jumle train mein mile, unhein benchmark se hata diya (858 se 851).
- Limitation: UrduSpeech benchmark aur corpus ek hi source se hain; speaker/video overlap ki tasdeeq nahi hui.

### Step 7
- A mein overfitting: training loss 0.41 se 0.15 gira, validation loss 0.92 se 0.94 barha. B mein validation loss lagataar gira (0.758 se 0.678).
- Validation WER A mein bohat uchhalta raha (0.42, 0.58, 0.44), kyunke validation sirf 100 clips ka hai.
- B ka validation set mein faida expected hai (validation mein UrduSpeech clips hain). FLEURS test par asal farq Step 8 mein.