# Urdu ASR aur Subtitle Generator

Whisper ko Urdu par fine-tune karke audio/video se Urdu subtitles (.srt) banane ka semester project.

## Maqsad

Audio/Video -> Speech Recognition -> Urdu text -> Timestamps -> Subtitle file (.srt), saath mein sab NLP/ML/DL terms ko project ke zariye seekhna.

## Tech stack (sab free)

Python, PyTorch, Hugging Face (transformers, datasets), OpenAI Whisper (open-source), jiwer, librosa, FastAPI, Streamlit, SQLite, Google Colab, GitHub.

## Dataset

Google FLEURS (Urdu, `ur_pk`)

| Split | Examples | Ghante |
|---|---|---|
| Train | 2109 | 7.03 |
| Validation | 267 | 0.76 |
| Test | 299 | 0.81 |

Note: Hugging Face `datasets` ka version 4 purani loading scripts support nahi karta, is liye project mein `datasets<4` use hota hai.

## Folder structure

```
urdu-asr-subtitles/
├── backend/        FastAPI
├── frontend/       Streamlit
├── notebooks/      Colab notebooks
├── src/
│   ├── data/       audio_utils.py, load_dataset.py
│   ├── evaluation/ metrics.py
│   ├── training/
│   └── inference/
├── docs/           glossary.md, results/
├── configs/
├── tests/
├── requirements.txt
└── README.md
```

## Chalane ka tareeqa

Local (project folder se, venv active):

```
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

Colab: har notebook ke pehle cell mein repo clone/pull hota hai.

## Progress

- [x] Step 1: Setup, GitHub, backend + frontend skeleton
- [x] Step 2: Audio basics (waveform, mel-spectrogram, MFCC)
- [x] Step 3: Dataset load aur exploration (FLEURS Urdu)
- [x] Step 4: Whisper baseline aur WER
- [ ] Step 5: Urdu text cleaning aur tokenization
- [ ] Step 6: Extra datasets
- [ ] Step 7: Augmentation + LoRA fine-tuning
- [ ] Step 8: Evaluation aur error analysis
- [ ] Step 9: LLM post-processing
- [ ] Step 10: Subtitles (.srt) aur translation
- [ ] Step 11: Backend + database + frontend + deployment

## Results

FLEURS Urdu test set (299 jumle). Scoring `normalize_urdu` (metrics.py, version 1) se hui.

| Model | WER | CER | Note |
|---|---|---|---|
| whisper-base (baseline) | 0.5155 | 0.2153 | Step 4, 117 sec |
| whisper-small (baseline) | 0.3672 | 0.1373 | Step 4, 280 sec |
| whisper-small (fine-tuned) | - | - | Step 7 ke baad |

### Step 4 se nateeje

- `small` ne `base` se WER taqriban 15 points kam kiya, lekin taqriban 2.4 guna slow hai (accuracy vs speed).
- Ghaltiyon ki qismein: milti julti awaaz wali spelling ghalat, naam aur ghair-zabaan alfaaz (جیسے "اتاوا کناڈا"), lafz ki hadd, adad ka format (`70` vs ستر), aur likhne ke jaiz alag tareeqe (کئے/کیے) jo scorer ne ghalat gine.
- Aakhri qism Step 5 (text normalization) mein theek hogi, is liye Step 5 ke baad baseline dobara naapa jayega.