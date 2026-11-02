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
├── notebooks/      Colab notebooks (01 se 04)
├── src/
│   ├── data/       audio_utils.py, load_dataset.py, clean_text.py
│   ├── evaluation/ metrics.py
│   ├── training/
│   └── inference/
├── tests/          test_clean_text.py
├── docs/           glossary.md, results/
├── configs/
├── requirements.txt
└── README.md
```

## Chalane ka tareeqa

Local (project folder se, venv active):

```
uvicorn backend.main:app --reload
streamlit run frontend/app.py
python -m pytest tests
```

Colab: har notebook ke pehle cell mein repo clone/pull hota hai.

## Progress

- [x] Step 1: Setup, GitHub, backend + frontend skeleton
- [x] Step 2: Audio basics (waveform, mel-spectrogram, MFCC)
- [x] Step 3: Dataset load aur exploration (FLEURS Urdu)
- [x] Step 4: Whisper baseline aur WER
- [x] Step 5: Urdu text cleaning aur tokenization
- [ ] Step 6: Extra datasets
- [ ] Step 7: Augmentation + LoRA fine-tuning
- [ ] Step 8: Evaluation aur error analysis
- [ ] Step 9: LLM post-processing
- [ ] Step 10: Subtitles (.srt) aur translation
- [ ] Step 11: Backend + database + frontend + deployment

## Results

FLEURS Urdu test set (299 jumle). Norm v1 = Step 4 wali normalization, norm v2 = Step 5 wali (`clean_text.normalize_for_wer`). Alag normalization ke scores seedhe compare nahi hote, aage sab models norm v2 se naape jayenge.

| Model | WER | CER | Note |
|---|---|---|---|
| whisper-base (baseline, norm v1) | 0.5155 | 0.2153 | Step 4, 117 sec |
| whisper-small (baseline, norm v1) | 0.3672 | 0.1373 | Step 4, 280 sec |
| whisper-small (baseline, norm v2) | 0.3631 | 0.1362 | Step 5, wahi predictions |
| whisper-small (fine-tuned) | - | - | Step 7 ke baad |

## Nateeje (ab tak)

### Step 4
- `small` ne `base` se WER taqriban 15 points kam kiya, lekin taqriban 2.4 guna slow hai (accuracy vs speed).

### Step 5
- Nayi normalization se WER sirf 0.41 points kam hui (0.3672 se 0.3631). Yani zyada tar ghaltiyan **asal** hain, sirf likhne ke tareeqe ka farq nahi.
- Model ki asal kamzoriyan: adad (`403` ko alfaaz mein likhna), naam aur shehron ke naam, English/science ke alfaaz, lafz ki hadd.
- Adad ka masla (digits vs alfaaz) abhi hal nahi hua, Step 8 mein dekha jayega.
- Characters ki tadaad safai se 129 se 82 hui.
- Whisper ka tokenizer Urdu ke liye taqriban 2.7 tokens per lafz leta hai (English ke liye 1.0), yani Urdu ka output lamba aur mehnga hota hai.