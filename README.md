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
├── notebooks/      Colab notebooks (01 se 05)
├── src/
│   ├── data/       audio_utils.py, load_dataset.py, clean_text.py,
│   │               filters.py, extra_datasets.py
│   ├── evaluation/ metrics.py
│   ├── training/
│   └── inference/
├── tests/          test_clean_text.py, test_filters.py
├── docs/           glossary.md, datasets.md, results/
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

Colab: har notebook ke pehle cell mein repo clone/pull hota hai. Bara data Google Drive (`MyDrive/urdu_asr`) mein rehta hai, GitHub par nahi.

## Progress

- [x] Step 1: Setup, GitHub, backend + frontend skeleton
- [x] Step 2: Audio basics (waveform, mel-spectrogram, MFCC)
- [x] Step 3: Dataset load aur exploration (FLEURS Urdu)
- [x] Step 4: Whisper baseline aur WER
- [x] Step 5: Urdu text cleaning aur tokenization
- [x] Step 6: Extra datasets (UrduSpeech subset + benchmark)
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
| whisper-small + LoRA, A (FLEURS only) | - | - | Step 7/8 ke baad |
| whisper-small + LoRA, B (FLEURS + UrduSpeech) | - | - | Step 7/8 ke baad |

## Nateeje (ab tak)

### Step 4
- `small` ne `base` se WER taqriban 15 points kam kiya, lekin taqriban 2.4 guna slow hai (accuracy vs speed).

### Step 5
- Nayi normalization se WER sirf 0.41 points kam hui (0.3672 se 0.3631). Yani zyada tar ghaltiyan **asal** hain, sirf likhne ke tareeqe ka farq nahi.
- Model ki asal kamzoriyan: adad (`403` ko alfaaz mein likhna), naam aur shehron ke naam, English/science ke alfaaz, lafz ki hadd.
- Adad ka masla (digits vs alfaaz) abhi hal nahi hua, Step 8 mein dekha jayega.
- Whisper ka tokenizer Urdu ke liye taqriban 2.7 tokens per lafz leta hai (English ke liye 1.0).

### Step 6
- Extra data sirf taqriban 1.5 ghante (1700 clips) ka bana, yani training data mein ~20% izafa. Is liye experiment B mein bara farq ki ummeed nahi, aur "farq nahi aaya" bhi ek jaiz nateeja hoga.
- `news` aur `interview` categories ke 993 clips ke labels jsonl se match nahi hue, is liye shamil nahi.
- Benchmark ke 7 jumle merged train mein mile, unhein benchmark se hata diya (858 se 851).
- Limitation: UrduSpeech benchmark aur corpus ek hi source dataset se hain. Text overlap hata diya gaya, lekin speaker/video overlap