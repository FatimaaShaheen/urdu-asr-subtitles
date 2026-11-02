# Glossary

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| ASR | Awaaz ko text mein badalna | Poora project |
| NLP | Computer ka insani zabaan samajhna | Urdu text cleaning, tokenization |
| ML | Data se computer ka khud seekhna | Model training |
| DL | ML jis mein bohat layers wala neural network ho | Whisper model |
| Dataset | Examples ka collection (audio + text) | FLEURS, Common Voice |
| Pipeline | Steps ki chain: audio, model, text, subtitle | Poora system |
| Frontend | Jo user dekhta hai | Streamlit app |
| Backend | Jo parde ke peeche kaam karta hai | FastAPI |
| API | Frontend aur backend ke baat karne ka tareeqa | /health, /transcribe |
| Database | Data ko sanbhal kar rakhne ki jagah | SQLite (Step 9) |
| Waveform | Awaaz ka waqt vs shiddat graph | Notebook 01 |
| Sampling rate | Ek second mein kitne samples (Whisper: 16kHz) | audio_utils.py |
| Spectrogram | Waqt vs frequency ki tasveer | Notebook 01 |
| Mel scale | Insani kaan ke mutabiq frequency scale | mel_spectrogram() |
| Mel-spectrogram | Whisper ka input (80 x frames) | mel_spectrogram() |
| MFCC | Mel-spectrogram ka compressed form | get_mfcc() |
| Resampling | Sampling rate badalna | load_audio() |
# Glossary

Har term: apne alfaaz mein matlab, aur project mein kahan aayi.

## Bunyadi

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| ASR | Awaaz ko text mein badalna | Poora project |
| NLP | Computer ka insani zabaan (text) samajhna | Urdu text cleaning, normalization |
| ML | Data se computer ka khud seekhna | Model training |
| DL | ML jis mein bohat layers wala neural network ho | Whisper model |
| Pipeline | Steps ki chain: audio, model, text, subtitle | Poora system |
| Frontend | Jo user dekhta hai | Streamlit app |
| Backend | Parde ke peeche ka code | FastAPI |
| API | Frontend aur backend ke baat karne ka tareeqa | /health, /audio-info |
| Database | Data sanbhal kar rakhne ki jagah | SQLite (baad mein) |

## Audio (Step 2)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Waveform | Awaaz ka waqt vs shiddat ka graph | Notebook 01 |
| Sampling rate | Ek second mein kitne samples (Whisper: 16 kHz) | audio_utils.py |
| Amplitude | Awaaz kitni zor ki hai | Waveform ki unchai |
| Mono / Stereo | 1 channel / 2 channels | load_audio() mono banata hai |
| Resampling | Sampling rate badalna | load_audio() |
| Spectrogram | Waqt vs frequency ki tasveer | Notebook 01 |
| Mel scale | Insani kaan ke mutabiq frequency scale | mel_spectrogram() |
| Mel-spectrogram | Whisper ka input (80 x frames) | mel_spectrogram() |
| MFCC | Mel-spectrogram ka compressed form | get_mfcc() |

## Dataset (Step 3)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Dataset | Examples ka collection (audio + text) | FLEURS Urdu |
| Split | Dataset ke hisse: train, validation, test | load_fleurs_urdu() |
| Train set | Jis se model seekhta hai (2109 examples, 7.03 ghante) | Fine-tuning |
| Validation set | Training ke dauran check karne ke liye (267 examples) | Fine-tuning |
| Test set | Aakhri imtihan (299 examples), sirf akhir mein | Evaluation |
| Transcription | Audio mein jo bola gaya, uska text | Dataset column |
| Label | Sahi jawab jo model ko seekhne ke liye diya jata hai | Transcription |
| Corpus | Text ka bara collection | Dataset ka text |
| Vocabulary | Data mein maujood alag alag characters/alfaaz ki list | char_vocab() |
| Data leakage | Test ka data train mein chala jana (score jhootha ho jata hai) | Overlap check |
| Streaming | Poora dataset download kiye baghair parhna | Aage use hoga |

## Evaluation (Step 4)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Inference | Trained model se prediction lena (training nahi) | Notebook 03 |
| Baseline | Shuru ka score, jis se baad mein behtari naapte hain | Notebook 03 |
| Reference | Sahi (asal) text | Test transcription |
| Hypothesis | Model ka likha hua text | Whisper ka output |
| WER | (badle + hataye + barhaye alfaaz) / asal alfaaz | metrics.py |
| CER | WER jaisa, lekin characters par | metrics.py |
| Normalization | Dono texts ko ek jaisa banana (punctuation, diacritics hataana) | normalize_urdu() |
| Diacritics | Zer, zabar, pesh jaise nishaan | normalize_urdu() |