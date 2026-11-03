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
| Resampling | Sampling rate badalna | load_audio(), Step 6 |
| Spectrogram | Waqt vs frequency ki tasveer | Notebook 01 |
| Mel scale | Insani kaan ke mutabiq frequency scale | mel_spectrogram() |
| Mel-spectrogram | Whisper ka input (80 x frames) | mel_spectrogram() |
| MFCC | Mel-spectrogram ka compressed form | get_mfcc() |

## Dataset (Step 3)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Dataset | Examples ka collection (audio + text) | FLEURS Urdu |
| Split | Dataset ke hisse: train, validation, test | load_fleurs_urdu() |
| Train set | Jis se model seekhta hai | Fine-tuning |
| Validation set | Training ke dauran check karne ke liye | Fine-tuning |
| Test set | Aakhri imtihan, sirf akhir mein | Evaluation |
| Transcription | Audio mein jo bola gaya, uska text | Dataset column |
| Label | Sahi jawab jo model ko seekhne ke liye diya jata hai | Transcription |
| Corpus | Text ka bara collection | Dataset ka text |
| Vocabulary | Data mein maujood alag alag characters/alfaaz ki list | char_vocab() |
| Data leakage | Test ka data train mein chala jana (score jhootha ho jata hai) | Overlap checks |
| Streaming | Poora dataset download kiye baghair parhna | Aage use ho sakta hai |

## Evaluation (Step 4)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Inference | Trained model se prediction lena (training nahi) | Notebook 03 |
| Baseline | Shuru ka score, jis se baad mein behtari naapte hain | Notebook 03 |
| Reference | Sahi (asal) text | Test transcription |
| Hypothesis | Model ka likha hua text | Whisper ka output |
| WER | (badle + hataye + barhaye alfaaz) / asal alfaaz, kam behtar | metrics.py |
| CER | WER jaisa, lekin characters par | metrics.py |
| Diacritics | Zer, zabar, pesh jaise nishaan | clean_text.py |
| Model size (base/small) | Model ke parameters ki tadaad; bara = behtar lekin slow | Notebook 03 |
| Accuracy vs speed | Behtar score ke liye zyada waqt lagta hai | base 117s, small 280s |

## Text aur Tokenization (Step 5)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Text normalization | Ek hi cheez ke alag likhne ke tareeqon ko ek bana dena | normalize_for_wer() |
| Unicode | Har harf ko ek number dene ka international nizam | Urdu characters |
| NFC | Unicode ki standard shakal, harf aur nishan ek saath jurte hain | basic_clean() |
| Zero-width character | Chhupa hua character jo nazar nahi aata | basic_clean() |
| Tatweel | Harf ko kheenchne wala nishan (ـ) | basic_clean() |
| Spelling variants | Ek lafz ke jaiz alag tareeqe (کئے / کیے) | SPELLING dictionary |
| Token | Text ka chhota tukra jo model samajhta hai | Notebook 04 |
| Tokenization | Text ko tokens mein todna | WhisperTokenizer |
| BPE | Aksar saath aane wale hisson ko jor kar token banana | Whisper ka tokenizer |
| Special tokens | Khaas ishare (zabaan, kaam, shuru, khatam) | <\|ur\|>, <\|transcribe\|> |
| Tokens per lafz | Ek lafz kitne tokens leta hai (Urdu ~2.7, English ~1.0) | Notebook 04 |
| Unit test | Chhota code jo check kare ke function sahi chalta hai | tests/ |

## Extra datasets (Step 6)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Data merging | Alag datasets ko ek jaisi shakal mein la kar jorna | fleurs_to_common(), concatenate_datasets |
| Schema | Columns ke naam aur qisam (audio, text, duration...) | extra_datasets.py |
| Domain shift | Training aur asal istemal ke data mein farq (kitabi vs TV ki zabaan) | FLEURS vs UrduSpeech |
| Noisy labels | Kuch ghalat labels jo training mein bhi shamil ho jayen | UrduSpeech corpus |
| Data filtering | Kharab examples hata dena (bohat chhoti/lambi, khali text) | keep_example() |
| Characters per second | Bolne ki raftaar, label audio se match karta hai ya nahi ka ishara | filters.py |
| Speaker leakage | Ek hi bolne wala train aur test dono mein aa jana | split_speakers() |
| Benchmark | Insani check shuda saaf test set | UrduSpeech benchmark |
| Ground truth | Insani tasdeeq shuda sahi text | Benchmark ka label |
| Attribution | CC-BY license mein dataset ke authors ka naam dena | README, datasets.md |
| Data card | Dataset ke source, license aur masle likhne ki file | docs/datasets.md |
| Embedding (audio ko dataset ke andar rakhna) | Audio bytes ko dataset ki file mein hi mehfooz karna taake raasta khone par bhi chale | embed_audio() |

## Training (Step 7)

| Term | Mere alfaaz mein | Project mein kahan |
|---|---|---|
| Fine-tuning | Pehle se trained model ko apne data par thora aur seekhana | lora_whisper.py |
| LoRA | Poora model nahi badalte, chhote adapters jor kar sirf unhein train karte hain | LoraConfig, 5% parameters |
| Rank (r) | Adapter kitna bara hai | r = 32 |
| Epoch | Poore training data ka ek chakkar | A ~4.6, B ~2.5 |
| Batch size | Ek baar mein kitne examples | 8 |
| Gradient accumulation | Chhote batches ke gradients jama karke ek bara step lena | 2, yani effective batch 16 |
| Learning rate | Har step mein model kitna badle | 0.0005 |
| Warmup | Shuru mein learning rate dheere dheere barhana | 50 steps |
| Loss | Model ki ghalti ka number (kam behtar) | Train/val loss |
| Optimizer (AdamW) | Weights update karne ka tareeqa | Trainer ka default |
| fp16 / mixed precision | Aadhe-size numbers se tez aur kam memory | fp16=True |
| Checkpoint | Training ke beech mein save kiya hua model | save_steps |
| Overfitting | Training data yaad kar lena, naye data par kharab hona | A ka val loss barha |
| Data augmentation | Audio ko thora badal kar data barhana | augment.py |
| SNR | Awaaz aur shor ka nisbat (kam = zyada shor) | add_noise() |
| SpecAugment | Spectrogram ke tukron ko chhupa dena | apply_spec_augment |
| Adapter | LoRA ke chhote trained weights (chhoti file) | runs/A/adapter |
- [x] Step 8: Evaluation aur error analysis