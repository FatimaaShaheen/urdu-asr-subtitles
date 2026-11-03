# Datasets

## 1. Google FLEURS (Urdu, ur_pk)
- Train 2109 (7.03 h), Validation 267 (0.76 h), Test 299 (0.81 h)
- Parhe hue jumle (Wikipedia), saaf labels
- Train mein 30 second se lambi 6 clips hata di gayi (Whisper ki hadd), 2103 bache
- Use: training (train), checking (validation), evaluation (test)

## 2. ASLP-lab/UrduSpeech: corpus subset
- Source: https://huggingface.co/datasets/ASLP-lab/UrduSpeech
- License: CC-BY-4.0 (attribution zaroori)
- Poora dataset 101 GB; hum ne sirf `corpus/US-Std/short` se chhota subset liya
- Chuni hui categories: vlogs, podcast, roadside_interview
  (news aur interview ke 993 clips ke labels jsonl se match nahi hue, is liye shamil nahi)
- Result: 1760 clips, 1.61 ghante
- Labels model-generated (Gemini 2.5 Pro), insani check shuda nahi (noisy labels)
- Filters: audio ki khud naapi hui lambai 1 se 30 sec, characters per second 2 se 25,
  English letters 10% se kam
- Audio 16 kHz mono mein badla gaya, aur Drive mein dataset ke andar embed kiya gaya
- Speaker-wise split: 17 speakers (5%) alag, 1700 clips train, 60 clips validation

## 3. UrduSpeech benchmark (US-benchmark-Std, short)
- Insani check shuda labels (`ground_truth`)
- Kuch clips ki jsonl duration galat hai, is liye lambai audio se naapi gayi
- 858 clips mein se 7 jumle hata diye jo merged train mein bhi the, bache 851 clips (1.01 ghante)
- Use: sirf evaluation (domain shift naapne ke liye), kabhi training mein nahi

## Merged training set
- FLEURS train (2103) + UrduSpeech train (1700) = 3803 clips, 8.53 ghante

## Limitations
- UrduSpeech benchmark aur corpus ek hi source dataset se hain. Text overlap hata diya
  gaya, lekin speaker ya video overlap ki tasdeeq nahi hui.
- Corpus ke labels model-generated hain, kuch ghalat labels honge.
- Extra data sirf ~1.5 ghante ka hai, isliye B experiment mein bara farq ki ummeed nahi.

## Citation
Haq, A. N. et al. (2026). UrduSpeech: A 156-Hour Urdu Speech Corpus with
12-Dimension Paralinguistic Annotations. arXiv:2605.17846

## Faisle
- FLEURS test aur UrduSpeech benchmark kabhi training mein shamil nahi
- Text normalization: clean_text.normalize_for_wer (train aur evaluation dono mein ek hi)
Training augmentation: gain (-6 se +6 dB), speed (0.9 se 1.1), noise (SNR 10 se 30 dB), SpecAugment. Sirf train par.