"""UrduSpeech: chhota subset download, 16kHz mono mein badalna, FLEURS ke saath ek schema."""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import soundfile as sf
from datasets import Audio, Dataset
from huggingface_hub import HfApi, hf_hub_download

from src.data.clean_text import normalize_for_wer
from src.data.filters import keep_example

REPO_ID = "ASLP-lab/UrduSpeech"
TARGET_SR = 16000
CORPUS_STD = "corpus/US-Std"
BENCH_STD = "benchmark/US-benchmark-Std"
# corpus: <cat>_final_transcription.jsonl, benchmark: clean_transcription.jsonl
JSONL_SUFFIXES = ("_final_transcription.jsonl", "clean_transcription.jsonl")


def list_repo():
    """Repo ki sab files ke naam (download nahi)."""
    return HfApi().list_repo_files(REPO_ID, repo_type="dataset")


def categories(files, root, fmt="short"):
    """Kisi root (corpus/benchmark) ke fmt (short/long) mein categories."""
    prefix = f"{root}/{fmt}/"
    return sorted({f.split("/")[3].lower() for f in files
                   if f.startswith(prefix) and len(f.split("/")) > 4})


def select(files, root, fmt, cats, n_per_cat, seed=42):
    """Har category se random n wav files + us category ki jsonl file."""
    prefix = f"{root}/{fmt}/"
    sub = [f for f in files if f.startswith(prefix)]
    rnd = random.Random(seed)
    chosen = []
    for cat in cats:
        in_cat = [f for f in sub if f.split("/")[3].lower() == cat.lower()]
        wavs = sorted(f for f in in_cat if f.lower().endswith(".wav"))
        rnd.shuffle(wavs)
        chosen += wavs[:n_per_cat]
        chosen += [f for f in in_cat if f.endswith(JSONL_SUFFIXES)]
    return chosen


def download(paths, local_dir, workers=8, retries=3):
    """Sirf chuni hui files download karo (poora 101 GB nahi)."""
    def _get(p):
        for attempt in range(retries):
            try:
                return hf_hub_download(REPO_ID, p, repo_type="dataset",
                                       local_dir=local_dir)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_get, paths))


def _load_meta(local_dir, paths):
    """Sab jsonl se: audio file ka naam (chhote harf) -> record."""
    meta = {}
    for p in paths:
        if p.endswith(JSONL_SUFFIXES):
            with open(os.path.join(local_dir, p), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        meta[r["Audio_Clip"].lower()] = r
    return meta


def _convert(src, dst):
    y, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    sf.write(dst, y, TARGET_SR, subtype="PCM_16")


def build_dataset(local_dir, paths, out_dir, source, workers=8):
    """Download shuda files -> filter -> 16kHz mono wav -> HF Dataset."""
    meta = _load_meta(local_dir, paths)
    kept = []
    stats = {"no_label": 0, "bad_audio": 0, "filtered": 0}

    for p in paths:
        if not p.lower().endswith(".wav"):
            continue
        src = os.path.join(local_dir, p)
        r = meta.get(os.path.basename(p).lower())
        if r is None or not os.path.exists(src):
            stats["no_label"] += 1
            continue
        try:
            dur = sf.info(src).duration           # asal lambai audio se
        except Exception:
            stats["bad_audio"] += 1
            continue
        raw = r.get("ground_truth") or r.get("Transcription") or ""
        text = normalize_for_wer(raw)
        if not keep_example(text, dur):
            stats["filtered"] += 1
            continue
        cat = p.split("/")[3].lower()
        name = os.path.splitext(os.path.basename(p))[0].lower() + ".wav"
        dst = os.path.join(out_dir, source, cat, name)
        spk = f"{source}_{cat}_{r.get('Speaker_id', 'spk')}"
        kept.append((src, dst, text, dur, spk, cat))

    for _, dst, *_ in kept:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda k: _convert(k[0], k[1]), kept))

    rows = {
        "audio":    [k[1] for k in kept],
        "text":     [k[2] for k in kept],
        "duration": [float(k[3]) for k in kept],
        "source":   [source] * len(kept),
        "speaker":  [k[4] for k in kept],
        "category": [k[5] for k in kept],
    }
    ds = Dataset.from_dict(rows).cast_column("audio", Audio(sampling_rate=TARGET_SR))
    return ds, stats


def fleurs_to_common(ds):
    """FLEURS ko wahi columns do jo UrduSpeech ke hain."""
    n = len(ds)
    texts = [normalize_for_wer(t) for t in ds["transcription"]]
    durs = [x / TARGET_SR for x in ds["num_samples"]]
    ds = ds.select_columns(["audio"])
    return (ds.add_column("text", texts)
              .add_column("duration", durs)
              .add_column("source", ["fleurs"] * n)
              .add_column("speaker", ["fleurs"] * n)
              .add_column("category", ["fleurs"] * n))