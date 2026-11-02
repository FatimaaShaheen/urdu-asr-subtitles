"""UrduSpeech ka chhota subset download karna aur FLEURS ke saath ek schema mein lana."""
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from datasets import Audio, Dataset
from huggingface_hub import HfApi, hf_hub_download

from src.data.clean_text import normalize_for_wer
from src.data.filters import keep_example

REPO_ID = "ASLP-lab/UrduSpeech"
TARGET_SR = 16000


def list_repo(subset="UrduSpeech-Std", fmt="short"):
    """Repo ki files ki list (sirf naam, download nahi)."""
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"{subset}/{fmt}/"
    return [f for f in files if f.startswith(prefix)]


def categories(files):
    return sorted({f.split("/")[2] for f in files if len(f.split("/")) > 3})


def pick(files, cats, n_per_cat, seed=42):
    """Har category se random n wav files + us category ki jsonl file."""
    rnd = random.Random(seed)
    chosen = []
    for cat in cats:
        wavs = sorted(f for f in files
                      if f.endswith(".wav") and f.split("/")[2] == cat)
        rnd.shuffle(wavs)
        chosen += wavs[:n_per_cat]
        chosen += [f for f in files
                   if f.split("/")[2] == cat
                   and f.endswith("_final_transcription.jsonl")]
    return chosen


def download(paths, local_dir, workers=16):
    """Sirf chuni hui files download karo (poora 101 GB nahi)."""
    def _get(p):
        return hf_hub_download(REPO_ID, p, repo_type="dataset", local_dir=local_dir)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_get, paths))


def build_subset(local_dir, paths):
    """Download shuda files se HF Dataset banao (filtering ke saath)."""
    # 1) sab jsonl se: audio file ka naam -> uska record
    meta = {}
    for root, _, names in os.walk(local_dir):
        for n in names:
            if n.endswith("_final_transcription.jsonl"):
                with open(os.path.join(root, n), encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        meta[r["Audio_Clip"]] = r

    # 2) har wav ke liye row banao
    rows = {"audio": [], "text": [], "duration": [],
            "source": [], "speaker": [], "category": []}
    dropped = 0
    for p in paths:
        if not p.endswith(".wav"):
            continue
        full = os.path.join(local_dir, p)
        r = meta.get(os.path.basename(p))
        if r is None or not os.path.exists(full):
            dropped += 1
            continue
        text = normalize_for_wer(r["Transcription"])
        dur = float(r["Duration_seconds"])
        if not keep_example(text, dur):
            dropped += 1
            continue
        cat = p.split("/")[2]
        rows["audio"].append(full)
        rows["text"].append(text)
        rows["duration"].append(dur)
        rows["source"].append("urduspeech")
        rows["speaker"].append(f"{cat}_{r['Speaker_id']}")
        rows["category"].append(cat)

    ds = Dataset.from_dict(rows).cast_column("audio", Audio(sampling_rate=TARGET_SR))
    return ds, dropped


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