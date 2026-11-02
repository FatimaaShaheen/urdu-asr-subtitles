"""Dataset load karne aur samajhne ke functions."""
from collections import Counter

from datasets import Audio, load_dataset

TARGET_SR = 16000


def load_fleurs_urdu(split="train", sampling_rate=TARGET_SR):
    """Google FLEURS ka Urdu (ur_pk) hissa. split: train / validation / test"""
    ds = load_dataset("google/fleurs", "ur_pk", split=split, trust_remote_code=True)
    # audio ko 16kHz par cast karo (Whisper ke liye)
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def total_hours(ds):
    """Kitne ghante audio hai (audio decode kiye baghair)."""
    return sum(ds["num_samples"]) / TARGET_SR / 3600


def char_vocab(ds, column="transcription"):
    """Text mein kaun se characters kitni baar aaye."""
    counter = Counter()
    for text in ds[column]:
        counter.update(text)
    return counter


def summary(ds, name=""):
    return {
        "name": name,
        "num_examples": len(ds),
        "hours": round(total_hours(ds), 2),
        "columns": ds.column_names,
    }