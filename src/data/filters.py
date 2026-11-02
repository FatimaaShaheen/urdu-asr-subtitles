"""Data filtering aur speaker-wise split (pure Python)."""
import random
import re

_LATIN = re.compile(r"[A-Za-z]")


def latin_ratio(text: str) -> float:
    """Text mein English letters ka hissa (0 se 1)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return len(_LATIN.findall(text)) / len(chars)


def keep_example(text, duration, min_dur=1.0, max_dur=30.0,
                 min_cps=2.0, max_cps=25.0, max_latin=0.1):
    """True agar example training ke laiq hai.

    cps = characters per second (bolne ki raftaar). Bohat kam ya bohat zyada
    ho to aksar label audio se match nahi karta.
    """
    if not text or not text.strip():
        return False
    if not (min_dur <= duration <= max_dur):
        return False
    cps = len(text.replace(" ", "")) / duration
    if not (min_cps <= cps <= max_cps):
        return False
    return latin_ratio(text) <= max_latin


def split_speakers(speakers, holdout_frac=0.05, seed=42):
    """Kuch speakers alag nikaalo (inhein sirf test mein rakhenge)."""
    unique = sorted(set(speakers))
    random.Random(seed).shuffle(unique)
    k = max(1, int(len(unique) * holdout_frac))
    return set(unique[:k])