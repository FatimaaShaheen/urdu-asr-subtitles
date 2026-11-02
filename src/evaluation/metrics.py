"""Urdu text normalization aur WER/CER."""
import re
import jiwer

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCT = re.compile(r"[۔،؟!?,.:;\"'()\[\]{}\-–—…٪%]")


def normalize_urdu(text: str) -> str:
    text = text.replace("ي", "ی").replace("ى", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("ه", "ہ").replace("ۃ", "ہ")
    text = _DIACRITICS.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(refs, hyps):
    r = [normalize_urdu(x) for x in refs]
    h = [normalize_urdu(x) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a]
    r, h = zip(*pairs)
    return {
        "wer": round(jiwer.wer(list(r), list(h)), 4),
        "cer": round(jiwer.cer(list(r), list(h)), 4),
        "n": len(r),
    }