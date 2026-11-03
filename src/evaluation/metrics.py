"""WER aur CER (normalization clean_text.py se aati hai)."""
import jiwer

from src.data.clean_text import normalize_for_wer as normalize_urdu


def compute_metrics(refs, hyps):
    r = [normalize_urdu(x) for x in refs]
    h = [normalize_urdu(x) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a]   # khali reference hata do
    r, h = zip(*pairs)
    h = [x if x else "<empty>" for x in h]        # khali prediction se jiwer na ruke
    return {
        "wer": round(jiwer.wer(list(r), list(h)), 4),
        "cer": round(jiwer.cer(list(r), list(h)), 4),
        "n": len(r),
    }