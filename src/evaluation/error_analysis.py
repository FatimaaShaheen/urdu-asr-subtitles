"""Error analysis: S/D/I, confusions, digits, hallucination, bootstrap (Step 8)."""
import re
from collections import Counter

import jiwer
import numpy as np

_DIGIT = re.compile(r"\d")


def _process(refs, hyps):
    return jiwer.process_words(list(refs), list(hyps))


def sentence_errors(refs, hyps):
    """Har jumle ki ghaltiyan (S+D+I) aur asal alfaaz ki tadaad."""
    out = _process(refs, hyps)
    errs, nref = [], []
    for chunks, ref_words in zip(out.alignments, out.references):
        e = 0
        for c in chunks:
            if c.type in ("substitute", "delete"):
                e += c.ref_end_idx - c.ref_start_idx
            elif c.type == "insert":
                e += c.hyp_end_idx - c.hyp_start_idx
        errs.append(e)
        nref.append(len(ref_words))
    return np.array(errs), np.array(nref)


def sdi_totals(refs, hyps):
    o = _process(refs, hyps)
    return {"wer": round(o.wer, 4), "hits": o.hits, "substitutions": o.substitutions,
            "deletions": o.deletions, "insertions": o.insertions}


def top_confusions(refs, hyps, k=15):
    """Sab se ziyada badle gaye (ref -> hyp) alfaaz."""
    o = _process(refs, hyps)
    cnt = Counter()
    for chunks, rw, hw in zip(o.alignments, o.references, o.hypotheses):
        for c in chunks:
            if c.type == "substitute":
                cnt[(" ".join(rw[c.ref_start_idx:c.ref_end_idx]),
                     " ".join(hw[c.hyp_start_idx:c.hyp_end_idx]))] += 1
    return cnt.most_common(k)


def top_deleted(refs, hyps, k=15):
    """Sab se ziyada chhut jane wale alfaaz."""
    o = _process(refs, hyps)
    cnt = Counter()
    for chunks, rw in zip(o.alignments, o.references):
        for c in chunks:
            if c.type == "delete":
                cnt.update(rw[c.ref_start_idx:c.ref_end_idx])
    return cnt.most_common(k)


def digit_stats(refs, hyps):
    """Reference ke adad wale alfaaz: kitne the, kitne ghalat nikle. (total, wrong)"""
    o = _process(refs, hyps)
    total = wrong = 0
    for chunks, rw in zip(o.alignments, o.references):
        for c in chunks:
            if c.type in ("equal", "substitute", "delete"):
                n = sum(1 for w in rw[c.ref_start_idx:c.ref_end_idx] if _DIGIT.search(w))
                total += n
                if c.type != "equal":
                    wrong += n
    return total, wrong


def hallucination_suspects(refs, hyps, ratio=2.0, min_hyp_words=8):
    """Jahan model ka text asal se bohat lamba ho (shak: dohrana ya banawati text)."""
    return [i for i, (r, h) in enumerate(zip(refs, hyps))
            if len(h.split()) >= min_hyp_words
            and len(h.split()) > ratio * max(1, len(r.split()))]


def paired_bootstrap(errs_x, errs_y, nref, n_boot=2000, seed=0):
    """diff = WER(y) - WER(x). Positive = x behtar. CI mein 0 na ho to farq asli."""
    rng = np.random.default_rng(seed)
    errs_x, errs_y, nref = map(np.asarray, (errs_x, errs_y, nref))
    n = len(nref)
    idx = rng.integers(0, n, size=(n_boot, n))
    denom = nref[idx].sum(axis=1)
    diff = errs_y[idx].sum(axis=1) / denom - errs_x[idx].sum(axis=1) / denom
    return {
        "wer_x": round(float(errs_x.sum() / nref.sum()), 4),
        "wer_y": round(float(errs_y.sum() / nref.sum()), 4),
        "diff": round(float(diff.mean()), 4),
        "ci_low": round(float(np.percentile(diff, 2.5)), 4),
        "ci_high": round(float(np.percentile(diff, 97.5)), 4),
    }