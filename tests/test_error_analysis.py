import numpy as np

from src.evaluation.error_analysis import (
    digit_stats, hallucination_suspects, paired_bootstrap,
    sentence_errors, top_confusions,
)

REFS = ["a b c", "d e"]
HYPS = ["a x c", "d"]


def test_sentence_errors():
    errs, nref = sentence_errors(REFS, HYPS)
    assert errs.tolist() == [1, 1]
    assert nref.tolist() == [3, 2]


def test_top_confusions():
    assert top_confusions(REFS, HYPS, 5)[0] == (("b", "x"), 1)


def test_digit_stats():
    assert digit_stats(["سکوٹر 403 بس"], ["سکوٹر چار سو بس"]) == (1, 1)


def test_hallucination():
    assert hallucination_suspects(["a b"], ["x " * 20]) == [0]


def test_bootstrap_clear_difference():
    r = paired_bootstrap(np.zeros(50), np.ones(50), np.full(50, 3))
    assert r["ci_low"] > 0


def test_bootstrap_identical():
    r = paired_bootstrap(np.ones(50), np.ones(50), np.full(50, 3))
    assert r["ci_low"] <= 0 <= r["ci_high"]