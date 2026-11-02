from src.data.clean_text import basic_clean, normalize_for_wer


def test_spelling_variants_match():
    assert normalize_for_wer("لئے") == normalize_for_wer("لیے")
    assert normalize_for_wer("کیوں کہ") == "کیونکہ"


def test_punctuation_removed():
    assert normalize_for_wer("یہ کتاب ہے۔") == "یہ کتاب ہے"


def test_digits_unified():
    assert basic_clean("٧٠") == "70"


def test_spaces_collapsed():
    assert normalize_for_wer("یہ   کتاب") == "یہ کتاب"