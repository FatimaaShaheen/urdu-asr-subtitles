from src.data.filters import keep_example, latin_ratio, split_speakers


def test_good_example_kept():
    assert keep_example("یہ ایک جملہ ہے", 2.0)


def test_empty_text_dropped():
    assert not keep_example("   ", 3.0)


def test_too_short_or_long_dropped():
    assert not keep_example("یہ ایک جملہ ہے", 0.5)
    assert not keep_example("یہ ایک جملہ ہے", 40.0)


def test_english_heavy_dropped():
    assert latin_ratio("hello world") == 1.0
    assert not keep_example("hello world this is english", 3.0)


def test_speaker_split_is_deterministic():
    spk = [f"s{i}" for i in range(100)]
    assert split_speakers(spk) == split_speakers(spk)
    assert len(split_speakers(spk, 0.1)) == 10