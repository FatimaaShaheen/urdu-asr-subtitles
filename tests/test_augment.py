import numpy as np

from src.data.augment import add_noise, augment_waveform, change_speed


def _tone(n=16000):
    return (0.9 * np.sin(np.linspace(0, 200, n))).astype(np.float32)


def test_speed_changes_length():
    out = change_speed(_tone(), np.random.default_rng(0), low=1.25, high=1.25)
    assert abs(len(out) - 12800) <= 1


def test_noise_keeps_shape_and_changes_signal():
    y = _tone()
    out = add_noise(y, np.random.default_rng(0))
    assert out.shape == y.shape and not np.allclose(out, y)


def test_silence_gets_no_noise():
    z = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(add_noise(z, np.random.default_rng(0)), z)


def test_augment_range_and_dtype():
    out = augment_waveform(_tone(), np.random.default_rng(1), p=1.0)
    assert out.dtype == np.float32 and out.max() <= 1.0 and out.min() >= -1.0


def test_same_seed_same_result():
    a = augment_waveform(_tone(), np.random.default_rng(1), p=1.0)
    b = augment_waveform(_tone(), np.random.default_rng(1), p=1.0)
    assert np.array_equal(a, b)