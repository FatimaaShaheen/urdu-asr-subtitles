"""Waveform augmentation (sirf numpy). Sirf training data par lagti hai."""
import numpy as np


def change_gain(y, rng, low_db=-6.0, high_db=6.0):
    db = rng.uniform(low_db, high_db)
    return y * (10 ** (db / 20))


def add_noise(y, rng, snr_low=10.0, snr_high=30.0):
    """SNR = awaaz aur shor ka nisbat (dB). Kam SNR = zyada shor."""
    power = float(np.mean(y ** 2))
    if power < 1e-10:                      # khamoshi par shor nahi jorna
        return y
    snr = rng.uniform(snr_low, snr_high)
    noise_power = power / (10 ** (snr / 10))
    return y + rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(y.dtype)


def change_speed(y, rng, low=0.9, high=1.1):
    """Raftaar badlo (pitch bhi thora badalti hai, tape ki tarah)."""
    rate = rng.uniform(low, high)
    new_len = max(1, int(round(len(y) / rate)))
    old_idx = np.linspace(0, len(y) - 1, num=new_len)
    return np.interp(old_idx, np.arange(len(y)), y).astype(y.dtype)


def augment_waveform(y, rng, p=0.5, speed=True, noise=True, gain=True):
    y = np.asarray(y, dtype=np.float32)
    if gain and rng.random() < p:
        y = change_gain(y, rng)
    if speed and rng.random() < p:
        y = change_speed(y, rng)
    if noise and rng.random() < p:
        y = add_noise(y, rng)
    return np.clip(y, -1.0, 1.0).astype(np.float32)