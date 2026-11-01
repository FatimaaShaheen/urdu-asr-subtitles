"""Audio load karne, samajhne aur plot karne ke functions."""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

TARGET_SR = 16000  # Whisper ka sampling rate


def load_audio(path, sr=TARGET_SR):
    """Audio load karo, mono banao, 16kHz par resample karo."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def audio_info(y, sr):
    """Audio ke bare mein bunyadi maloomat."""
    return {
        "duration_sec": round(len(y) / sr, 2),
        "sample_rate": sr,
        "num_samples": int(len(y)),
        "max_amplitude": round(float(np.max(np.abs(y))), 4),
    }


def mel_spectrogram(y, sr, n_mels=80):
    """Whisper jaisi settings: 25ms window (400), 10ms hop (160), 80 mel bins."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels
    )
    return librosa.power_to_db(S, ref=np.max)  # decibel scale


def get_mfcc(y, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Waqt (seconds)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_mel(y, sr):
    S_db = mel_spectrogram(y, sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-spectrogram (Whisper ka input)")
    fig.tight_layout()
    return fig


def plot_mfcc(y, sr):
    m = get_mfcc(y, sr)
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(m, x_axis="time", ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC")
    fig.tight_layout()
    return fig