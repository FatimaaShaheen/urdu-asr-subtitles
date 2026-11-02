"""Urdu text cleaning (NLP)."""
import re
import unicodedata

# Chhupe hue characters (zero-width) aur tatweel (kheencha hua harf)
_ZERO_WIDTH = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
_TATWEEL = "\u0640"

# Zer, zabar, pesh wagera
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

# Urdu aur English punctuation
_PUNCT = re.compile(r"[۔،؟؛!?,.:;\"'“”‘’()\[\]{}\-–—…٪%/]")

# Ek jaise harf jo alag code se likhe ja sakte hain
CHAR_MAP = {
    "ي": "ی", "ى": "ی",   # Arabic ye -> Urdu ye
    "ك": "ک",              # Arabic kaaf -> Urdu kaaf
    "ه": "ہ", "ۃ": "ہ",    # alag alag "he" -> ہ
}

# Arabic-Indic aur Urdu digits -> 0-9
_digits = {}
for _i in range(10):
    _digits[chr(0x0660 + _i)] = str(_i)
    _digits[chr(0x06F0 + _i)] = str(_i)
_DIGIT_TABLE = str.maketrans(_digits)

# Likhne ke jaiz alag tareeqe -> ek tareeqa.
# Yeh list error analysis dekh kar barhti rahegi.
SPELLING = {
    "کئے": "کیے",
    "لئے": "لیے",
    "چاہئے": "چاہیے",
    "کیوں کہ": "کیونکہ",
}


def basic_clean(text: str) -> str:
    """Halki safai: matlab nahi badalta (punctuation rehta hai)."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text).replace(_TATWEEL, "")
    text = "".join(CHAR_MAP.get(c, c) for c in text)
    text = text.translate(_DIGIT_TABLE)
    text = text.lower()                      # English alfaaz ke liye
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_wer(text: str) -> str:
    """Zyada safai: WER/CER naapne ke liye."""
    text = basic_clean(text)
    text = _DIACRITICS.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    padded = " " + text + " "
    for old, new in SPELLING.items():
        padded = padded.replace(" " + old + " ", " " + new + " ")
    return padded.strip()