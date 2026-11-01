import os
import sys
import tempfile

import requests
import streamlit as st

# project root ko path mein daalo taake src import ho
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.audio_utils import load_audio, plot_mel, plot_mfcc, plot_waveform

API_URL = "http://127.0.0.1:8000"

st.title("Urdu ASR aur Subtitle Generator")

tab1, tab2 = st.tabs(["Backend check", "Audio explorer (Step 2)"])

with tab1:
    if st.button("Backend check karo"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(f"Backend se connect nahi hua: {e}")

with tab2:
    up = st.file_uploader("Audio upload karo", type=["wav", "mp3", "flac", "ogg"])
    if up is not None:
        st.audio(up)
        data = up.getvalue()

        # 1) Backend se info
        try:
            r = requests.post(
                f"{API_URL}/audio-info",
                files={"file": (up.name, data)},
                timeout=30,
            )
            st.subheader("Backend ka jawab")
            st.json(r.json())
        except Exception as e:
            st.error(f"Backend error: {e}")

        # 2) Plots (frontend par locally)
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            path = tmp.name
        y, sr = load_audio(path)
        os.remove(path)

        st.pyplot(plot_waveform(y, sr))
        st.pyplot(plot_mel(y, sr))
        st.pyplot(plot_mfcc(y, sr))