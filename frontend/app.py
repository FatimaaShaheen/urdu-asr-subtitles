import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.title("Urdu ASR aur Subtitle Generator")
st.write("Step 1: Frontend aur Backend ka connection test")

if st.button("Backend check karo"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.success(r.json())
    except Exception as e:
        st.error(f"Backend se connect nahi hua: {e}")