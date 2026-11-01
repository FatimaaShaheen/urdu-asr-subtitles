import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data.audio_utils import audio_info, load_audio

app = FastAPI(title="Urdu ASR Subtitle API")


@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend chal raha hai"}


@app.post("/audio-info")
async def get_audio_info(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        y, sr = load_audio(path)
    except Exception:
        raise HTTPException(status_code=400, detail="Audio file parh nahi saka")
    finally:
        os.remove(path)
    return {"filename": file.filename, **audio_info(y, sr)}