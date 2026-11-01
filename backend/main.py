from fastapi import FastAPI

app = FastAPI(title="Urdu ASR Subtitle API")

@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend chal raha hai"}
