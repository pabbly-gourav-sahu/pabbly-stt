import os
import tempfile
from contextlib import asynccontextmanager

import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ALLOWED_EXTENSIONS = {".wav", ".webm", ".mp3"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Whisper model on startup."""
    print("Loading Whisper 'base' model...")
    app.state.model = whisper.load_model("base")
    print("Model loaded successfully.")
    yield


app = FastAPI(title="Local STT Service", lifespan=lifespan)

# Enable CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe an audio file to text."""
    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save uploaded file to a temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Transcribe and translate to English
    try:
        result = app.state.model.transcribe(tmp_path, task="translate")
        text = result.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content={"text": text})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
