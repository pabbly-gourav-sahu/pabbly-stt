import os
import tempfile
from contextlib import asynccontextmanager

from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
ALLOWED_EXTENSIONS = {".wav", ".webm", ".mp3", ".ogg", ".m4a"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load faster-whisper model on startup."""
    print(f"Loading faster-whisper '{MODEL_NAME}' model...")
    app.state.model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    print(f"Model '{MODEL_NAME}' loaded successfully (faster-whisper, int8).")
    yield


app = FastAPI(title="Local STT Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query(default=None, description="Language code (en, hi, etc.) or None for auto-detect"),
    task: str = Query(default="transcribe", description="'transcribe' or 'translate'"),
):
    """Transcribe an audio file to text."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    transcribe_opts = {
        "task": task if task in ("transcribe", "translate") else "transcribe",
        "beam_size": 5,
        "best_of": 5,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "initial_prompt": "Hinglish and English conversation.",
        "vad_filter": True,
    }

    if language and language != "auto":
        transcribe_opts["language"] = language

    try:
        segments, _ = app.state.model.transcribe(tmp_path, **transcribe_opts)
        text = " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content={"text": text})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
