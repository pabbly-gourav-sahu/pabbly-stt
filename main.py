import os
import tempfile
from contextlib import asynccontextmanager

from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Multilingual model for English + Hindi + Hinglish support
# Use "small" for multilingual, NOT "small.en" (English-only)
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
ALLOWED_EXTENSIONS = {".wav", ".webm", ".mp3", ".ogg", ".m4a"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load faster-whisper model on startup."""
    print(f"Loading faster-whisper '{MODEL_NAME}' model...")
    # Use int8 for CPU, float16 for GPU
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    app.state.model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=compute_type)
    print(f"Model '{MODEL_NAME}' loaded successfully ({DEVICE}, {compute_type}).")
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
    language: str = Query(default=None, description="Language code (en, hi) or None for auto-detect (best for Hinglish)"),
    task: str = Query(default="transcribe", description="'transcribe' or 'translate'"),
):
    """
    Transcribe audio supporting English, Hindi, and Hinglish.

    For Hinglish (mixed Hindi + English): Don't set language param - auto-detect works best.
    For pure Hindi: Set language='hi'
    For pure English: Set language='en'
    """
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

    # Speed-optimized settings for near real-time transcription
    transcribe_opts = {
        "task": task if task in ("transcribe", "translate") else "transcribe",
        "beam_size": 1,           # Greedy decoding: 2-3x faster
        "best_of": 1,             # No resampling: faster
        "temperature": 0.0,       # Single temperature: no fallback overhead
        "vad_filter": True,       # Skip silence: 20-40% faster
        "vad_parameters": {
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 200,
        },
    }

    # Language handling:
    # - For Hinglish: Don't set language (auto-detect handles code-switching)
    # - For pure Hindi/English: Set explicitly for better accuracy
    if language and language != "auto":
        transcribe_opts["language"] = language

    try:
        segments, info = app.state.model.transcribe(tmp_path, **transcribe_opts)
        text = " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content={
        "text": text,
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 2)
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
