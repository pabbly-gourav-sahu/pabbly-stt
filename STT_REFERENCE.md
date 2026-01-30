# STT Service Reference

## Run
```bash
cd stt-service
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API
```
GET  /health      → {"status":"ok"}
POST /transcribe  → {"text":"..."}  (form: file=audio.webm)
```

## Test
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/transcribe -F "file=@sample.wav"
```
