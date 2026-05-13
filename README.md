# CaliGuia F5-TTS Worker

FastAPI service for local/open-source voice cloning with F5-TTS.

## Local

```powershell
cd worker
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

The Next.js app calls this service through:

```env
F5_TTS_API_URL=http://127.0.0.1:8010/tts
```

`POST /tts` expects multipart form data:

- `text`: text to synthesize.
- `speaker_wav`: 5-12 seconds of clean reference voice.
- `reference_text`: exact transcript of `speaker_wav`.

`reference_text` is required on purpose. F5-TTS can transcribe the reference audio, but that loads Whisper and hurts latency.

## Docker

```powershell
docker build -t caliguia-voice-worker ./worker
docker run --rm -p 8010:8080 caliguia-voice-worker
```

## Cloud Run

Deploy this folder as its own Cloud Run service. Then set the Next.js service env var:

```env
F5_TTS_API_URL=https://YOUR-VOICE-WORKER.run.app/tts
```

Recommended starting resources:

```txt
CPU: 2
Memory: 4Gi
Concurrency: 1-4
Timeout: 300s
```

F5-TTS downloads and loads model weights on first startup. For Spanish production voices, prefer a Spanish F5 checkpoint by setting `F5_TTS_CKPT_FILE` and `F5_TTS_VOCAB_FILE`.
