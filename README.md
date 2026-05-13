# CaliGuia XTTS Worker

FastAPI service for local/open-source voice cloning with Coqui XTTS-v2.

## Local

```powershell
cd worker
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

The Next.js app calls this service through:

```env
XTTS_API_URL=http://127.0.0.1:8010/tts
```

## Docker

```powershell
docker build -t caliguia-voice-worker ./worker
docker run --rm -p 8010:8080 caliguia-voice-worker
```

## Cloud Run

Deploy this folder as its own Cloud Run service. Then set the Next.js service env var:

```env
XTTS_API_URL=https://YOUR-VOICE-WORKER.run.app/tts
```

Recommended starting resources:

```txt
CPU: 2
Memory: 4Gi
Concurrency: 1-4
Timeout: 300s
```

XTTS is model-heavy, so the first request after a cold start can be slow while the model loads.
