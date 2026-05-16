FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV F5_TTS_FORCE_CPU=1
ENV F5_TTS_MODEL_NAME=F5TTS_Base
ENV F5_TTS_SPANISH_REPO=jpgallegoar/F5-Spanish
ENV F5_TTS_CKPT_FILE=/app/models/F5-Spanish/model_1200000.safetensors
ENV F5_TTS_VOCAB_FILE=/app/models/F5-Spanish/vocab.txt
ENV F5_TTS_NFE_STEP=64
ENV F5_TTS_CFG_STRENGTH=1.5
ENV F5_TTS_SPEED=0.92
ENV F5_TTS_SWAY_SAMPLING_COEF=-1.0
ENV F5_TTS_REMOVE_SILENCE=1
ENV F5_TTS_CROSS_FADE_DURATION=0.15
ENV F5_TTS_PRELOAD_ON_STARTUP=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir torch==2.8.0+cpu torchaudio==2.8.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python -c "from huggingface_hub import hf_hub_download; import os; repo=os.environ['F5_TTS_SPANISH_REPO']; target='/app/models/F5-Spanish'; os.makedirs(target, exist_ok=True); hf_hub_download(repo_id=repo, filename='model_1200000.safetensors', local_dir=target); hf_hub_download(repo_id=repo, filename='vocab.txt', local_dir=target)"

# Validate installation and preload model
RUN python -c "import os; from f5_tts.api import F5TTS; F5TTS(model=os.environ['F5_TTS_MODEL_NAME'], ckpt_file=os.environ['F5_TTS_CKPT_FILE'], vocab_file=os.environ['F5_TTS_VOCAB_FILE'], device='cpu')"

COPY app ./app
COPY main.py ./main.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
