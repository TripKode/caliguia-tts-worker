FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV F5_TTS_FORCE_CPU=1
ENV F5_TTS_MODEL_NAME=F5TTS_v1_Base
ENV F5_TTS_NFE_STEP=16
ENV F5_TTS_PRELOAD_ON_STARTUP=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
  && pip install -r requirements.txt

RUN python -c "import os; from f5_tts.api import F5TTS; F5TTS(model=os.environ['F5_TTS_MODEL_NAME'], device='cpu')"

COPY app ./app
COPY main.py ./main.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
