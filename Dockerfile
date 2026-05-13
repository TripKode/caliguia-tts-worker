FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV COQUI_TOS_AGREED=1
ENV XTTS_FORCE_CPU=1
ENV XTTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2
ENV XTTS_PRELOAD_ON_STARTUP=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
  && pip install -r requirements.txt

RUN python -c "import os; from TTS.api import TTS; TTS(os.environ['XTTS_MODEL_NAME'], gpu=False)"

COPY app ./app
COPY main.py ./main.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
