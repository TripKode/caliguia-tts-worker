from pathlib import Path
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.core.config import get_settings
from app.services.f5 import InvalidSpeakerAudioError, MissingReferenceTextError, f5_tts_service


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    settings = get_settings()
    return {"ok": True, "engine": "f5-tts", "model": settings.model_name}


@router.post("/tts")
async def tts(
    text: str = Form(...),
    language: str = Form("es"),
    voice_id: str | None = Form(None),
    reference_text: str | None = Form(None),
    ref_text: str | None = Form(None),
    speaker_text: str | None = Form(None),
    speaker_wav: UploadFile = File(...),
):
    settings = get_settings()
    clean_text = text.strip()
    clean_language = language.strip().lower()

    if not clean_text:
        raise HTTPException(status_code=400, detail="Missing text")

    if len(clean_text) > settings.max_text_chars:
        raise HTTPException(status_code=413, detail="Text is too long")

    if clean_language not in settings.supported_languages:
        clean_language = "es"

    speaker_suffix = Path(speaker_wav.filename or "speaker.wav").suffix or ".wav"
    clean_reference_text = reference_text or ref_text or speaker_text
    try:
        output_path = f5_tts_service.synthesize(
            text=clean_text,
            speaker_bytes=await speaker_wav.read(),
            speaker_suffix=speaker_suffix,
            reference_text=clean_reference_text,
        )
    except HTTPException:
        raise
    except MissingReferenceTextError as exc:
        raise HTTPException(
            status_code=400,
            detail="Missing reference_text: send the exact transcript of speaker_wav for fast F5-TTS cloning",
        ) from exc
    except InvalidSpeakerAudioError as exc:
        logger.warning("Invalid speaker audio: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid speaker audio") from exc
    except Exception as exc:
        logger.exception("F5-TTS synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail="F5-TTS synthesis failed") from exc

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="caliguia-voice.wav",
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )
