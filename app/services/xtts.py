import tempfile
from pathlib import Path
import logging
import subprocess

import torch
from TTS.api import TTS

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class InvalidSpeakerAudioError(ValueError):
    pass


class XttsService:
    def __init__(self) -> None:
        self._model: TTS | None = None

    def _get_model(self) -> TTS:
        if self._model is None:
            settings = get_settings()
            use_gpu = torch.cuda.is_available() and not settings.force_cpu
            logger.info("Loading XTTS model '%s' on %s", settings.model_name, "cuda" if use_gpu else "cpu")
            self._model = TTS(settings.model_name, gpu=use_gpu)
        return self._model

    def synthesize(self, text: str, language: str, speaker_bytes: bytes, speaker_suffix: str) -> Path:
        suffix = speaker_suffix or ".wav"

        with tempfile.TemporaryDirectory() as tmpdir:
            speaker_path = Path(tmpdir) / f"speaker{suffix}"
            normalized_speaker_path = Path(tmpdir) / "speaker.wav"
            output_path = Path(tmpdir) / "speech.wav"

            speaker_path.write_bytes(speaker_bytes)
            self._normalize_speaker_audio(speaker_path, normalized_speaker_path)
            self._get_model().tts_to_file(
                text=text,
                speaker_wav=[str(normalized_speaker_path)],
                language=language,
                file_path=str(output_path),
                split_sentences=True,
            )

            final_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
            final_path.write_bytes(output_path.read_bytes())
            return final_path

    def _normalize_speaker_audio(self, input_path: Path, output_path: Path) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "24000",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or "ffmpeg could not decode speaker audio"
            raise InvalidSpeakerAudioError(message)


xtts_service = XttsService()
