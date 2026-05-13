import logging
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from f5_tts.api import F5TTS

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class InvalidSpeakerAudioError(ValueError):
    pass


class MissingReferenceTextError(ValueError):
    pass


class F5TtsService:
    def __init__(self) -> None:
        self._model: F5TTS | None = None

    def _get_model(self) -> F5TTS:
        if self._model is None:
            settings = get_settings()
            use_gpu = torch.cuda.is_available() and not settings.force_cpu
            device = "cuda" if use_gpu else "cpu"
            logger.info("Loading F5-TTS model '%s' on %s", settings.model_name, device)
            self._model = F5TTS(
                model=settings.model_name,
                ckpt_file=settings.ckpt_file,
                vocab_file=settings.vocab_file,
                device=device,
                hf_cache_dir=settings.hf_cache_dir,
            )
        return self._model

    def preload(self) -> None:
        self._get_model()

    def synthesize(
        self,
        text: str,
        speaker_bytes: bytes,
        speaker_suffix: str,
        reference_text: str | None,
    ) -> Path:
        clean_reference_text = (reference_text or "").strip()
        if not clean_reference_text:
            raise MissingReferenceTextError("F5-TTS needs the exact transcript of the reference audio")

        suffix = speaker_suffix or ".wav"
        settings = get_settings()

        with tempfile.TemporaryDirectory() as tmpdir:
            speaker_path = Path(tmpdir) / f"speaker{suffix}"
            normalized_speaker_path = Path(tmpdir) / "speaker.wav"
            output_path = Path(tmpdir) / "speech.wav"

            speaker_path.write_bytes(speaker_bytes)
            self._normalize_speaker_audio(speaker_path, normalized_speaker_path)

            self._get_model().infer(
                ref_file=str(normalized_speaker_path),
                ref_text=clean_reference_text,
                gen_text=text,
                nfe_step=settings.nfe_step,
                cfg_strength=settings.cfg_strength,
                sway_sampling_coef=settings.sway_sampling_coef,
                speed=settings.speed,
                cross_fade_duration=settings.cross_fade_duration,
                remove_silence=settings.remove_silence,
                file_wave=str(output_path),
                show_info=logger.info,
                progress=None,
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

        try:
            data, sample_rate = sf.read(str(output_path))
        except Exception as exc:
            raise InvalidSpeakerAudioError("speaker audio could not be read after normalization") from exc

        if sample_rate != 24000 or len(data) == 0:
            raise InvalidSpeakerAudioError("speaker audio normalization produced an invalid wav")


f5_tts_service = F5TtsService()
