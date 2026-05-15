import logging
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
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
        self._whisper_model: whisper.Whisper | None = None
        self._device: str | None = None

    def _get_model(self) -> F5TTS:
        if self._model is None:
            settings = get_settings()
            use_gpu = torch.cuda.is_available() and not settings.force_cpu
            device = "cuda" if use_gpu else "cpu"
            self._device = device
            logger.info(
                "Loading F5-TTS model '%s' on %s (cuda_available=%s, force_cpu=%s)",
                settings.model_name,
                device,
                torch.cuda.is_available(),
                settings.force_cpu,
            )
            start = time.perf_counter()
            self._model = F5TTS(
                model=settings.model_name,
                ckpt_file=settings.ckpt_file,
                vocab_file=settings.vocab_file,
                device=device,
                hf_cache_dir=settings.hf_cache_dir,
            )
            logger.info("F5-TTS model loaded in %.2fs on %s", time.perf_counter() - start, device)
        return self._model

    def _get_whisper_model(self) -> whisper.Whisper:
        if self._whisper_model is None:
            settings = get_settings()
            use_gpu = torch.cuda.is_available() and not settings.force_cpu
            device = "cuda" if use_gpu else "cpu"
            # Usamos 'turbo' o 'large-v3' según disponibilidad. 
            # El usuario pidió large-v3, pero 'turbo' es más rápido y muy preciso.
            # Implementaremos una opción para configurar esto luego, por ahora usamos el pedido.
            model_size = "large-v3" 
            logger.info("Loading Whisper model '%s' on %s", model_size, device)
            start = time.perf_counter()
            self._whisper_model = whisper.load_model(model_size, device=device)
            logger.info("Whisper model loaded in %.2fs", time.perf_counter() - start)
        return self._whisper_model

    def preload(self) -> None:
        self._get_model()

    def synthesize(
        self,
        text: str,
        speaker_bytes: bytes,
        speaker_suffix: str,
        reference_text: str | None,
    ) -> Path:
        suffix = speaker_suffix or ".wav"
        settings = get_settings()
        total_start = time.perf_counter()

        # --- Preparar ref_text inicial para logging ---
        clean_reference_text = (reference_text or "").strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            speaker_path = Path(tmpdir) / f"speaker{suffix}"
            normalized_speaker_path = Path(tmpdir) / "speaker.wav"
            output_path = Path(tmpdir) / "speech.wav"

            speaker_path.write_bytes(speaker_bytes)
            logger.info(
                "TTS request: text_chars=%s ref_chars=%s speaker_bytes=%s nfe_step=%s cfg_strength=%s speed=%s remove_silence=%s device=%s",
                len(text),
                len(clean_reference_text),
                len(speaker_bytes),
                settings.nfe_step,
                settings.cfg_strength,
                settings.speed,
                settings.remove_silence,
                self._device or "not_loaded",
            )

            normalize_start = time.perf_counter()
            self._normalize_speaker_audio(speaker_path, normalized_speaker_path)
            logger.info("Speaker normalization/denoising completed in %.2fs", time.perf_counter() - normalize_start)

            # --- Transcripción automática si no hay ref_text o es AUTO ---
            if not clean_reference_text or clean_reference_text == "AUTO_TRANSCRIBE":
                logger.info("Reference text missing or AUTO requested. Transcribing with Whisper...")
                transcribe_start = time.perf_counter()
                clean_reference_text = self._transcribe_audio(normalized_speaker_path)
                logger.info("Whisper transcription completed in %.2fs: \"%s\"", time.perf_counter() - transcribe_start, clean_reference_text)

            if not clean_reference_text:
                raise MissingReferenceTextError("F5-TTS needs the exact transcript of the reference audio")

            model = self._get_model()
            infer_start = time.perf_counter()
            
            # Usamos parámetros optimizados para mayor fidelidad
            # nfe_step=32 da un detalle superior a 16
            # cfg_strength=1.5 es el punto dulce para naturalidad
            model.infer(
                ref_file=str(normalized_speaker_path),
                ref_text=clean_reference_text,
                gen_text=text,
                nfe_step=max(32, settings.nfe_step),
                cfg_strength=1.5,
                sway_sampling_coef=settings.sway_sampling_coef,
                speed=settings.speed,
                cross_fade_duration=settings.cross_fade_duration,
                remove_silence=True, # Forzamos para mayor fluidez
                file_wave=str(output_path),
                show_info=logger.info,
                progress=None,
            )
            infer_seconds = time.perf_counter() - infer_start

            final_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
            final_path.write_bytes(output_path.read_bytes())
            output_bytes = final_path.stat().st_size
            logger.info(
                "TTS infer completed in %.2fs, total %.2fs, output_bytes=%s, device=%s",
                infer_seconds,
                time.perf_counter() - total_start,
                output_bytes,
                self._device or "unknown",
            )
            return final_path

    def _normalize_speaker_audio(self, input_path: Path, output_path: Path) -> None:
        # Paso 1: denoising + loudness + resampling + TRIM DE SILENCIOS (inicio/fin)
        # Esto es vital para que F5-TTS no alucine al inicio de la frase
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-af",
            "arnndn=m=mp.rnnn,loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB:stop_periods=1:stop_silence=0.1:stop_threshold=-50dB",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or "ffmpeg could not process speaker audio"
            logger.error("FFmpeg error: %s", message)
            # Si falla arnndn (puede pasar en algunas versiones de ffmpeg), intentamos sin él
            command_alt = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1", str(output_path)
            ]
            subprocess.run(command_alt, check=True)

        # Paso 2: Normalización RMS manual a 0.1 (target interno de F5-TTS)
        try:
            audio, sr = sf.read(str(output_path))
            rms = np.sqrt(np.mean(audio**2))
            target_rms = 0.1
            if rms > 0:
                audio = audio * (target_rms / rms)
            sf.write(str(output_path), audio, sr)
        except Exception as exc:
            logger.error("RMS normalization failed: %s", exc)
            raise InvalidSpeakerAudioError("speaker audio could not be read or normalized") from exc

    def _transcribe_audio(self, audio_path: Path) -> str:
        model = self._get_whisper_model()
        # El initial_prompt guía a Whisper hacia el dialecto correcto
        result = model.transcribe(
            str(audio_path),
            language="es",
            task="transcribe",
            initial_prompt="El siguiente audio es en español colombiano, acento caleño."
        )
        return result["text"].strip()


f5_tts_service = F5TtsService()


