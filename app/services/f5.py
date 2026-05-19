import logging
import re
import subprocess
import tempfile
import time
from difflib import SequenceMatcher
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

        clean_text = self._prepare_tts_text(text)
        chunks = self._split_tts_text(clean_text, settings.max_chunk_chars)

        # --- Preparar ref_text inicial para logging ---
        clean_reference_text = (reference_text or "").strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            speaker_path = Path(tmpdir) / f"speaker{suffix}"
            normalized_speaker_path = Path(tmpdir) / "speaker.wav"
            output_path = Path(tmpdir) / "speech.wav"

            speaker_path.write_bytes(speaker_bytes)
            logger.info(
                "TTS request: text_chars=%s chunks=%s ref_chars=%s speaker_bytes=%s nfe_step=%s cfg_strength=%s speed=%s remove_silence=%s device=%s",
                len(clean_text),
                len(chunks),
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
                clean_reference_text = self._transcribe_audio(normalized_speaker_path, "es")
                logger.info("Whisper transcription completed in %.2fs: \"%s\"", time.perf_counter() - transcribe_start, clean_reference_text)

            if not clean_reference_text:
                raise MissingReferenceTextError("F5-TTS needs the exact transcript of the reference audio")

            model = self._get_model()
            infer_start = time.perf_counter()

            chunk_paths: list[Path] = []
            for index, chunk in enumerate(chunks):
                chunk_output_path = output_path if len(chunks) == 1 else Path(tmpdir) / f"speech_{index}.wav"
                logger.info("Generating TTS chunk %s/%s: chars=%s", index + 1, len(chunks), len(chunk))
                model.infer(
                    ref_file=str(normalized_speaker_path),
                    ref_text=clean_reference_text,
                    gen_text=chunk,
                    nfe_step=max(32, settings.nfe_step),
                    cfg_strength=settings.cfg_strength,
                    sway_sampling_coef=settings.sway_sampling_coef,
                    speed=settings.speed,
                    cross_fade_duration=settings.cross_fade_duration,
                    remove_silence=settings.remove_silence,
                    file_wave=str(chunk_output_path),
                    show_info=logger.info,
                    progress=None,
                )
                chunk_paths.append(chunk_output_path)

            if len(chunk_paths) > 1:
                self._combine_wavs(chunk_paths, output_path, settings.cross_fade_duration)

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

    def validate_reference(
        self,
        speaker_bytes: bytes,
        speaker_suffix: str,
        reference_text: str,
        language: str = "es",
    ) -> dict[str, object]:
        suffix = speaker_suffix or ".wav"
        settings = get_settings()

        with tempfile.TemporaryDirectory() as tmpdir:
            speaker_path = Path(tmpdir) / f"speaker{suffix}"
            normalized_speaker_path = Path(tmpdir) / "speaker.wav"
            speaker_path.write_bytes(speaker_bytes)
            self._normalize_speaker_audio(speaker_path, normalized_speaker_path)
            transcription = self._transcribe_audio(normalized_speaker_path, language)

        expected = self._normalize_for_match(reference_text)
        actual = self._normalize_for_match(transcription)
        score = SequenceMatcher(None, expected, actual).ratio() if expected and actual else 0.0

        return {
            "accepted": score >= settings.min_reference_match_score,
            "match_score": score,
            "threshold": settings.min_reference_match_score,
            "transcription": transcription,
        }

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
            "afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=start_periods=1:start_silence=0.12:start_threshold=-48dB:stop_periods=1:stop_silence=0.2:stop_threshold=-48dB",
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
            # Fallback without denoise if the FFmpeg build lacks the filter.
            command_alt = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=start_periods=1:start_silence=0.12:start_threshold=-48dB:stop_periods=1:stop_silence=0.2:stop_threshold=-48dB",
                "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", str(output_path)
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

    def _prepare_tts_text(self, text: str) -> str:
        clean = re.sub(r"\[\[(.*?)\]\]", r"\1", text)
        clean = re.sub(r"https?://\S+|www\.\S+", " enlace ", clean)
        replacements = {
            r"\bCra\.?\b": "Carrera",
            r"\bCr\.?\b": "Carrera",
            r"\bCl\.?\b": "Calle",
            r"\bAv\.?\b": "Avenida",
            r"\bNo\.?\b": "numero",
            r"&": " y ",
            r"%": " por ciento ",
        }
        for pattern, value in replacements.items():
            clean = re.sub(pattern, value, clean, flags=re.IGNORECASE)
        clean = clean.replace("°", " grados ")
        clean = re.sub(r"[*_`#>{}\[\]]+", " ", clean)
        clean = re.sub(r"\s+([,.!?;:])", r"\1", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean and clean[-1] not in ".!?":
            clean += "."
        return clean

    def _split_tts_text(self, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long_sentence(sentence, max_chars))
                continue

            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current)

        return chunks or [text]

    def _split_long_sentence(self, sentence: str, max_chars: int) -> list[str]:
        parts = [part.strip() for part in re.split(r"(?<=[,;:])\s+", sentence) if part.strip()]
        chunks: list[str] = []
        current = ""

        for part in parts:
            if len(part) > max_chars:
                if current:
                    chunks.append(self._ensure_sentence_end(current))
                    current = ""
                chunks.extend(self._split_by_words(part, max_chars))
                continue

            candidate = f"{current} {part}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(self._ensure_sentence_end(current))
                current = part
            else:
                current = candidate

        if current:
            chunks.append(self._ensure_sentence_end(current))

        return chunks

    def _split_by_words(self, text: str, max_chars: int) -> list[str]:
        chunks: list[str] = []
        current = ""

        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(self._ensure_sentence_end(current))
                current = word
            else:
                current = candidate

        if current:
            chunks.append(self._ensure_sentence_end(current))

        return chunks

    def _ensure_sentence_end(self, text: str) -> str:
        return text if text[-1] in ".!?" else f"{text}."

    def _combine_wavs(self, paths: list[Path], output_path: Path, cross_fade_duration: float) -> None:
        combined: np.ndarray | None = None
        sample_rate = 24000

        for path in paths:
            audio, sr = sf.read(str(path), always_2d=False)
            sample_rate = sr
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            if combined is None:
                combined = audio
                continue

            fade_samples = min(
                int(cross_fade_duration * sr),
                len(combined) // 2,
                len(audio) // 2,
            )
            if fade_samples > 0:
                fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                overlap = combined[-fade_samples:] * fade_out + audio[:fade_samples] * fade_in
                combined = np.concatenate([combined[:-fade_samples], overlap, audio[fade_samples:]])
            else:
                silence = np.zeros(int(0.06 * sr), dtype=np.float32)
                combined = np.concatenate([combined, silence, audio])

        if combined is None:
            raise InvalidSpeakerAudioError("no generated audio chunks")

        sf.write(str(output_path), combined, sample_rate)

    def _normalize_for_match(self, text: str) -> str:
        clean = text.lower()
        clean = re.sub(r"[^\wáéíóúüñ]+", " ", clean, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", clean).strip()

    def _transcribe_audio(self, audio_path: Path, language: str = "es") -> str:
        model = self._get_whisper_model()
        prompts = {
            "es": "El siguiente audio es en español colombiano, acento caleño.",
            "en": "The following audio is in clear natural English.",
            "pt": "O audio a seguir esta em portugues brasileiro, com fala natural e clara.",
        }
        clean_language = language if language in prompts else "es"
        result = model.transcribe(
            str(audio_path),
            language=clean_language,
            task="transcribe",
            initial_prompt=prompts[clean_language],
        )
        return result["text"].strip()


f5_tts_service = F5TtsService()


