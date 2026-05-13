from functools import lru_cache
from pydantic import BaseModel
import os


class Settings(BaseModel):
    model_name: str = "F5TTS_v1_Base"
    ckpt_file: str = ""
    vocab_file: str = ""
    hf_cache_dir: str | None = None
    max_text_chars: int = 900
    force_cpu: bool = False
    preload_on_startup: bool = False
    supported_languages: set[str] = {"es", "en", "pt"}
    nfe_step: int = 16
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    speed: float = 1.0
    cross_fade_duration: float = 0.15
    remove_silence: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings(
        model_name=os.getenv("F5_TTS_MODEL_NAME", Settings().model_name),
        ckpt_file=os.getenv("F5_TTS_CKPT_FILE", Settings().ckpt_file),
        vocab_file=os.getenv("F5_TTS_VOCAB_FILE", Settings().vocab_file),
        hf_cache_dir=os.getenv("F5_TTS_HF_CACHE_DIR") or Settings().hf_cache_dir,
        max_text_chars=int(os.getenv("F5_TTS_MAX_TEXT_CHARS", str(Settings().max_text_chars))),
        force_cpu=os.getenv("F5_TTS_FORCE_CPU", "0") == "1",
        preload_on_startup=os.getenv("F5_TTS_PRELOAD_ON_STARTUP", "0") == "1",
        nfe_step=int(os.getenv("F5_TTS_NFE_STEP", str(Settings().nfe_step))),
        cfg_strength=float(os.getenv("F5_TTS_CFG_STRENGTH", str(Settings().cfg_strength))),
        sway_sampling_coef=float(os.getenv("F5_TTS_SWAY_SAMPLING_COEF", str(Settings().sway_sampling_coef))),
        speed=float(os.getenv("F5_TTS_SPEED", str(Settings().speed))),
        cross_fade_duration=float(os.getenv("F5_TTS_CROSS_FADE_DURATION", str(Settings().cross_fade_duration))),
        remove_silence=os.getenv("F5_TTS_REMOVE_SILENCE", "0") == "1",
    )
