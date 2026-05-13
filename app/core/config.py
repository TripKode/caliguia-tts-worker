from functools import lru_cache
from pydantic import BaseModel
import os


class Settings(BaseModel):
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    max_text_chars: int = 900
    force_cpu: bool = False
    preload_on_startup: bool = False
    supported_languages: set[str] = {"es", "en", "pt"}


@lru_cache
def get_settings() -> Settings:
    return Settings(
        model_name=os.getenv("XTTS_MODEL_NAME", Settings().model_name),
        max_text_chars=int(os.getenv("XTTS_MAX_TEXT_CHARS", str(Settings().max_text_chars))),
        force_cpu=os.getenv("XTTS_FORCE_CPU", "0") == "1",
        preload_on_startup=os.getenv("XTTS_PRELOAD_ON_STARTUP", "0") == "1",
    )
