from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.f5 import f5_tts_service


app = FastAPI(title="CaliGuia F5-TTS Worker")
app.include_router(router)


@app.on_event("startup")
def preload_f5_tts_model() -> None:
    if get_settings().preload_on_startup:
        f5_tts_service.preload()
