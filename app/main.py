from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.xtts import xtts_service


app = FastAPI(title="CaliGuia XTTS Worker")
app.include_router(router)


@app.on_event("startup")
def preload_xtts_model() -> None:
    if get_settings().preload_on_startup:
        xtts_service.preload()
