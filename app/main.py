# Apply compatibility patches before importing other modules
import app.core.compat_patch

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import generate
from app.core.config import OUTPUT_DIR
from app.services.model_loader import load_model

app = FastAPI(title="StableHair API")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.include_router(generate.router, prefix="/api", tags=["Hair Transfer"])

@app.on_event("startup")
def startup_event():
    load_model()
