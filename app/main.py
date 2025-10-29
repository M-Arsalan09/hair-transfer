from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import generate
from app.core.config import OUTPUT_DIR

app = FastAPI(title="StableHair API")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.include_router(generate.router, prefix="/api", tags=["Hair Transfer"])
