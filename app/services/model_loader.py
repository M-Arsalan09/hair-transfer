from app.core.config import OUTPUT_DIR, CONFIG_PATH, MODEL_DIR, RUNPOD_PROXY, DEVICE
from app.services.stable_hair_service import StableHair
import torch

model = None

def load_model():
    """Load StableHair model at startup"""
    global model
    print("🚀 Loading StableHair model...")
    model = StableHair(config=CONFIG_PATH, device=DEVICE, weight_dtype=torch.float32)
    print("✅ Model loaded successfully!")