import os
from dotenv import load_dotenv
import torch

load_dotenv()

# BASE_DIR points to the freshpass-ai root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "app/services/configs/hair_transfer.yaml"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUNPOD_PROXY = os.getenv("RUNPOD_PROXY_URL", "")
