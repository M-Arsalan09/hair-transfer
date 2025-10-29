from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
import uuid, os, shutil
from PIL import Image
from app.core.config import OUTPUT_DIR, CONFIG_PATH, MODEL_DIR, RUNPOD_PROXY, DEVICE
from app.services.stable_hair_service import StableHair
from app.core.utils import detect_and_align_face
from omegaconf import OmegaConf
import torch
import cv2
import mediapipe as mp
import numpy as np
from app.services.model_loader import model

router = APIRouter()



@router.post("/generate")
async def generate(request: Request, source_image: UploadFile = File(...)):
    """
    Perform hair transfer with 3 reference styles:
    1. Face Match
    2. Trendy Haircut
    3. AI Suggested
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet")

    # Create temp directory for user session
    uid = str(uuid.uuid4())[:8]
    temp_dir = os.path.join("tmp", uid)
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded source image
    src_path = os.path.join(temp_dir, "source.jpg")
    with open(src_path, "wb") as f:
        shutil.copyfileobj(source_image.file, f)

    # Reference images (hardcoded)
    reference_images = {
        "Face Match": "./images/facematch.jpg",
        "Trendy Haircut": "./images/celeb1.jpg",
        "AI Suggested": "./images/suggested.jpg"
    }

    # Base output structure
    results = {}
    
    try:
        print("Step 1: Detecting and aligning face")
        detected_path = os.path.join(temp_dir, "detected_face.jpg")
        aligned_path = os.path.join(temp_dir, "aligned_face.jpg")
        detected_face, aligned_face, meta = detect_and_align_face(src_path, detected_path, aligned_path)
        
    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    

    try:
        kwargs = OmegaConf.to_container(model.config.inference_kwargs)
        kwargs["source_image"] = aligned_path

        RUNPOD_PROXY = os.getenv("RUNPOD_PROXY_URL", "https://wuxg73nt9byvrv-8000.proxy.runpod.net")

        # Loop through all 3 styles
        for style_name, ref_path in reference_images.items():
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Missing reference image: {ref_path}")

            kwargs["reference_image"] = ref_path
            print(f"🧠 Running inference for {style_name}...")

            # Run model inference
            _, gen_img, _, _ = model.Hair_Transfer(**kwargs)

            # Convert tensor/image format properly
            if isinstance(gen_img, torch.Tensor):
                gen_img = gen_img.detach().cpu().numpy()
            if gen_img.dtype != np.uint8:
                gen_img = (np.clip(gen_img, 0, 1) * 255).astype(np.uint8)
            if gen_img.ndim == 4:
                gen_img = gen_img[0]
            if gen_img.shape[0] == 3:
                gen_img = np.transpose(gen_img, (1, 2, 0))
                
            # print("realining face back to original orientation")
            # gen_img = realign_face(gen_img, meta)
            # Save result
            out_filename = f"{uid}_{style_name.replace(' ', '_').lower()}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            Image.fromarray(gen_img).save(out_path)

            # Add to results
            results[style_name] = f"{RUNPOD_PROXY}/outputs/{out_filename}"

        # Cleanup
        shutil.rmtree(temp_dir)

        # Return structured JSON
        return JSONResponse({
            "details": {
                "status": "success",
                "Face Match": results.get("Face Match"),
                "Trendy Haircut": results.get("Trendy Haircut"),
                "AI Suggested": results.get("AI Suggested")
            }
        })

    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
