from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time
import json
import numpy as np

from app.schemas import SoftMixtureResponse
from app.services.preprocessing import decode_image, preprocess_image
from app.services.postprocessing import postprocess_image
from app.services.corruption import apply_deterministic_corruption
from app.services.model_manager import model_manager
from app.config import CORRUPTION_LABELS

router = APIRouter()

@router.post("/api/soft-mixture", response_model=SoftMixtureResponse)
async def soft_mixture(
    image: UploadFile = File(...),
    corruption_type: str = Form("clean"),
    corruption_params: str = Form("{}")
):
    try:
        params = json.loads(corruption_params)
        img_bytes = await image.read()
        pil_img = decode_image(img_bytes)
        
        # Preprocess to NCHW [0,1]
        input_tensor = preprocess_image(pil_img, normalize_to_minus_one_one=False)
        corrupted_tensor = apply_deterministic_corruption(input_tensor, corruption_type, params)
        
        start_time = time.time()
        
        session = model_manager.get_session("task3_soft_moe")
        input_name = session.get_inputs()[0].name
        
        ort_outputs = session.run(None, {input_name: corrupted_tensor})
        # Model returns: ['reconstructed', 'weights', 'logits']
        output_tensor = ort_outputs[0]
        weights = ort_outputs[1][0] # (4,)
        
        end_time = time.time()
        
        # Map weights
        routing_weights = {label: float(w) for label, w in zip(CORRUPTION_LABELS, weights)}
        
        # Find dominant expert
        dominant_idx = int(np.argmax(weights))
        dominant_expert = CORRUPTION_LABELS[dominant_idx]
        if dominant_expert == "clean":
            dominant_expert = "identity"
            
        input_b64 = postprocess_image(corrupted_tensor, is_tanh=False)
        output_b64 = postprocess_image(output_tensor, is_tanh=False)
        
        return SoftMixtureResponse(
            input_image=input_b64,
            output_image=output_b64,
            routing_weights=routing_weights,
            dominant_expert=dominant_expert,
            inference_time_ms=(end_time - start_time) * 1000.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
