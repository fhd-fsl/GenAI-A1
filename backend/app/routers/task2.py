from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time
import json
import numpy as np
from typing import Optional

from app.schemas import HardRouteResponse
from app.services.preprocessing import decode_image, preprocess_image
from app.services.postprocessing import postprocess_image
from app.services.corruption import apply_deterministic_corruption
from app.services.model_manager import model_manager
from app.config import CORRUPTION_LABELS

router = APIRouter()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@router.post("/api/hard-route", response_model=HardRouteResponse)
async def hard_route(
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
        
        total_start_time = time.time()
        
        # 1. Classification
        cls_start_time = time.time()
        cls_session = model_manager.get_session("task2_classifier")
        cls_input_name = cls_session.get_inputs()[0].name
        cls_outputs = cls_session.run(None, {cls_input_name: corrupted_tensor})
        logits = cls_outputs[0] # (1, 4)
        
        probs = softmax(logits)[0]
        predicted_idx = int(np.argmax(probs))
        predicted_class = CORRUPTION_LABELS[predicted_idx]
        
        cls_end_time = time.time()
        
        # 2. Routing to Specialist
        spec_start_time = time.time()
        
        expert_map = {
            1: "task2_specialist_sp",
            2: "task2_specialist_blur",
            3: "task2_specialist_occ"
        }
        
        if predicted_idx == 0:
            # Clean -> Identity bypass
            selected_expert = "identity"
            output_tensor = corrupted_tensor
        else:
            selected_expert = expert_map[predicted_idx]
            spec_session = model_manager.get_session(selected_expert)
            spec_input_name = spec_session.get_inputs()[0].name
            spec_outputs = spec_session.run(None, {spec_input_name: corrupted_tensor})
            output_tensor = spec_outputs[0]
            
        spec_end_time = time.time()
        total_end_time = time.time()
        
        classifier_probs = {label: float(prob) for label, prob in zip(CORRUPTION_LABELS, probs)}
        
        input_b64 = postprocess_image(corrupted_tensor, is_tanh=False)
        output_b64 = postprocess_image(output_tensor, is_tanh=False)
        
        return HardRouteResponse(
            input_image=input_b64,
            output_image=output_b64,
            classifier_probs=classifier_probs,
            predicted_corruption=predicted_class,
            selected_expert=selected_expert,
            inference_time_ms=(total_end_time - total_start_time) * 1000.0,
            classifier_time_ms=(cls_end_time - cls_start_time) * 1000.0,
            specialist_time_ms=(spec_end_time - spec_start_time) * 1000.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
