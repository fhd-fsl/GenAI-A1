from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time
import json
from typing import Optional

from app.schemas import RestorationResponse
from app.services.preprocessing import decode_image, preprocess_image
from app.services.postprocessing import postprocess_image
from app.services.corruption import apply_deterministic_corruption
from app.services.model_manager import model_manager

router = APIRouter()

@router.post("/api/universal-restore", response_model=RestorationResponse)
async def universal_restore(
    image: UploadFile = File(...),
    corruption_type: str = Form("clean"),
    corruption_params: str = Form("{}")
):
    try:
        # Parse params
        params = json.loads(corruption_params)
        
        # Read and preprocess
        img_bytes = await image.read()
        pil_img = decode_image(img_bytes)
        
        # Preprocess to NCHW [0,1]
        input_tensor = preprocess_image(pil_img, normalize_to_minus_one_one=False)
        
        # Apply corruption
        corrupted_tensor = apply_deterministic_corruption(input_tensor, corruption_type, params)
        
        # Run inference
        start_time = time.time()
        
        session = model_manager.get_session("universal_ae")
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: corrupted_tensor}
        
        ort_outputs = session.run(None, ort_inputs)
        output_tensor = ort_outputs[0]
        
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000.0
        
        # Postprocess (AE uses Sigmoid -> [0,1], no rescaling needed)
        input_b64 = postprocess_image(corrupted_tensor, is_tanh=False)
        output_b64 = postprocess_image(output_tensor, is_tanh=False)
        
        return RestorationResponse(
            input_image=input_b64,
            output_image=output_b64,
            corruption_type=corruption_type,
            corruption_params=params,
            inference_time_ms=inference_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
