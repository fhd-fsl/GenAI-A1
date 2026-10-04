from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time
import numpy as np

from app.schemas import FaceToSketchResponse
from app.services.preprocessing import decode_image, preprocess_image
from app.services.postprocessing import postprocess_image
from app.services.model_manager import model_manager
from app.config import STYLE_LABELS

router = APIRouter()

@router.post("/api/face-to-sketch", response_model=FaceToSketchResponse)
async def face_to_sketch(
    image: UploadFile = File(...),
    style_idx: int = Form(0)
):
    try:
        if style_idx not in [0, 1, 2]:
            raise ValueError("style_idx must be 0, 1, or 2")
            
        img_bytes = await image.read()
        pil_img = decode_image(img_bytes)
        
        # For Generator: Normalize to [-1, 1]
        input_tensor = preprocess_image(pil_img, normalize_to_minus_one_one=True)
        
        # Create style tensor
        style_tensor = np.array([style_idx], dtype=np.int64)
        
        start_time = time.time()
        
        session = model_manager.get_session("generator")
        
        ort_inputs = {
            "photo": input_tensor,
            "style_idx": style_tensor
        }
        
        ort_outputs = session.run(None, ort_inputs)
        output_tensor = ort_outputs[0]
        
        end_time = time.time()
        
        # Postprocess: output is Tanh [-1, 1], so rescale to [0, 1] first
        input_b64 = postprocess_image(input_tensor, is_tanh=True)
        output_b64 = postprocess_image(output_tensor, is_tanh=True)
        
        return FaceToSketchResponse(
            input_image=input_b64,
            output_image=output_b64,
            style=STYLE_LABELS[style_idx],
            inference_time_ms=(end_time - start_time) * 1000.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
