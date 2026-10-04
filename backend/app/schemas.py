from pydantic import BaseModel
from typing import Dict, Any

class RestorationResponse(BaseModel):
    input_image: str          # base64 encoded png
    output_image: str         # base64 encoded png
    corruption_type: str
    corruption_params: Dict[str, Any]
    inference_time_ms: float

class HardRouteResponse(BaseModel):
    input_image: str          
    output_image: str         
    classifier_probs: Dict[str, float]  
    predicted_corruption: str
    selected_expert: str
    inference_time_ms: float
    classifier_time_ms: float
    specialist_time_ms: float

class SoftMixtureResponse(BaseModel):
    input_image: str          
    output_image: str         
    routing_weights: Dict[str, float]  
    dominant_expert: str
    inference_time_ms: float

class FaceToSketchResponse(BaseModel):
    input_image: str          
    output_image: str         
    style: str
    inference_time_ms: float
