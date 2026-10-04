from fastapi import APIRouter
from app.services.model_manager import model_manager
from typing import Dict, Any

router = APIRouter()

@router.get("/health")
def health_check() -> Dict[str, Any]:
    loaded_models = list(model_manager.sessions.keys())
    return {
        "status": "healthy",
        "loaded_models": loaded_models,
        "count": len(loaded_models)
    }
