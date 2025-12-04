"""
健康检查接口
"""
from fastapi import APIRouter

from ..core.config import DEVICE, SAM3_HF_REPO, SAM3_MODE
from ..core.sam3_model import sam3_model

router = APIRouter()


@router.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "ok" if sam3_model.is_loaded else "model_not_loaded",
        "mode": SAM3_MODE,  # "mock" or "real"
        "device": DEVICE,
        "hf_repo": SAM3_HF_REPO,
        "model_loaded": sam3_model.is_loaded,
        "backend": "fastapi",
    }
