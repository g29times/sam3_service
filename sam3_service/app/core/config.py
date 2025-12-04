"""
SAM3 服务配置
"""
import os
from pathlib import Path

# === 运行模式 ===
# 通过环境变量控制：SAM3_MODE=mock（默认）或 SAM3_MODE=real
SAM3_MODE = os.getenv("SAM3_MODE", "mock").lower()  # "mock" or "real"

# === 路径配置 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # sam3_service 的上级
MODEL_DIR = PROJECT_ROOT / "models" / "sam3"
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

# === 模型配置 ===
# SAM3 模型通过 HuggingFace 下载，需先申请访问权限：
# https://huggingface.co/facebook/sam3
# 然后运行 `huggingface-cli login` 进行认证
SAM3_HF_REPO = "facebook/sam3"  # HuggingFace 仓库名

# === 设备配置 ===
def get_device() -> str:
    """延迟检测设备，避免 Mock 模式必须安装 torch"""
    if SAM3_MODE == "mock":
        return "cpu"
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()

# === 图像处理配置 ===
MAX_IMAGE_SIZE = 2048  # 长边最大尺寸，超过会缩放
DEFAULT_BLUR_STRENGTH = 21  # 默认模糊强度

# === 分割配置 ===
AUTO_MASK_MIN_AREA_RATIO = 0.01  # 自动分割时，mask 最小面积占比（过滤噪点）
AUTO_MASK_MAX_COUNT = 50  # 自动分割最多返回的 mask 数量
