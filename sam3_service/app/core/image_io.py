"""
图像编解码工具
"""
import base64
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

from .config import MAX_IMAGE_SIZE


def decode_image_from_bytes(data: bytes) -> np.ndarray:
    """从字节流解码为 numpy 数组 (RGB)"""
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)


def decode_image_from_base64(b64_str: str) -> np.ndarray:
    """从 base64 字符串解码为 numpy 数组 (RGB)"""
    # 去掉可能的 data:image/xxx;base64, 前缀
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return decode_image_from_bytes(data)


def encode_image_to_base64(img: np.ndarray, format: str = "PNG") -> str:
    """将 numpy 数组编码为 base64 字符串（带 data URI 前缀）"""
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


def resize_if_needed(img: np.ndarray, max_size: int = MAX_IMAGE_SIZE) -> Tuple[np.ndarray, float]:
    """
    如果图像长边超过 max_size，则等比缩放。
    返回：(缩放后图像, 缩放比例)
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_size:
        return img, 1.0
    scale = max_size / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = Image.fromarray(img)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(resized), scale
