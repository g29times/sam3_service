"""
隐私过滤流水线
"""
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from PIL import Image, ImageFilter

from .config import AUTO_MASK_MIN_AREA_RATIO, DEFAULT_BLUR_STRENGTH
from .sam3_model import sam3_model, MaskResult


BlurType = Literal["gaussian", "pixelate", "solid"]


@dataclass
class AppliedRegion:
    """记录被处理的区域"""
    mask_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    area: int


@dataclass
class PrivacyFilterResult:
    """隐私过滤结果"""
    filtered_image: np.ndarray
    applied_regions: List[AppliedRegion]


def apply_gaussian_blur(
    image: np.ndarray,
    mask: np.ndarray,
    strength: int = DEFAULT_BLUR_STRENGTH,
) -> np.ndarray:
    """对 mask 区域应用高斯模糊"""
    pil_img = Image.fromarray(image)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=strength))
    blurred_arr = np.array(blurred)
    
    result = image.copy()
    result[mask] = blurred_arr[mask]
    return result


def apply_pixelate(
    image: np.ndarray,
    mask: np.ndarray,
    strength: int = DEFAULT_BLUR_STRENGTH,
) -> np.ndarray:
    """对 mask 区域应用像素化（马赛克）"""
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)
    
    # 像素化：先缩小再放大
    block_size = max(4, strength)
    small_w, small_h = max(1, w // block_size), max(1, h // block_size)
    small = pil_img.resize((small_w, small_h), Image.NEAREST)
    pixelated = small.resize((w, h), Image.NEAREST)
    pixelated_arr = np.array(pixelated)
    
    result = image.copy()
    result[mask] = pixelated_arr[mask]
    return result


def apply_solid_color(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 0, 0),
) -> np.ndarray:
    """对 mask 区域填充纯色"""
    result = image.copy()
    result[mask] = color
    return result


class PrivacyPipeline:
    """隐私过滤流水线"""
    
    def filter_auto(
        self,
        image: np.ndarray,
        blur_type: BlurType = "gaussian",
        blur_strength: int = DEFAULT_BLUR_STRENGTH,
        min_area_ratio: float = AUTO_MASK_MIN_AREA_RATIO,
    ) -> PrivacyFilterResult:
        """
        自动模式隐私过滤：
        1. 调用 SAM3 自动分割
        2. 筛选满足条件的 mask
        3. 对每个 mask 应用模糊/遮挡
        """
        # 获取自动分割结果
        masks: List[MaskResult] = sam3_model.segment_auto(
            image,
            min_area_ratio=min_area_ratio,
        )
        
        # 应用模糊
        result_image = image.copy()
        applied_regions: List[AppliedRegion] = []
        
        for m in masks:
            if blur_type == "gaussian":
                result_image = apply_gaussian_blur(result_image, m.mask, blur_strength)
            elif blur_type == "pixelate":
                result_image = apply_pixelate(result_image, m.mask, blur_strength)
            elif blur_type == "solid":
                result_image = apply_solid_color(result_image, m.mask)
            
            applied_regions.append(AppliedRegion(
                mask_id=m.mask_id,
                bbox=m.bbox,
                area=m.area,
            ))
        
        return PrivacyFilterResult(
            filtered_image=result_image,
            applied_regions=applied_regions,
        )


# 全局实例
privacy_pipeline = PrivacyPipeline()
