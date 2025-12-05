"""
隐私过滤接口
"""
from typing import List, Literal

from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel

from ...core.config import DEFAULT_BLUR_STRENGTH, AUTO_MASK_MIN_AREA_RATIO
from ...core.image_io import decode_image_from_bytes, encode_image_to_base64, resize_if_needed
from ...core.pipeline_privacy import privacy_pipeline, BlurType

router = APIRouter(prefix="/privacy", tags=["privacy"])


class AppliedRegionInfo(BaseModel):
    mask_id: int
    bbox: List[int]
    area: int


class PrivacyFilterResponse(BaseModel):
    filtered_image_base64: str
    applied_regions: List[AppliedRegionInfo]


@router.post("/filter", response_model=PrivacyFilterResponse)
async def privacy_filter(
    image: UploadFile = File(...),
    mode: str = Form(default="auto"),  # 目前只支持 "auto"
    blur_type: BlurType = Form(default="gaussian"),
    blur_strength: int = Form(default=DEFAULT_BLUR_STRENGTH),
    min_area_ratio: float = Form(default=AUTO_MASK_MIN_AREA_RATIO),
    text_prompt: str = Form(default="all objects"),
):
    """
    隐私过滤接口
    
    - mode: 目前只支持 "auto"（自动分割所有区域）
    - blur_type: gaussian / pixelate / solid
    - blur_strength: 模糊强度
    - min_area_ratio: 最小 mask 面积占比
    """
    # 读取并预处理图像
    data = await image.read()
    img_arr = decode_image_from_bytes(data)
    img_arr, scale = resize_if_needed(img_arr)
    
    # 调用隐私过滤流水线
    result = privacy_pipeline.filter_auto(
        image=img_arr,
        blur_type=blur_type,
        blur_strength=blur_strength,
        min_area_ratio=min_area_ratio,
        text_prompt=text_prompt,
    )
    
    # 编码结果图像
    filtered_b64 = encode_image_to_base64(result.filtered_image)
    
    # 构造响应
    regions = [
        AppliedRegionInfo(
            mask_id=r.mask_id,
            bbox=list(r.bbox),
            area=r.area,
        )
        for r in result.applied_regions
    ]
    
    return PrivacyFilterResponse(
        filtered_image_base64=filtered_b64,
        applied_regions=regions,
    )
