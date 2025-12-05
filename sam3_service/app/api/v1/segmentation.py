"""
分割接口
"""
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image, ImageDraw
import numpy as np

from ...core.config import AUTO_MASK_MIN_AREA_RATIO, AUTO_MASK_MAX_COUNT
from ...core.image_io import decode_image_from_bytes, encode_image_to_base64, resize_if_needed
from ...core.sam3_model import sam3_model

router = APIRouter(prefix="/segment", tags=["segmentation"])


class MaskInfo(BaseModel):
    mask_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    area: int
    score: float


class SegmentAutoResponse(BaseModel):
    masks: List[MaskInfo]
    image_size: List[int]  # [H, W]


class SegmentPromptResponse(BaseModel):
    masks: List[MaskInfo]
    image_size: List[int]


class TextPreviewResponse(BaseModel):
    preview_image_base64: str
    applied_regions: List[MaskInfo]


@router.post("/auto", response_model=SegmentAutoResponse)
async def segment_auto(
    image: UploadFile = File(...),
    max_masks: int = Form(default=AUTO_MASK_MAX_COUNT),
    min_area_ratio: float = Form(default=AUTO_MASK_MIN_AREA_RATIO),
):
    """
    自动分割（无 prompt）
    """
    # 读取图像
    data = await image.read()
    img_arr = decode_image_from_bytes(data)
    h, w = img_arr.shape[:2]
    
    # 调用模型
    results = sam3_model.segment_auto(
        img_arr,
        min_area_ratio=min_area_ratio,
        max_masks=max_masks,
    )
    
    # 构造响应
    masks = [
        MaskInfo(
            mask_id=r.mask_id,
            bbox=list(r.bbox),
            area=r.area,
            score=r.score,
        )
        for r in results
    ]
    
    return SegmentAutoResponse(masks=masks, image_size=[h, w])


@router.post("/text_preview", response_model=TextPreviewResponse)
async def text_preview(
    image: UploadFile = File(...),
    text_prompt: str = Form(default="all objects"),
    max_masks: int = Form(default=AUTO_MASK_MAX_COUNT),
    min_area_ratio: float = Form(default=AUTO_MASK_MIN_AREA_RATIO),
):
    """基于文本提示词的分割预览：返回叠加 bbox 的预览图，不做模糊"""
    data = await image.read()
    img_arr = decode_image_from_bytes(data)
    img_arr, scale = resize_if_needed(img_arr)
    h, w = img_arr.shape[:2]

    results = sam3_model.segment_auto(
        img_arr,
        min_area_ratio=min_area_ratio,
        max_masks=max_masks,
        text_prompt=text_prompt,
    )

    # 叠加轮廓预览
    pil_img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_img)
    for r in results:
        x1, y1, x2, y2 = r.bbox
        # 画椭圆（内切于 bbox），与 mock 的圆形 mask 风格一致
        draw.ellipse([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    preview_b64 = encode_image_to_base64(np.array(pil_img))

    regions = [
        MaskInfo(
            mask_id=r.mask_id,
            bbox=list(r.bbox),
            area=r.area,
            score=r.score,
        )
        for r in results
    ]

    return TextPreviewResponse(
        preview_image_base64=preview_b64,
        applied_regions=regions,
    )


@router.post("/prompt", response_model=SegmentPromptResponse)
async def segment_prompt(
    image: UploadFile = File(...),
    points: Optional[str] = Form(default=None),  # JSON string: [{"x":..., "y":..., "label":1/-1}]
    boxes: Optional[str] = Form(default=None),   # JSON string: [{"x1":..., "y1":..., "x2":..., "y2":...}]
):
    """
    基于 prompt 的分割（点/框）
    
    TODO: 解析 points/boxes JSON，调用 sam3_model.segment_with_prompts
    """
    # 读取图像
    data = await image.read()
    img_arr = decode_image_from_bytes(data)
    h, w = img_arr.shape[:2]
    
    # TODO: 解析 points/boxes 并调用模型
    # 目前返回空结果
    return SegmentPromptResponse(masks=[], image_size=[h, w])
