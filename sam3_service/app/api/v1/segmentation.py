"""
分割接口
"""
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage

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


def apply_outline_preview(img_arr: np.ndarray, masks: list, outline_width: int = 3) -> np.ndarray:
    """轮廓描边预览"""
    preview_arr = img_arr.copy()
    outline_color = np.array([255, 255, 0], dtype=np.uint8)  # 黄色高亮
    
    for r in masks:
        mask = r.mask.astype(bool)
        eroded = ndimage.binary_erosion(mask, iterations=outline_width)
        outline = mask ^ eroded
        preview_arr[outline] = outline_color
    
    return preview_arr


def apply_heatmap_preview(img_arr: np.ndarray, masks: list, alpha: float = 0.6) -> np.ndarray:
    """热力图预览：基于距离场的冷暖色渐变"""
    preview_arr = img_arr.copy().astype(np.float32)
    
    # 冷暖色谱：蓝 -> 青 -> 绿 -> 黄 -> 橙
    # 使用 matplotlib 风格的 colormap
    colormap = np.array([
        [0, 0, 255],      # 蓝 (边缘)
        [0, 255, 255],    # 青
        [0, 255, 0],      # 绿
        [255, 255, 0],    # 黄
        [255, 128, 0],    # 橙 (中心)
    ], dtype=np.float32)
    
    for r in masks:
        mask = r.mask.astype(bool)
        if not mask.any():
            continue
        
        # 计算距离场：每个像素到边缘的距离
        dist = ndimage.distance_transform_edt(mask)
        
        # 归一化到 [0, 1]
        max_dist = dist.max()
        if max_dist > 0:
            dist_norm = dist / max_dist
        else:
            dist_norm = dist
        
        # 映射到色谱（线性插值）
        n_colors = len(colormap)
        indices = dist_norm * (n_colors - 1)
        lower = np.floor(indices).astype(int)
        upper = np.ceil(indices).astype(int)
        upper = np.clip(upper, 0, n_colors - 1)
        frac = indices - lower
        
        # 插值颜色
        heat_color = np.zeros((*mask.shape, 3), dtype=np.float32)
        for i in range(3):
            heat_color[..., i] = (
                colormap[lower, i] * (1 - frac) + 
                colormap[upper, i] * frac
            )
        
        # 只在 mask 区域叠加热力图（alpha 混合）
        for i in range(3):
            preview_arr[..., i] = np.where(
                mask,
                preview_arr[..., i] * (1 - alpha) + heat_color[..., i] * alpha,
                preview_arr[..., i]
            )
    
    return np.clip(preview_arr, 0, 255).astype(np.uint8)


@router.post("/text_preview", response_model=TextPreviewResponse)
async def text_preview(
    image: UploadFile = File(...),
    text_prompt: str = Form(default="all objects"),
    preview_mode: str = Form(default="heatmap"),  # "outline" 或 "heatmap"
    max_masks: int = Form(default=AUTO_MASK_MAX_COUNT),
    min_area_ratio: float = Form(default=AUTO_MASK_MIN_AREA_RATIO),
):
    """
    基于文本提示词的分割预览
    
    - preview_mode: "outline"（轮廓描边）或 "heatmap"（热力图渐变）
    """
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

    # 根据模式生成预览
    if preview_mode == "heatmap":
        preview_arr = apply_heatmap_preview(img_arr, results)
    else:
        preview_arr = apply_outline_preview(img_arr, results)

    preview_b64 = encode_image_to_base64(preview_arr)

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
