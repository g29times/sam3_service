"""
SAM3 模型封装（单例）
支持 mock 和 real 两种模式，通过环境变量 SAM3_MODE 控制
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from .config import DEVICE, SAM3_HF_REPO, SAM3_MODE


@dataclass
class MaskResult:
    """单个 mask 结果"""
    mask_id: int
    mask: np.ndarray  # (H, W) bool 或 0/1
    bbox: tuple  # (x1, y1, x2, y2)
    area: int
    score: float = 1.0


class SAM3Model:
    """SAM3 模型单例封装"""
    
    _instance: Optional["SAM3Model"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.processor = None
        self.device = DEVICE
        self.hf_repo = SAM3_HF_REPO
        self.mode = SAM3_MODE
        self._loaded = False
    
    def load(self) -> bool:
        """加载模型"""
        if self.mode == "real":
            return self._load_real()
        else:
            return self._load_mock()
    
    def _load_mock(self) -> bool:
        """Mock 模式：不加载真实模型"""
        print(f"[SAM3Model] Mock load: hf_repo={self.hf_repo}, device={self.device}")
        self._loaded = True
        return True
    
    def _load_real(self) -> bool:
        """Real 模式：加载真实 SAM3 模型"""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            print(f"[SAM3Model] Loading real model from {self.hf_repo}...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            print(f"[SAM3Model] Model loaded successfully, device={self.device}")
            self._loaded = True
            return True
        except Exception as e:
            print(f"[SAM3Model] Failed to load model: {e}")
            raise
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def segment_auto(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.01,
        max_masks: int = 50,
        text_prompt: str = "all objects",
    ) -> List[MaskResult]:
        """
        自动分割。
        
        在 real 模式下使用 SAM3 的文本 prompt 能力，默认 prompt 为 "all objects"。
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self.mode == "real":
            return self._segment_real(image, text_prompt, min_area_ratio, max_masks)
        else:
            return self._segment_mock(image, min_area_ratio, max_masks)
    
    def _segment_mock(
        self,
        image: np.ndarray,
        min_area_ratio: float,
        max_masks: int,
    ) -> List[MaskResult]:
        """Mock 分割：返回一个假的中心区域 mask"""
        h, w = image.shape[:2]
        
        mock_mask = np.zeros((h, w), dtype=bool)
        cx, cy = w // 2, h // 2
        r = min(h, w) // 4
        y_indices, x_indices = np.ogrid[:h, :w]
        circle = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= r ** 2
        mock_mask[circle] = True
        
        return [
            MaskResult(
                mask_id=0,
                mask=mock_mask,
                bbox=(cx - r, cy - r, cx + r, cy + r),
                area=int(mock_mask.sum()),
                score=0.95,
            )
        ]
    
    def _segment_real(
        self,
        image: np.ndarray,
        text_prompt: str,
        min_area_ratio: float,
        max_masks: int,
    ) -> List[MaskResult]:
        """真实 SAM3 分割"""
        h, w = image.shape[:2]
        total_area = h * w
        min_area = int(total_area * min_area_ratio)
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image)
        
        # 设置图像并执行分割
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        masks = output["masks"]  # List of mask arrays
        boxes = output["boxes"]  # List of bounding boxes
        scores = output["scores"]  # List of confidence scores
        
        results = []
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # mask 可能是 tensor，转为 numpy
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            mask = mask.astype(bool)
            
            # 如果 mask 是 3D (1, H, W)，squeeze 掉
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            
            area = int(mask.sum())
            if area < min_area:
                continue
            
            # box 格式：[x1, y1, x2, y2]
            if hasattr(box, "cpu"):
                box = box.cpu().numpy()
            bbox = tuple(int(v) for v in box[:4])
            
            if hasattr(score, "item"):
                score = score.item()
            
            results.append(MaskResult(
                mask_id=i,
                mask=mask,
                bbox=bbox,
                area=area,
                score=float(score),
            ))
        
        # 按 score 排序，取 top max_masks
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_masks]
    
    def segment_with_prompts(
        self,
        image: np.ndarray,
        points: Optional[List[dict]] = None,
        boxes: Optional[List[dict]] = None,
    ) -> List[MaskResult]:
        """
        基于 prompt 的分割（点/框）。
        
        TODO: 后续实现
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # 目前返回空列表，后续可扩展
        return []


# 全局单例
sam3_model = SAM3Model()
