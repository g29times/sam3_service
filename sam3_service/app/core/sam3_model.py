"""
SAM3 模型封装（单例）
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import DEVICE, MODEL_DIR, SAM3_HF_REPO, SAM3_REPO_DIR


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
        self._loaded = False
    
    def load(self) -> bool:
        """
        加载模型。
        TODO: 对接官方 SAM3 推理接口
        """
        # TODO: 实现真正的模型加载
        # SAM3 官方用法：
        # import sys
        # sys.path.insert(0, str(SAM3_REPO_DIR))
        # from sam3.model_builder import build_sam3_image_model
        # from sam3.model.sam3_image_processor import Sam3Processor
        # self.model = build_sam3_image_model()  # 会自动从 HuggingFace 下载
        # self.processor = Sam3Processor(self.model)
        
        print(f"[SAM3Model] Mock load: hf_repo={self.hf_repo}, device={self.device}")
        self._loaded = True
        return True
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def segment_auto(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.01,
        max_masks: int = 50,
    ) -> List[MaskResult]:
        """
        自动分割（无 prompt）。
        
        TODO: 对接官方 SAM3 automatic mask generator
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        h, w = image.shape[:2]
        total_area = h * w
        min_area = int(total_area * min_area_ratio)
        
        # TODO: 替换为真实推理
        # masks = self.predictor.generate(image)
        # results = []
        # for i, m in enumerate(masks):
        #     if m["area"] < min_area:
        #         continue
        #     results.append(MaskResult(
        #         mask_id=i,
        #         mask=m["segmentation"],
        #         bbox=m["bbox"],
        #         area=m["area"],
        #         score=m.get("predicted_iou", 1.0),
        #     ))
        # return results[:max_masks]
        
        # Mock: 返回一个假的中心区域 mask
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
    
    def segment_with_prompts(
        self,
        image: np.ndarray,
        points: Optional[List[dict]] = None,
        boxes: Optional[List[dict]] = None,
    ) -> List[MaskResult]:
        """
        基于 prompt 的分割。
        
        TODO: 对接官方 SAM3 predictor
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # TODO: 实现真正的 prompt 分割
        # 目前返回空列表
        return []


# 全局单例
sam3_model = SAM3Model()
