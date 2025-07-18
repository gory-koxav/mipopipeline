# cctv_event_detector/inference/strategies/automatic_segmenter.py
import torch
import numpy as np
import cv2
from typing import Dict, Any, List

from .base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage
from config import SAM_MODEL_TYPE, SAM_CHECKPOINT_PATH
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class AutomaticSegmenter(InferenceStrategy):
    """
    SAM의 AutomaticMaskGenerator를 사용하여 이미지에서 객체 마스크들을 자동으로 생성하는 전략입니다.
    이 클래스는 '분할'의 책임만 가집니다.
    """
    def __init__(self):
        print("Initializing Automatic Segmenter...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SAM 자동 마스크 생성기 모델 로드
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH))
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=sam)
        print(f"✅ Automatic SAM model ('{SAM_MODEL_TYPE}') loaded successfully.")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Automatic Segmenter...")

        for capture in captured_images:
            image_id = capture.image_id
            image_data = capture.image_data
            h, w, _ = image_data.shape


            # 이미지 전처리
            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.GaussianBlur(rgb_image, (25, 25), 0)
            
            # "여기에 CUTOFF 함수를 추가

            print(f"Generating masks for {image_id}...")
            raw_masks = self.mask_generator.generate(rgb_image)
            
            # 특정 영역 무시를 위한 마스크 생성
            ignore_mask = np.zeros((h, w), dtype=bool)
            ignore_mask[:int(h * 0.30), :] = True

            # 마스크 필터링
            filtered_masks = self._filter_masks(raw_masks, ignore_mask)
            print(f"Found {len(filtered_masks)} final masks for {image_id}.")

            # 결과를 딕셔너리에 저장
            if image_id not in inference_results:
                inference_results[image_id] = {}
            inference_results[image_id]['pinjig_masks'] = filtered_masks
            
        return inference_results

    def _filter_masks(self, masks: List[Dict[str, Any]], ignore_mask: np.ndarray, 
                      nesting_threshold: float = 0.9, ignore_iou_threshold: float = 0.8) -> List[Dict[str, Any]]:
        # 이 메서드는 분할 로직의 일부이므로 이 클래스에 유지됩니다.
        if not masks: return []
        
        masks_in_roi = [m for m in masks if np.logical_and(m['segmentation'], ignore_mask).sum() / m['area'] < ignore_iou_threshold]
        if not masks_in_roi: return []

        sorted_masks = sorted(masks_in_roi, key=(lambda x: x['area']), reverse=True)
        final_masks_indices = list(range(len(sorted_masks)))
        
        for i in range(len(sorted_masks)):
            if i not in final_masks_indices: continue
            mask_i_seg = sorted_masks[i]['segmentation']
            for j in range(i + 1, len(sorted_masks)):
                if j not in final_masks_indices: continue
                mask_j_seg = sorted_masks[j]['segmentation']
                intersection = np.logical_and(mask_i_seg, mask_j_seg).sum()
                if intersection / sorted_masks[j]['area'] > nesting_threshold:
                    final_masks_indices.remove(j)
        
        return [
            sorted_masks[i] for i in final_masks_indices 
            if sorted_masks[i]['bbox'][2] >= 100 and sorted_masks[i]['bbox'][3] >= 100
        ]