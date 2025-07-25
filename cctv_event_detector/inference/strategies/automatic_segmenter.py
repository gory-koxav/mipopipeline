# cctv_event_detector/inference/strategies/automatic_segmenter.py
import torch
import numpy as np
import cv2
from typing import Dict, Any, List

from .base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage
# config 파일에서 TOP_CUTOFF_PERCENT, BOTTOM_CUTOFF_PERCENT 임포트 추가
from config import SAM_MODEL_TYPE, SAM_CHECKPOINT_PATH, TOP_CUTOFF_PERCENT, BOTTOM_CUTOFF_PERCENT
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

    def _apply_top_bottom_gray_padding(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        이미지의 상단과 하단을 지정된 비율만큼 회색으로 채웁니다.
        설정값은 config 파일에서 가져옵니다.

        Args:
            image_rgb (np.ndarray): 원본 RGB 이미지 배열.

        Returns:
            np.ndarray: 상단과 하단이 회색으로 채워진 수정된 이미지 배열.
        """
        # 원본 이미지를 보호하기 위해 복사본 사용
        padded_image = image_rgb.copy()
        height, _ = padded_image.shape[:2]

        # 비율을 실제 y좌표로 변환
        top_end_y = int(height * (TOP_CUTOFF_PERCENT / 100.0))
        bottom_start_y = int(height * (BOTTOM_CUTOFF_PERCENT / 100.0))

        # 회색 컬러 값 (RGB: 114, 114, 114)
        gray_color = [114, 114, 114]

        # 상단과 하단 영역을 회색으로 채우기
        padded_image[0:top_end_y, :] = gray_color
        padded_image[bottom_start_y:height, :] = gray_color

        return padded_image

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Automatic Segmenter...")

        for capture in captured_images:
            image_id = capture.image_id
            image_data = capture.image_data
            h, w, _ = image_data.shape

            # 1. BGR 이미지를 RGB로 변환
            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

            # 2. CUTOFF 함수 적용 (지그류 검출을 위해 이미지 상단과 하단을 회색으로 채우기)
            padded_image = self._apply_top_bottom_gray_padding(rgb_image)

            # 3. 이미지 전처리 (가우시안 블러)
            processed_image = cv2.GaussianBlur(padded_image, (25, 25), 0)

            # 4. 마스크 생성
            print(f"Generating masks for {image_id}...")
            raw_masks = self.mask_generator.generate(processed_image)
            
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