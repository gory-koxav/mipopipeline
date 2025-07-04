import os
import sys

# --- 패키지 루트를 sys.path에 추가하는 로직 ---
# 현재 파일의 절대 경로
current_file_path = os.path.abspath(__file__)
# 'cctv_event_detector' 패키지의 루트 디렉토리를 찾음
# project_root = os.path.join(os.path.dirname(current_file_path), '..', '..', '..') # 3단계 위로

# A more robust way: find the specific marker for your project root
# assuming 'cctv_event_detector' is directly inside your project root
temp_path = os.path.dirname(current_file_path) # strategies
temp_path = os.path.dirname(temp_path) # inference
temp_path = os.path.dirname(temp_path) # cctv_event_detector
project_root = os.path.dirname(temp_path) # L2025022_mipo_operationsystem_uv (the directory containing cctv_event_detector)

project_root = os.path.normpath(project_root)

# project_root가 sys.path에 없으면 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------
# cctv_event_detector/inference/strategies/pinjig_segmenter.py
import torch
import numpy as np
import cv2
import datetime
from typing import Dict, Any, List

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage

class SamPinjigSegmenter(InferenceStrategy):
    """
    Segment Anything Model (SAM)을 사용하여 'everything' 모드로 이미지 분할을 수행하고,
    결과를 필터링하는 전략 클래스입니다.
    """
    def __init__(self, model_type: str = "vit_h", checkpoint: str = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/checkpoints/segmentation/sam_vit_h_4b8939.pth"): #나중에 컨피그로 변경
        print("Initializing SAM model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        
        # # SamAutomaticMaskGenerator의 옵션을 조절하여 초기 노이즈를 줄일 수 있습니다.
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model=sam,
        #     points_per_side=128,
        #     pred_iou_thresh=0.86,
        #     stability_score_thresh=0.92,
        #     crop_n_layers=1,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=50,  # 작은 노이즈 마스크 생성 억제
        # )
        self.mask_generator = SamAutomaticMaskGenerator(model=sam)
        print("SAM model loaded successfully.")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running SAM Pinjig Segmenter...")

        for capture in captured_images:
            image_data = capture.image_data
            h, w, _ = image_data.shape
            
            # --- 관심 없는 영역(Ignore Mask) 정의 ---
            # 예시: 이미지 상단 30% 영역을 무시 영역으로 설정
            ignore_mask = np.zeros((h, w), dtype=bool)
            ignore_mask[:int(h * 0.30), :] = True
            
            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            
            # 이미지 전처리는 결과에 큰 영향을 줍니다. 가우시안 블러를 적용해 세부 노이즈를 줄입니다.
            rgb_image = cv2.GaussianBlur(rgb_image, (25, 25), 0)

            print(f"Generating masks for {capture.image_id}...")
            raw_masks = self.mask_generator.generate(rgb_image)
            print(f"Found {len(raw_masks)} raw masks.")

            # *** 마스크 필터링 실행 ***
            # filtered_masks = self._filter_masks(raw_masks, ignore_mask)
            filtered_masks = raw_masks
            print(f"Found {len(filtered_masks)} filtered masks.")

            # 파이프라인 결과에 필터링된 마스크 추가
            if capture.image_id not in inference_results:
                inference_results[capture.image_id] = {}
            # numpy array는 JSON 직렬화가 안되므로, 필요시 리스트로 변환해야 합니다.
            # 여기서는 객체 그대로 전달합니다.
            inference_results[capture.image_id]['pinjig_masks'] = filtered_masks

            # 필터링된 결과 시각화
            visualized_image = self._draw_masks(image_data.copy(), filtered_masks)
            cv2.imshow(f"Original - {capture.image_id}", image_data)
            cv2.imshow(f"Filtered SAM Segmentation - {capture.image_id}", visualized_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        return inference_results

    def _filter_masks(self, masks: List[Dict[str, Any]], ignore_mask: np.ndarray = None, 
                      nesting_threshold: float = 0.9, ignore_iou_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        주어진 기준에 따라 마스크를 필터링합니다.
        1. '무시 영역'과 많이 겹치는 마스크를 제거합니다.
        2. 큰 마스크에 포함된 작은 마스크를 제거합니다.
        """
        if not masks:
            return []

        # 1. '무시 영역(ignore_mask)'과 많이 겹치는 마스크를 먼저 제거
        if ignore_mask is not None:
            masks_in_roi = []
            for mask in masks:
                intersection = np.logical_and(mask['segmentation'], ignore_mask).sum()
                # (교차 영역 / 마스크 영역)이 임계값보다 작으면 통과
                if intersection / mask['area'] < ignore_iou_threshold:
                    masks_in_roi.append(mask)
        else:
            masks_in_roi = masks # 무시 영역이 없으면 모두 통과

        if not masks_in_roi:
            return []

        # 2. 남은 마스크들을 대상으로 중첩(nested) 마스크 제거
        # 면적이 넓은 순으로 정렬
        sorted_masks = sorted(masks_in_roi, key=(lambda x: x['area']), reverse=True)
        
        final_masks_indices = list(range(len(sorted_masks)))
        for i in range(len(sorted_masks)):
            if i not in final_masks_indices:
                continue
            
            mask_i_seg = sorted_masks[i]['segmentation']
            for j in range(i + 1, len(sorted_masks)):
                if j not in final_masks_indices:
                    continue
                
                mask_j_seg = sorted_masks[j]['segmentation']
                intersection = np.logical_and(mask_i_seg, mask_j_seg).sum()
                
                # mask_j가 mask_i에 거의 포함되는 경우 j를 제거
                if intersection / sorted_masks[j]['area'] > nesting_threshold:
                    final_masks_indices.remove(j)
        
        return [sorted_masks[i] for i in final_masks_indices]

    def _draw_masks(self, image: np.ndarray, masks: List[Dict[str, Any]]) -> np.ndarray:
        if len(masks) == 0:
            return image
        
        overlay = image.copy()
        # 이미 면적 순으로 정렬되어 있으므로 다시 정렬할 필요 없음
        for ann in masks:
            m = ann['segmentation']
            color = np.random.randint(0, 256, (1, 3)).tolist()[0]
            overlay[m] = color
        
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)


# if __name__ == "__main__" 블록은 이전과 동일하게 사용하시면 됩니다.
# 이미지 경로와 체크포인트 경로만 확인해주세요.
if __name__ == "__main__":
    import cv2
    import datetime
    # --- 실행 전 설정 ---
    # 1. 테스트할 이미지 파일 경로를 지정하세요.
    image_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/D04_20250703165331.jpg"
    
    # 2. 다운로드한 SAM 모델 체크포인트 파일의 경로를 지정하세요.
    checkpoint_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/checkpoints/segmentation/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # --- 테스트 코드 ---
    print("Loading test image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        # 실제 Segmenter 객체 생성
        segmenter = SamPinjigSegmenter(model_type=model_type, checkpoint=checkpoint_path)

        # 테스트용 이미지 객체 생성
        test_images = [
            CapturedImage(
                image_id="test_img_01", 
                camera_name="test_cam", 
                image_data=image, 
                captured_at=datetime.datetime.now()
            )
        ]
        
        # 분할 실행 및 결과 확인
        results = segmenter.run(test_images, {})
        print("\n--- Pipeline Results ---")
        # 최종적으로 inference_results에 'pinjig_masks' 키로 결과가 저장된 것을 확인
        for image_id, result_data in results.items():
            if 'pinjig_masks' in result_data:
                print(f"Image ID: {image_id}, Found {len(result_data['pinjig_masks'])} final masks.")