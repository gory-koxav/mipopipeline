import os
import sys
import torch
import numpy as np
import cv2
import datetime
from typing import Dict, Any, List
import glob

# --- 패키지 루트를 sys.path에 추가하는 로직 ---
current_file_path = os.path.abspath(__file__)
temp_path = os.path.dirname(current_file_path)
temp_path = os.path.dirname(temp_path)
temp_path = os.path.dirname(temp_path)
project_root = os.path.dirname(temp_path)
project_root = os.path.normpath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------

# --- config 파일에서 설정값 불러오기 ---
from config import (
    SAM_MODEL_TYPE,
    SAM_CHECKPOINT_PATH,
    SAM_YOLO_CLS_MODEL_PATH,
    SAM_CLASSIFICATION_OUTPUT_DIR
)
# -----------------------------------------------

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage


class SamPinjigSegmenter(InferenceStrategy):
    """
    Segment Anything Model (SAM)을 사용하여 이미지 분할을 수행하고,
    분할된 영역을 YOLO 모델로 분류한 뒤 저장하는 전략 클래스입니다.
    """
    def __init__(self):
        """
        config 파일로부터 모델 경로 및 기타 설정을 로드하여 모델을 초기화합니다.
        """
        print("Initializing models from config...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 설정 파일(config.py)에서 SAM 모델 경로 및 타입 로드
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH))
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=sam)
        print(f"SAM model ('{SAM_MODEL_TYPE}') loaded successfully from: {SAM_CHECKPOINT_PATH}")

        # 설정 파일(config.py)에서 YOLO 분류 모델 경로 로드
        self.cls_model = YOLO(SAM_YOLO_CLS_MODEL_PATH)
        print(f"YOLO classification model loaded successfully from: {SAM_YOLO_CLS_MODEL_PATH}")

        # 설정 파일(config.py)에서 출력 디렉토리 로드
        self.output_dir = SAM_CLASSIFICATION_OUTPUT_DIR
        print(f"Output directory set to: {self.output_dir}")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running SAM Pinjig Segmenter...")

        for capture in captured_images:
            image_data = capture.image_data
            h, w, _ = image_data.shape

            ignore_mask = np.zeros((h, w), dtype=bool)
            ignore_mask[:int(h * 0.30), :] = True

            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.GaussianBlur(rgb_image, (25, 25), 0)

            print(f"Generating masks for {capture.image_id}...")
            raw_masks = self.mask_generator.generate(rgb_image)
            print(f"Found {len(raw_masks)} raw masks.")

            filtered_masks = self._filter_masks(raw_masks, ignore_mask)
            print(f"Found {len(filtered_masks)} filtered masks after all filters.")

            if capture.image_id not in inference_results:
                inference_results[capture.image_id] = {}
            inference_results[capture.image_id]['pinjig_masks'] = filtered_masks
            
            # (선택) 인식된 마스크 저장
            if filtered_masks:
                self._classify_and_save_masks(
                    original_image=image_data,
                    masks=filtered_masks,
                    image_id=capture.image_id
                )

        return inference_results

    def _classify_and_save_masks(self, original_image: np.ndarray, masks: List[Dict[str, Any]], image_id: str):
        # config에서 불러온 기본 출력 디렉토리 하위에 타임스탬프 폴더 생성
        # timed_output_dir = self.output_dir / f"{image_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timed_output_dir = self.output_dir / image_id
        timed_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving classified crops to: {timed_output_dir}")

        # ★★★ 전체 마스크 오버레이 이미지 저장 ★★★
        if masks:
            overlay_image = self._draw_masks(original_image.copy(), masks)
            overlay_filename = "_OVERLAY_RESULT.jpg"
            overlay_path = timed_output_dir / overlay_filename
            cv2.imwrite(str(overlay_path), overlay_image)
            print(f"Saved overlay image to: {overlay_path}")

        # ★★★ 마스크별 크롭 이미지 저장 ★★★
        for i, mask_ann in enumerate(masks):
            bbox = mask_ann.get('bbox')
            if not bbox:
                print(f"Warning: Mask {i} is missing 'bbox'. Skipping.")
                continue
            x, y, w, h = bbox

            rect_crop = original_image[y:y+h, x:x+w]
            segmentation_mask = mask_ann['segmentation']
            cropped_mask = segmentation_mask[y:y+h, x:x+w]
            black_background = np.zeros_like(rect_crop)
            black_background[cropped_mask] = rect_crop[cropped_mask]
            cropped_image = black_background
            
            if cropped_image.size == 0:
                print(f"Warning: Skipped empty crop for mask {i} at bbox {bbox}.")
                continue

            resized_image = cv2.resize(cropped_image, (512, 512))
            results = self.cls_model(resized_image, verbose=False)
            r = results[0]

            annotated_image = resized_image.copy()
            pos_y = 40
            font_scale = 1.0
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            white_color = (255, 255, 255)
            black_color = (0, 0, 0)
            outline_thickness = 3

            for j, idx in enumerate(r.probs.top5):
                class_name = self.cls_model.names[int(idx)]
                conf = r.probs.top5conf[j].item()
                text = f"{j+1}. {class_name}: {conf:.2f}"
                cv2.putText(annotated_image, text, (20, pos_y), font, font_scale, black_color, outline_thickness, cv2.LINE_AA)
                cv2.putText(annotated_image, text, (20, pos_y), font, font_scale, white_color, thickness, cv2.LINE_AA)
                pos_y += 40

            top1_class_name = self.cls_model.names[r.probs.top1]
            base_filename = f"mask_{i:03d}_classified.png"
            output_filename = f"{top1_class_name}_{base_filename}"
            
            output_path = timed_output_dir / output_filename
            cv2.imwrite(str(output_path), annotated_image)

        print(f"Finished processing and saving {len(masks)} cropped images.")

    def _filter_masks(self, masks: List[Dict[str, Any]], ignore_mask: np.ndarray = None, 
                      nesting_threshold: float = 0.9, ignore_iou_threshold: float = 0.8) -> List[Dict[str, Any]]:
        if not masks: return []
        
        if ignore_mask is not None:
            masks_in_roi = [m for m in masks if np.logical_and(m['segmentation'], ignore_mask).sum() / m['area'] < ignore_iou_threshold]
        else:
            masks_in_roi = masks
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
        
        final_masks = [
            sorted_masks[i] for i in final_masks_indices 
            if sorted_masks[i]['bbox'][2] >= 100 and sorted_masks[i]['bbox'][3] >= 100
        ]

        return final_masks

    def _draw_masks(self, image: np.ndarray, masks: List[Dict[str, Any]]) -> np.ndarray:
        if len(masks) == 0: return image
        overlay = image.copy()
        for ann in masks:
            m = ann['segmentation']
            color = np.random.randint(0, 256, (1, 3)).tolist()[0]
            overlay[m] = color
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

if __name__ == "__main__":
    # main에서 직접 실행할 때만 사용되는 데이터 디렉토리 경로는 여기에 유지합니다.
    data_directory = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data"

    # 이제 SamPinjigSegmenter는 config 파일에서 모든 설정을 가져오므로,
    # 클래스를 생성할 때 인자를 전달할 필요가 없습니다.
    print("Initializing models...")
    segmenter = SamPinjigSegmenter()
    print("Models loaded successfully. Starting batch processing...")
    
    image_paths = glob.glob(os.path.join(data_directory, '*.jpg'))
    if not image_paths:
        print(f"No JPG files found in {data_directory}")

    for image_path in image_paths:
        print(f"\n{'='*20}\nProcessing: {image_path}\n{'='*20}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}. Skipping.")
            continue
        
        image_id = os.path.basename(image_path)
        
        test_images = [
            CapturedImage(
                image_id=image_id,
                camera_name="batch_cam",
                image_data=image,
                captured_at=datetime.datetime.now()
            )
        ]
        
        results = segmenter.run(test_images, {})
        
        print("\n--- Pipeline Results ---")
        for res_id, result_data in results.items():
            if 'pinjig_masks' in result_data:
                print(f"Image ID: {res_id}, Found {len(result_data['pinjig_masks'])} final masks.")

    print("\nAll tasks completed.")