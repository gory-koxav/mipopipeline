import os
import sys
import torch
import numpy as np
import cv2
import datetime
from typing import Dict, Any, List

# --- 패키지 루트를 sys.path에 추가하는 로직 ---
# 현재 파일의 절대 경로
current_file_path = os.path.abspath(__file__)
# 'cctv_event_detector' 패키지의 루트 디렉토리를 찾음
temp_path = os.path.dirname(current_file_path)
temp_path = os.path.dirname(temp_path)
temp_path = os.path.dirname(temp_path)
project_root = os.path.dirname(temp_path)
project_root = os.path.normpath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------

# +++ 추가된 임포트 +++
from ultralytics import YOLO

# cctv_event_detector/inference/strategies/pinjig_segmenter.py
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage

class SamPinjigSegmenter(InferenceStrategy):
    """
    Segment Anything Model (SAM)을 사용하여 이미지 분할을 수행하고,
    분할된 영역을 YOLO 모델로 분류한 뒤 저장하는 전략 클래스입니다.
    """
    def __init__(self,
                 model_type: str = "vit_h",
                 checkpoint: str = "/path/to/your/sam_checkpoint.pth",
                 cls_model_path: str = "/path/to/your/yolo_cls_model.pt"): # YOLO 모델 경로 추가
        
        # SAM 모델 초기화
        print("Initializing SAM model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=sam)
        print("SAM model loaded successfully.")

        # +++ YOLO 분류 모델 초기화 +++
        print("Initializing YOLO classification model...")
        self.cls_model = YOLO(cls_model_path)
        print("YOLO classification model loaded successfully.")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running SAM Pinjig Segmenter...")

        for capture in captured_images:
            image_data = capture.image_data
            h, w, _ = image_data.shape
            
            # --- 관심 없는 영역(Ignore Mask) 정의 ---
            ignore_mask = np.zeros((h, w), dtype=bool)
            ignore_mask[:int(h * 0.30), :] = True
            
            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.GaussianBlur(rgb_image, (25, 25), 0)

            print(f"Generating masks for {capture.image_id}...")
            raw_masks = self.mask_generator.generate(rgb_image)
            print(f"Found {len(raw_masks)} raw masks.")

            # *** 마스크 필터링 실행 ***
            filtered_masks = self._filter_masks(raw_masks, ignore_mask)
            print(f"Found {len(filtered_masks)} filtered masks.")

            if capture.image_id not in inference_results:
                inference_results[capture.image_id] = {}
            inference_results[capture.image_id]['pinjig_masks'] = filtered_masks
            
            # +++ 필터링된 마스크를 크롭, 리사이즈, 분류 후 저장하는 기능 호출 +++
            if filtered_masks:
                self._classify_and_save_masks(
                    original_image=image_data,
                    masks=filtered_masks,
                    image_id=capture.image_id
                )

            # 필터링된 결과 시각화
            visualized_image = self._draw_masks(image_data.copy(), filtered_masks)
            cv2.imshow(f"Original - {capture.image_id}", image_data)
            cv2.imshow(f"Filtered SAM Segmentation - {capture.image_id}", visualized_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        return inference_results

    def _classify_and_save_masks(self, original_image: np.ndarray, masks: List[Dict[str, Any]], image_id: str):
        """
        마스크 영역을 자르고, 크기를 조절하고, 분류한 뒤, 결과를 적어 저장합니다.
        
        Args:
            original_image (np.ndarray): 원본 BGR 이미지.
            masks (List[Dict[str, Any]]): 필터링된 마스크 정보 리스트.
            image_id (str): 현재 처리 중인 이미지의 ID.
        """
        # 결과를 저장할 고유한 디렉토리 생성
        output_dir = os.path.join("classification_results", f"{image_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving classified crops to: {output_dir}")

        for i, mask_ann in enumerate(masks):
            # 1. 필터링이 완료된 마스크 영역을 원본 이미지에서 크롭
            bbox = mask_ann.get('bbox')
            if not bbox:
                print(f"Warning: Mask {i} is missing 'bbox'. Skipping.")
                continue
            x, y, w, h = bbox

            # --- ★★★ 요청에 따라 수정된 크롭 로직 ★★★ ---
            # 먼저, 경계 상자 기준으로 사각형 영역을 잘라냅니다.
            rect_crop = original_image[y:y+h, x:x+w]

            # 전체 이미지 크기의 boolean 마스크에서 해당 경계 상자 부분만 잘라냅니다.
            segmentation_mask = mask_ann['segmentation']
            cropped_mask = segmentation_mask[y:y+h, x:x+w]

            # 잘라낸 사각형과 같은 크기의 검은색 배경 이미지를 준비합니다.
            black_background = np.zeros_like(rect_crop)

            # boolean 마스크를 사용해, 마스크 영역에 해당하는 픽셀만 원본에서 가져와 검은 배경 위에 놓습니다.
            black_background[cropped_mask] = rect_crop[cropped_mask]
            
            # 이 최종 이미지를 이후 단계에서 사용합니다.
            cropped_image = black_background
            # --- ★★★ 로직 수정 완료 ★★★ ---
            
            if cropped_image.size == 0:
                print(f"Warning: Skipped empty crop for mask {i} at bbox {bbox}.")
                continue

            # 2. 크롭된 이미지들을 512x512px 사이즈로 리사이즈
            resized_image = cv2.resize(cropped_image, (512, 512))

            # 3. 이미지들을 YOLO classification 모델에 추론
            results = self.cls_model(resized_image, verbose=False)
            r = results[0]

            # 4. top-5 결과를 각 이미지에 검은색 테두리가 있는 흰색 글자로 적기
            annotated_image = resized_image.copy()
            pos_y = 40
            font_scale = 1.0
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            white_color = (255, 255, 255)
            black_color = (0, 0, 0)
            outline_thickness = 3 # 테두리 두께

            for j, idx in enumerate(r.probs.top5):
                class_name = self.cls_model.names[(int(idx))]
                conf = r.probs.top5conf[(j)].item()
                text = f"{j+1}. {class_name}: {conf:.2f}"

                # 먼저 검은색 테두리 글자를 약간 더 두껍게 그림
                cv2.putText(annotated_image, text, (20, pos_y), font, font_scale, black_color, outline_thickness, cv2.LINE_AA)
                # 그 위에 흰색 실제 글자를 그림
                cv2.putText(annotated_image, text, (20, pos_y), font, font_scale, white_color, thickness, cv2.LINE_AA)
                pos_y += 40

            # 5. 결과 이미지 저장
            output_filename = f"mask_{i:03d}_classified.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, annotated_image)

        print(f"Finished processing and saving {len(masks)} cropped images.")


    def _filter_masks(self, masks: List[Dict[str, Any]], ignore_mask: np.ndarray = None, 
                      nesting_threshold: float = 0.9, ignore_iou_threshold: float = 0.8) -> List[Dict[str, Any]]:
        if not masks:
            return []
        
        if ignore_mask is not None:
            masks_in_roi = [
                m for m in masks 
                if np.logical_and(m['segmentation'], ignore_mask).sum() / m['area'] < ignore_iou_threshold
            ]
        else:
            masks_in_roi = masks

        if not masks_in_roi:
            return []

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
                if intersection / sorted_masks[j]['area'] > nesting_threshold:
                    final_masks_indices.remove(j)
        
        return [sorted_masks[i] for i in final_masks_indices]

    def _draw_masks(self, image: np.ndarray, masks: List[Dict[str, Any]]) -> np.ndarray:
        if len(masks) == 0:
            return image
        
        overlay = image.copy()
        for ann in masks:
            m = ann['segmentation']
            color = np.random.randint(0, 256, (1, 3)).tolist()[0]
            overlay[m] = color
        
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

if __name__ == "__main__":
    # --- 실행 전 설정 ---
    # 1. 테스트할 이미지 파일 경로를 지정하세요.
    image_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/D04_20250703165331.jpg"
    
    # 2. 다운로드한 SAM 모델 체크포인트 파일의 경로를 지정하세요.
    sam_checkpoint_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/checkpoints/segmentation/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # +++ 3. YOLO Classification 모델 체크포인트 파일의 경로를 지정하세요. +++
    cls_model_path = "/home/ksoeadmin/Projects/PYPJ/sam/checkpoints/cls/best.pt"

    # --- 테스트 코드 ---
    print("Loading test image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        # 실제 Segmenter 객체 생성 (YOLO 모델 경로 전달)
        segmenter = SamPinjigSegmenter(
            model_type=model_type, 
            checkpoint=sam_checkpoint_path,
            cls_model_path=cls_model_path
        )

        # 테스트용 이미지 객체 생성
        test_images = [
            CapturedImage(
                image_id="test_img_01",
                camera_name="test_cam",
                image_data=image,
                captured_at=datetime.datetime.now()
            )
        ]
        
        # 분할 및 분류 실행
        results = segmenter.run(test_images, {})
        
        print("\n--- Pipeline Results ---")
        for image_id, result_data in results.items():
            if 'pinjig_masks' in result_data:
                print(f"Image ID: {image_id}, Found {len(result_data['pinjig_masks'])} final masks.")