# cctv_event_detector/inference/strategies/mask_classifier.py
import numpy as np
import cv2
from typing import Dict, Any, List

from .base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage
from config import SAM_YOLO_CLS_MODEL_PATH, SAM_CLASSIFICATION_OUTPUT_DIR
from ultralytics import YOLO

class MaskClassifier(InferenceStrategy):
    """
    입력된 마스크를 기반으로 객체를 크롭하고, YOLO 분류 모델로 추론한 뒤,
    결과를 시각화하여 저장하는 전략입니다. 이 클래스는 '분류 및 시각화'의 책임만 가집니다.
    """
    def __init__(self):
        print("Initializing Mask Classifier...")
        self.cls_model = YOLO(SAM_YOLO_CLS_MODEL_PATH)
        print(f"✅ YOLO classification model loaded successfully.")

        self.output_dir = SAM_CLASSIFICATION_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Classification output dir: {self.output_dir}")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Mask Classifier...")

        for capture in captured_images:
            image_id = capture.image_id
            original_image = capture.image_data

            masks = inference_results.get(image_id, {}).get("pinjig_masks", [])

            if masks:
                classification_results = self._perform_classification(original_image, masks, image_id)
                
                if classification_results:
                    inference_results[image_id]['pinjig_classifications'] = classification_results
                    self._visualize_results(original_image, masks, classification_results, image_id)

        return inference_results

    def _perform_classification(self, original_image: np.ndarray, masks: List[Dict[str, Any]], image_id: str) -> List[Dict[str, Any]]:
        timed_output_dir = self.output_dir / image_id
        timed_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Classifying {len(masks)} masks for '{image_id}'...")

        classifications = []
        for i, mask_ann in enumerate(masks):
            bbox = mask_ann.get('bbox')
            if not bbox: continue
            
            # --- 🐛 에러 해결 1: bbox 좌표를 정수형으로 변환 ---
            # segmentation 모델의 결과값인 float를 int로 변환하여 TypeError 방지
            try:
                x, y, w, h = map(int, bbox)
            except (ValueError, TypeError):
                print(f"⚠️ Skipping mask with invalid bbox data: {bbox}")
                continue

            # --- 🐛 에러 해결 2: bbox 크기 유효성 검사 ---
            # 너비나 높이가 0 이하인 경우, crop 시 에러가 발생하므로 이를 방지
            if w <= 0 or h <= 0:
                print(f"⚠️ Skipping mask with invalid bbox dimensions: w={w}, h={h}")
                continue
            
            rect_crop = original_image[y:y+h, x:x+w]
            cropped_mask = mask_ann['segmentation'][y:y+h, x:x+w]
            black_background = np.zeros_like(rect_crop)
            black_background[cropped_mask] = rect_crop[cropped_mask]
            
            if black_background.size == 0: continue

            resized_image = cv2.resize(black_background, (512, 512))
            results = self.cls_model(resized_image, verbose=False)
            r = results[0]

            top1_class_name = self.cls_model.names[r.probs.top1]
            top1_conf = r.probs.top1conf.item()
            
            color = np.random.uniform(0, 255, 3).tolist()

            classifications.append({
                "mask_index": i,
                "top1_class": top1_class_name,
                "confidence": top1_conf,
                "top5_probs": r.probs.top5conf.tolist(),
                "color": color
            })
            
            annotated_image = self._annotate_image(resized_image, r)
            output_filename = f"{top1_class_name}_mask_{i:03d}.png"
            cv2.imwrite(str(timed_output_dir / output_filename), annotated_image)
        
        return classifications

    def _visualize_results(self, original_image: np.ndarray, masks: List[Dict[str, Any]], classifications: List[Dict[str, Any]], image_id: str):
        timed_output_dir = self.output_dir / image_id
        overlay_image = original_image.copy()
        
        # --- ✨ 개선: 이미지 경계 보정을 위해 이미지 크기 가져오기 ---
        img_h, img_w = overlay_image.shape[:2]

        # --- 🎨 1단계: 모든 마스크 그리기 ---
        for i, classification in enumerate(classifications):
            mask = masks[i]['segmentation']
            color = np.array(classification['color'])
            overlay_image[mask] = overlay_image[mask] * 0.4 + color * 0.6
        
        # --- ✍️ 2단계: 연결선 및 텍스트 그리기 ---
        for i, classification in enumerate(classifications):
            mask = masks[i]['segmentation']
            bbox = list(map(int, masks[i]['bbox']))
            x, y, w, h = bbox
            color = classification['color']

            # --- 마스크 중심점 계산 (연결선 끝점) ---
            try:
                M = cv2.moments(mask.astype(np.uint8))
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                cX, cY = x + w // 2, y + h // 2

            # --- 텍스트 정보 계산 ---
            text = f"{classification['top1_class']} ({classification['confidence']:.2f})"
            font_scale = 0.6
            font_thickness = 1
            (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # --- 💡 핵심 개선: 경계 보정 로직 ---
            # 1. 이상적인 텍스트 위치 계산 (기존 로직)
            ideal_text_pos_x = x
            ideal_text_pos_y = y - 10 if y > (h_text + 20) else y + h + h_text + 10

            # 2. 배경 사각형 전체 크기 계산 (패딩 포함)
            rect_x1 = ideal_text_pos_x - 5
            rect_y1 = ideal_text_pos_y - h_text - 5
            rect_x2 = ideal_text_pos_x + w_text + 5
            rect_y2 = ideal_text_pos_y + 5
            
            # 3. 이미지 경계를 벗어나는 정도(offset) 계산
            dx = 0
            if rect_x1 < 0: dx = -rect_x1  # 왼쪽 오버플로우
            if rect_x2 > img_w: dx = img_w - rect_x2 # 오른쪽 오버플로우
            
            dy = 0
            if rect_y1 < 0: dy = -rect_y1 # 위쪽 오버플로우
            if rect_y2 > img_h: dy = img_h - rect_y2 # 아래쪽 오버플로우

            # 4. 계산된 offset으로 모든 관련 좌표 보정
            final_text_pos = (ideal_text_pos_x + dx, ideal_text_pos_y + dy)
            final_rect_pos1 = (rect_x1 + dx, rect_y1 + dy)
            final_rect_pos2 = (rect_x2 + dx, rect_y2 + dy)
            # -----------------------------------------------

            # --- 그리기 ---
            # 1. 연결선 그리기 (보정된 위치 기반)
            line_start_point = (final_text_pos[0] + w_text // 2, final_rect_pos2[1] if final_text_pos[1] > y else final_rect_pos1[1])
            # cv2.line(overlay_image, line_start_point, (cX, cY), (0, 0, 0), 4) # 테두리
            # cv2.line(overlay_image, line_start_point, (cX, cY), (255, 255, 255), 2) # 중심선
            cv2.line(overlay_image, line_start_point, (cX, cY), (255, 255, 255), 5) # 테두리
            cv2.line(overlay_image, line_start_point, (cX, cY), color, 3) # 중심선

            # 2. 배경 사각형 그리기 (보정된 위치 기반)
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
            cv2.rectangle(overlay_image, final_rect_pos1, final_rect_pos2, color, -1)

            # 3. 텍스트 그리기 (보정된 위치 기반)
            cv2.putText(overlay_image, text, final_text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

        cv2.imwrite(str(timed_output_dir / "_OVERLAY_RESULT.jpg"), overlay_image)
        print(f"✅ 최종 시각화 결과 저장 완료: {timed_output_dir / '_OVERLAY_RESULT.jpg'}")


    def _annotate_image(self, image: np.ndarray, result) -> np.ndarray:
        annotated_image = image.copy()
        pos_y = 40
        for j, idx in enumerate(result.probs.top5):
            text = f"{j+1}. {self.cls_model.names[int(idx)]}: {result.probs.top5conf[j]:.2f}"
            cv2.putText(annotated_image, text, (20, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated_image, text, (20, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            pos_y += 40
        return annotated_image