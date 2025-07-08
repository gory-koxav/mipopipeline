# cctv_event_detector/inference/strategies/object_boundary_segmenter.py
import torch
import numpy as np
import cv2
from typing import Dict, Any, List

# --- 설정 및 기본 클래스 임포트 ---
from config import (
    SAM_CHECKPOINT_PATH,
    SAM_MODEL_TYPE,
    BOUNDARY_TARGET_CLASSES,
    SAM_BOUNDARY_OUTPUT_DIR,
)
from .base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage

# --- SAM 관련 클래스 임포트 ---
from segment_anything import sam_model_registry, SamPredictor

class SAMObjectBoundarySegmenter(InferenceStrategy):
    """
    YOLO로 탐지된 객체의 BBox를 프롬프트로 사용하여 SAM으로 정확한 경계선을 찾는 전략 클래스입니다.
    """
    def __init__(self):
        """
        SAM 모델과 Predictor를 초기화하고, config에서 설정을 불러옵니다.
        """
        print("Initializing SAM Object Boundary Segmenter...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SAM 모델 로드
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH))
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print(f"✅ SAM Predictor ('{SAM_MODEL_TYPE}') loaded successfully.")

        # ✨ 요구사항 2: 타겟 클래스 설정이 없거나 비어있으면 빈 set으로 처리
        self.target_classes = set(BOUNDARY_TARGET_CLASSES) if BOUNDARY_TARGET_CLASSES else set()
        
        if self.target_classes:
            print(f"🎯 Target classes for boundary finding: {self.target_classes}")
        else:
            print("🎯 Target classes not specified, processing all classes.")

        self.output_dir = SAM_BOUNDARY_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Boundary visualization output dir: {self.output_dir}")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running SAM Object Boundary Segmenter...")

        for capture in captured_images:
            image_id = capture.image_id
            original_image = capture.image_data

            detections = inference_results.get(image_id, {}).get("detections", [])
            if not detections:
                continue

            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(rgb_image)

            boundary_masks = []
            vis_image = original_image.copy()
            
            # ✨ 요구사항 1: 시각화 강화를 위해 모든 YOLO 탐지 결과를 먼저 그림
            for detection in detections:
                self._draw_yolo_bbox_and_label(vis_image, detection)

            for detection in detections:
                # ✨ 요구사항 2: 타겟 클래스가 비어있거나, 클래스가 타겟에 포함되면 처리
                should_process = not self.target_classes or detection['class_name'] in self.target_classes
                
                if should_process:
                    x, y, w, h = detection['bbox_xywh']
                    input_box = np.array([x - w/2, y - h/2, x + w/2, y + h/2])

                    masks, scores, _ = self.predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )
                    mask = masks[0]
                    
                    boundary_masks.append({
                        "class_name": detection['class_name'],
                        "confidence": scores[0],
                        "segmentation_mask": mask
                    })
                    # SAM 마스크는 처리된 객체에 대해서만 그림
                    self._draw_sam_mask_on_image(vis_image, mask)

            if boundary_masks:
                # output_path = self.output_dir / image_id / f"{image_id}_boundaries.jpg"
                output_dir_path = self.output_dir / image_id
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_path = output_dir_path / f"{image_id}_boundaries.jpg"
                cv2.imwrite(str(output_path), vis_image)
                print(f"🖼️ Saved boundary visualization to {output_path}")

                if "boundary_masks" not in inference_results[image_id]:
                    inference_results[image_id]['boundary_masks'] = []
                inference_results[image_id]['boundary_masks'].extend(boundary_masks)

        return inference_results

    def _draw_yolo_bbox_and_label(self, image: np.ndarray, detection: Dict[str, Any]):
        """
        이미지에 YOLO 바운딩 박스와 클래스 레이블을 그립니다.
        """
        x, y, w, h = detection['bbox_xywh']
        class_name = detection['class_name']
        
        # BBox 좌표 계산
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        # BBox 그리기 (초록색)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 클래스명 및 신뢰도 텍스트 그리기
        label = f"{class_name}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # 텍스트 배경 및 텍스트 그리기
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def _draw_sam_mask_on_image(self, image: np.ndarray, mask: np.ndarray):
        """
        주어진 이미지에 SAM 마스크를 랜덤 색상으로 오버레이합니다.
        """
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        overlay = image.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)