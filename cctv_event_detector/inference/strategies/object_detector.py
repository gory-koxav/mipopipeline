# cctv_event_detector/inference/strategies/object_detector.py
from typing import Dict, Any, List
from ultralytics import YOLO
from collections import Counter # 클래스 개수를 세기 위해 Counter를 임포트합니다.

# config 파일에서 객체 탐지 모델 경로를 불러옵니다.
from config import YOLO_OD_MODEL_PATH

from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage

class YOLOObjectDetector(InferenceStrategy):
    """
    YOLO 모델을 사용하여 이미지에서 객체를 탐지하는 실제 구현체입니다.
    """
    def __init__(self):
        """
        config 파일로부터 모델 경로를 로드하여 YOLO 모델을 초기화합니다.
        """
        self.model = YOLO(YOLO_OD_MODEL_PATH)
        print(f"✅ YOLO Object Detection model loaded from: {YOLO_OD_MODEL_PATH}")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력된 이미지들에 대해 YOLO 객체 탐지를 수행하고, 결과를 딕셔너리에 저장합니다.
        """
        print("Running YOLO Object Detector...")
        
        image_data_list = [img.image_data for img in captured_images]
        if not image_data_list:
            return inference_results

        yolo_results = self.model(image_data_list, verbose=False)

        for i, capture in enumerate(captured_images):
            image_id = capture.image_id
            if image_id not in inference_results:
                inference_results[image_id] = {}

            result = yolo_results[i]
            detected_objects = []

            for box in result.boxes:
                bbox_xywh = [round(coord) for coord in box.xywh[0].tolist()]
                detected_objects.append({
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox_xywh": bbox_xywh
                })
            
            inference_results[image_id]["detections"] = detected_objects
            
            # --- ✨ 강화된 디버깅 프린트문 ✨ ---
            total_objects = len(detected_objects)
            if total_objects > 0:
                # 1. 클래스별 개수 세기 (예: {'part_A': 2, 'tool_X': 1})
                class_counts = Counter(obj['class_name'] for obj in detected_objects)
                
                # 2. 출력할 문자열 생성 (예: "part_A (2), tool_X (1)")
                class_summary = ", ".join([f"{name} ({count})" for name, count in class_counts.items()])
                
                # 3. 최종 결과 출력
                print(f"🔍 Found {total_objects} objects in '{image_id}': {class_summary}")
            else:
                # 탐지된 객체가 없을 경우
                print(f"🗑️ Found 0 objects in '{image_id}'")
            
        return inference_results