# cctv_event_detector/inference/strategies/object_detector.py
import cv2 # 이미지 처리를 위해 OpenCV를 임포트합니다.
import numpy as np # OpenCV와 함께 사용될 NumPy를 임포트합니다.
from typing import Dict, Any, List
from ultralytics import YOLO
from collections import Counter

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

    # ✨ --- START: YOLO 추론 전용 전처리 메서드 추가 --- ✨
    def _preprocess_for_yolo(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        입력된 이미지 리스트를 YOLO 추론에 맞게 전처리합니다.
        가장 중요한 단계는 OpenCV의 BGR 채널을 모델이 기대하는 RGB 채널로 변환하는 것입니다.
        """
        processed_images = []
        for img in images:
            # CapturedImage.image_data는 OpenCV로 생성된 BGR 이미지입니다.
            # YOLO 모델은 RGB 입력을 기대하므로, 컬러 채널 순서를 변환합니다.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append(img_rgb)
        
        # 전처리가 완료되었음을 로그로 남깁니다.
        if processed_images:
            print(f"INFO: Preprocessed {len(processed_images)} images for YOLO (BGR -> RGB).")
            
        return processed_images
    # ✨ --- END: YOLO 추론 전용 전처리 메서드 추가 --- ✨

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력된 이미지들에 대해 YOLO 객체 탐지를 수행하고, 결과를 딕셔너리에 저장합니다.
        """
        print("Running YOLO Object Detector...")
        
        # 1. 추론할 원본 이미지 데이터 리스트를 가져옵니다.
        image_data_list = [img.image_data for img in captured_images]
        if not image_data_list:
            return inference_results

        # ✨ --- START: 핵심 수정 부분 --- ✨
        # 2. YOLO 모델에 입력하기 전에 전용 전처리 메서드를 호출합니다.
        # processed_image_list = self._preprocess_for_yolo(image_data_list)
        processed_image_list = image_data_list # cvt 전처리 없이 바로 사용합니다.
        
        # 3. 전처리된 이미지로 추론을 수행합니다.
        yolo_results = self.model(processed_image_list, verbose=False)
        # ✨ --- END: 핵심 수정 부분 --- ✨

        # 4. 추론 결과를 정리합니다. (이후 로직은 동일)
        for i, capture in enumerate(captured_images):
            image_id = capture.image_id
            if image_id not in inference_results:
                inference_results[image_id] = {}

            result = yolo_results[i]
            detected_objects = []

            for box in result.boxes:
                # 🔴 중요: YOLO는 중심점 기준 좌표를 반환하므로 좌상단 기준으로 변환
                center_x, center_y, width, height = box.xywh[0].tolist()
                
                # 좌상단 기준 좌표로 변환
                x_min = center_x - width / 2
                y_min = center_y - height / 2
                
                # 반올림하여 정수로 변환
                bbox_xywh = [round(x_min), round(y_min), round(width), round(height)]
                
                detected_objects.append({
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox_xywh": bbox_xywh  # 이제 [x_min, y_min, width, height] 형식
                })
            
            inference_results[image_id]["detections"] = detected_objects
            
            total_objects = len(detected_objects)
            if total_objects > 0:
                class_counts = Counter(obj['class_name'] for obj in detected_objects)
                class_summary = ", ".join([f"{name} ({count})" for name, count in class_counts.items()])
                print(f"🔍 Found {total_objects} objects in '{image_id}': {class_summary}")
            else:
                print(f"🗑️ Found 0 objects in '{image_id}'")
            
        return inference_results