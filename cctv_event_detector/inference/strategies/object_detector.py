# cctv_event_detector/inference/strategies/object_detector.py
from typing import Dict, Any, List

from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage # 추가

import random # 개발후 삭제

class MockObjectDetector(InferenceStrategy):
    """객체 탐지 모델 (Strategy 구현체)"""
    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Mock Object Detector...")
        # 실제 모델은 여기서 이미지 배치([img.image_data for img in captured_images])를 받아 추론합니다.
        
        for capture in captured_images:
            # 결과를 해당 이미지의 image_id 아래에 저장
            if capture.image_id not in inference_results:
                inference_results[capture.image_id] = {}

            # 시뮬레이션을 위해 랜덤 객체 생성
            num_objects = random.randint(1, 3)
            detected_objects = []
            for j in range(num_objects):
                detected_objects.append({
                    "object_id": f"obj_{capture.camera_name}_{j}",
                    "class": random.choice(["part_A", "part_B", "tool_X"]),
                    "bbox": [random.randint(10, 400), random.randint(10, 400), 50, 50]
                })
            
            inference_results[capture.image_id]["detections"] = detected_objects
        return inference_results