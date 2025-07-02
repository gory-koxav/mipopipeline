# cctv_event_detector/inference/strategies.py
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from cctv_event_detector.core.models import CapturedImage # 추가

class InferenceStrategy(ABC):
    @abstractmethod
    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        pass

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

class MockAssemblyClassifier(InferenceStrategy):
    """조립 상태 분류 모델 (Strategy 구현체)"""
    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Mock Assembly Classifier...")
        
        for capture in captured_images:
            image_id = capture.image_id
            # 앞선 detection 결과를 바탕으로 추가 분석
            if image_id in inference_results and "detections" in inference_results[image_id]:
                num_detections = len(inference_results[image_id]["detections"])
                assembly_status = "complex_assembly" if num_detections > 1 else "simple_assembly"
                inference_results[image_id]["assembly_status"] = assembly_status
        return inference_results

# --- 실제 AI 모델들을 시뮬레이션하는 Mock 클래스 ---
''' 예시
class MockObjectDetector(InferenceStrategy):
    """객체 탐지 모델 (Strategy 구현체)"""
    def run(self, images: List[str], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Mock Object Detector...")
        # 실제로는 여기서 모델 로드 및 추론 수행
        detected_objects = []
        for i, img_path in enumerate(images):
            # 시뮬레이션을 위해 랜덤 객체 생성
            num_objects = random.randint(1, 3)
            for j in range(num_objects):
                detected_objects.append({
                    "image_path": str(img_path),
                    "object_id": f"obj_{i}_{j}",
                    "class": random.choice(["part_A", "part_B", "tool_X"]),
                    "bbox": [random.randint(10, 400), random.randint(10, 400), 50, 50]
                })
        inference_results["detections"] = detected_objects
        return inference_results
'''