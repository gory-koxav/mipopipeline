# cctv_event_detector/inference/strategies/assembly_classifier.py
from typing import Dict, Any, List

from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage # 추가

class MockAssemblyClassifier(InferenceStrategy):
    """조립 상태 분류 모델 (Strategy 구현체)"""
    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        print("Running Mock Assembly Classifier...")
        
        for capture in captured_images:
            image_id = capture.image_id
            if image_id in inference_results and "detections" in inference_results[image_id]:
                num_detections = len(inference_results[image_id]["detections"])
                assembly_status = "complex_assembly" if num_detections > 1 else "simple_assembly"
                inference_results[image_id]["assembly_status"] = assembly_status
        return inference_results