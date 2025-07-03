# cctv_event_detector/inference/facade.py
from typing import List, Dict, Any

from cctv_event_detector.inference.strategies import InferenceStrategy, MockObjectDetector, MockAssemblyClassifier
from cctv_event_detector.core.models import CapturedImage # 추가


class AIInferenceFacade:
    def __init__(self):
        # AI 모델(전략)들은 한 번만 로드
        self._pipeline: List[InferenceStrategy] = [
            MockObjectDetector(),
            MockAssemblyClassifier(),
        ]

    def process_batch(self, captured_images: List[CapturedImage]) -> Dict[str, Any]:
        """퍼사드 메서드: CapturedImage 객체 리스트를 받아 전체 AI 추론 파이프라인을 실행"""
        print("--- AI Inference Facade: Starting pipeline ---")
        
        # 초기 결과 딕셔너리 (이제 image_id를 키로 사용)
        inference_results: Dict[str, Any] = {}
        
        # 파이프라인의 각 단계를 순차적으로 실행
        for strategy in self._pipeline:
            inference_results = strategy.run(captured_images, inference_results)
            
        print("--- AI Inference Facade: Pipeline finished ---")
        return inference_results