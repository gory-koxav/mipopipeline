# cctv_event_detector/inference/strategies/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from cctv_event_detector.core.models import CapturedImage # 추가

# AI 모델 추론 전략에 대한 인터페이스 정의
class InferenceStrategy(ABC):
    @abstractmethod
    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        캡처된 이미지 리스트를 받아 추론을 수행하고,
        결과 딕셔너리를 업데이트하여 반환합니다.
        """
        pass