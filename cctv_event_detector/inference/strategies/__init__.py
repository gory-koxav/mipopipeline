# cctv_event_detector/inference/strategies/__init__.py

# 1. 각 파일에 정의된 클래스들을 현재 패키지 레벨로 가져옵니다.
from .base import InferenceStrategy
from .object_detector import MockObjectDetector
from .assembly_classifier import MockAssemblyClassifier
from .pinjig_segmentor import SamPinjigSegmenter

# 2. from .strategies import * 구문을 사용할 때,
#    어떤 클래스들을 외부에 공개할지 명시적으로 정의합니다. (권장 사항)
__all__ = [
    "InferenceStrategy",
    "MockObjectDetector",
    "MockAssemblyClassifier",
    "SamPinjigSegmenter",
]