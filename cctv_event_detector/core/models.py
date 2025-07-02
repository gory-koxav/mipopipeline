# cctv_event_detector/core/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional
import numpy as np
import datetime

@dataclass
class SceneState:
    """특정 시점의 현장 상태를 나타내는 데이터 클래스"""
    timestamp: str
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    assembly_status: str = "unknown"

@dataclass
class Event:
    """탐지된 이벤트를 나타내는 데이터 클래스"""
    event_type: Literal["object_appeared", "object_disappeared"]
    description: str
    timestamp: str

@dataclass
class Camera:
    """CCTV 카메라의 데이터 구조와 기본 동작을 정의하는 데이터 클래스"""
    bay: str
    name: str
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)

@dataclass
class CapturedImage:
    """
    카메라에서 캡처된 이미지와 관련 메타데이터를 담는 데이터 클래스 (DTO)
    """
    image_id: str          # 각 이미지를 식별할 고유 ID (예: "cctv_cam_01_20250702161000")
    camera_name: str       # 이미지를  촬영한 카메라 이름
    image_data: np.ndarray # 실제 이미지 데이터 (NumPy 배열)
    captured_at: datetime.datetime # 촬영 시각 (문자열 대신 datetime 객체 사용 권장)
    
    # NumPy 배열은 기본적으로 비교가 복잡하므로, eq=False로 설정
    def __post_init__(self):
        self.__repr__ = lambda: f"CapturedImage(image_id={self.image_id})"