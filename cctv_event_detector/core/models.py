# cctv_event_detector/core/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional, Tuple
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

@dataclass
class FrameData:
    """Redis에서 수집 및 파싱한 단일 카메라 프레임의 모든 정보를 담는 DTO"""
    image_id: str
    camera_name: str
    image_path: str
    captured_at: str
    image_shape: Tuple[int, int]
    detections: List[Dict[str, Any]] = field(default_factory=list)
    boundary_masks: List[np.ndarray] = field(default_factory=list) # 키가 아닌 실제 마스크 배열
    pinjig_masks: List[np.ndarray] = field(default_factory=list)  # pinjig 마스크 추가
    pinjig_classifications: List[Dict[str, Any]] = field(default_factory=list)  # pinjig 분류 정보 추가
    assembly_classifications: List[Dict[str, Any]] = field(default_factory=list)  # assembly 분류 정보 추가

@dataclass
class ProjectedData:
    """단일 카메라의 데이터를 절대 좌표계에 투영한 결과물"""
    camera_name: str
    warped_image: Optional[np.ndarray]
    warped_masks: List[np.ndarray]
    warped_pinjig_masks: List[np.ndarray]  # warped pinjig 마스크 추가
    projected_boxes: List[np.ndarray]
    projected_assembly_boxes: List[np.ndarray]  # assembly classification의 투영된 박스
    projected_assembly_labels: List[Dict[str, Any]]  # 투영된 assembly 라벨 정보 추가
    extent: List[float]  # Matplotlib.imshow의 extent [left, right, bottom, top]
    clip_polygon: np.ndarray # 이미지 및 마스크 클리핑 경로
    is_valid: bool = True
    merged_boxes: List[np.ndarray] = field(default_factory=list)  # 병합된 박스들 추가