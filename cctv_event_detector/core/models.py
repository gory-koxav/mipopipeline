# cctv_event_detector/core/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional

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
    name: str
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)

    def get_image(self) -> Optional[Any]:
        """
        카메라의 호스트 정보 등을 이용해 실제 이미지를 가져옵니다.
        이미지를 가져오지 못하면 None을 반환합니다.
        """
        if not self.host:
            print(f"'{self.name}' 카메라의 호스트 정보가 없습니다.")
            return None
        
        print(f"'{self.host}'에서 이미지를 가져오는 중...")
        # TODO: RTSP/HTTP 라이브러리를 사용한 실제 이미지 요청 로직 구현
        # 예시: image = some_cv_library.get(self.host)
        image = "가져온 이미지 데이터" # 임시 데이터
        return image