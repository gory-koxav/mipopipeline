# cctv_event_detector/core/services.py
from datetime import datetime
from typing import List, Optional

from ..inference.facade import AIInferenceFacade
from .models import SceneState, Event

# --- Producer를 위한 서비스 ---
class StateCreationService:
    """AI 추론 결과를 바탕으로 표준화된 '상태' 객체를 생성하는 책임"""
    def __init__(self, ai_facade: AIInferenceFacade):
        self._ai_facade = ai_facade

    def create_state_from_images(self, image_paths: List[str]) -> SceneState:
        # AI 추론 실행
        ai_results = self._ai_facade.process(image_paths)
        
        # 결과를 SceneState 객체로 변환 (비즈니스 로직 후처리)
        print("Creating SceneState from AI results.")
        return SceneState(
            timestamp=datetime.now().isoformat(),
            detected_objects=ai_results.get("detections", []),
            assembly_status=ai_results.get("assembly_status", "unknown")
        )

# --- Consumer를 위한 서비스 ---
class EventDetectionService:
    """두 상태를 비교하여 '이벤트'를 찾아내는 책임"""
    def detect_events(self, prev_state: Optional[SceneState], curr_state: SceneState) -> List[Event]:
        if not prev_state:
            print("No previous state found. Cannot detect events yet.")
            return []

        print(f"Comparing states: {prev_state.timestamp} vs {curr_state.timestamp}")
        events = []
        now = datetime.now().isoformat()
        
        prev_object_ids = {obj['object_id'] for obj in prev_state.detected_objects}
        curr_object_ids = {obj['object_id'] for obj in curr_state.detected_objects}

        # 사라진 객체 탐지
        disappeared_ids = prev_object_ids - curr_object_ids
        for obj_id in disappeared_ids:
            events.append(Event("object_disappeared", f"Object ID '{obj_id}' disappeared.", now))

        # 새로 생긴 객체 탐지
        appeared_ids = curr_object_ids - prev_object_ids
        for obj_id in appeared_ids:
            events.append(Event("object_appeared", f"New object ID '{obj_id}' appeared.", now))
            
        print(f"Event Detection: {len(events)} events found.")
        return events