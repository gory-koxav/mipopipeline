import cv2
import time
import numpy as np
from cctv_event_detector.core.models import Camera
from config import CAMERA_INFO
from typing import Dict, Any, Optional

# 데이터 모델과 리포지토리를 각각의 위치에서 import
from cctv_event_detector.core.models import Camera
from cctv_event_detector.repository.onvif_repository import OnvifRepository
from cctv_event_detector.inference.facade import AIInferenceFacade
# from cctv_event_detector.repository.state_repository import StateRepository # 최종 저장을 위해 필요

def show_captured_image(image):       
    # 이미지를 화면에 표시 (로컬 환경에서 실행 시)
    try:
        cv2.imshow(f"Capture Result", image)
        print("이미지 창을 닫으려면 아무 키나 누르세요.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("GUI를 사용할 수 없는 환경에서는 이미지를 화면에 표시할 수 없습니다.")

def main():
    print("===== test_segmentation_main 시작 =====")

    # 1. 설정 및 객체 초기화
    # - 이 객체들은 프로그램 실행 동안 단 한 번만 생성됩니다.
    # - AIInferenceFacade 내부에서 무거운 AI 모델들이 로드됩니다.
    cameras = [Camera(**data) for data in CAMERA_INFO]
    ai_facade = AIInferenceFacade()

    print(f"성공적으로 {len(cameras)}개의 카메라와 AI 파이프라인을 초기화했습니다.")