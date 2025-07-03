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
    print("===== CCTV Event Detector Producer 시작 =====")

    # 1. 설정 및 객체 초기화
    # - 이 객체들은 프로그램 실행 동안 단 한 번만 생성됩니다.
    # - AIInferenceFacade 내부에서 무거운 AI 모델들이 로드됩니다.
    cameras = [Camera(**data) for data in CAMERA_INFO]
    onvif_repo = OnvifRepository()
    ai_facade = AIInferenceFacade()
    # state_repo = StateRepository() # Redis 저장을 위한 리포지토리

    print(f"성공적으로 {len(cameras)}개의 카메라와 AI 파이프라인을 초기화했습니다.")

    # 2. 메인 루프 (주기적으로 이미지를 캡처하고 처리)
    try:
        while True:
            print("\n--- 새로운 캡처 및 추론 사이클 시작 ---")
            
            # 2.1. 모든 카메라에서 이미지 일괄 캡처
            # - 반환값: List[CapturedImage]
            captured_images = onvif_repo.capture_images_from_all(cameras)
            print(f"캡처 완료: {len(captured_images)}개의 이미지를 메모리에 로드했습니다.")
            for capture in captured_images:
                cv2.imwrite(f"./data/{capture.image_id}.jpg", capture.image_data)  # 이미지 파일로 저장 (디버깅용)
            # print(f"캡처된 이미지 객체: {[img for img in captured_images]}") # 디버깅용
            # show_captured_image(captured_images[0].image_data)  # 첫 번째 이미지 표시 (로컬 환경에서 실행 시)
            
            # 2.2. 캡처된 이미지 배치를 AI 추론 파이프라인에 전달 # 이미지 인식 파이프라인 순차 실행 -> 인식 결과를 "inference_results" dict 에 순차적으로 별도의 key 를 만들어 저장
            # - 입력: List[CapturedImage]
            # - 반환값: Dict[image_id, result_dict]
            inference_results = ai_facade.process_batch(captured_images)
            # 2.2.1 이미지 전처리
            # 2.2.2 객체 탐지 (블록 위치, 블록 경계, 용접)
            # 2.2.3 정반 탐지
            print("AI 추론 완료.")

            # 2.3. 결과 데이터와 원본 메타데이터 통합 및 출력 (또는 Redis 저장)
            print("\n--- 최종 결과 ---")
            for capture in captured_images:
                image_id = capture.image_id
                if image_id in inference_results:
                    print(f"  - 이미지 ID: {image_id}")
                    print(f"    - 촬영 시각: {capture.captured_at.isoformat()}")
                    print(f"    - 카메라: {capture.camera_name}")
                    print(f"    - 추론 결과: {inference_results[image_id]}")
                    
                    # 여기에 Redis에 저장하는 로직 추가
                    # 예: state_repo.save_scene_state(image_id, capture, inference_results[image_id])
                else:
                    print(f"  - 이미지 ID {image_id}에 대한 추론 결과가 없습니다.")
            
            print("----------------------------------------")
            # 다음 사이클까지 대기
            time.sleep(10) # 예: 10초마다 반복

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()