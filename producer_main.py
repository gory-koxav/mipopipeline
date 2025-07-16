import cv2
import time
import numpy as np

# --- 데이터 모델 및 리포지토리 임포트 ---
from cctv_event_detector.core.models import Camera
from cctv_event_detector.repository.onvif_repository import OnvifRepository
# [수정] StateRepository 임포트 추가
from cctv_event_detector.repository.state_repository import StateRepository
from cctv_event_detector.inference.facade import AIInferenceFacade
from config import CAMERA_INFO


# 로컬 환경에서 디버깅 시 이미지를 확인하기 위한 함수 (선택 사항)
def show_captured_image(image):
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
    cameras = [Camera(**data) for data in CAMERA_INFO]
    onvif_repo = OnvifRepository()
    ai_facade = AIInferenceFacade()
    
    # [수정] StateRepository 인스턴스 생성
    try:
        state_repo = StateRepository()
    except Exception as e:
        print(f"프로그램을 시작할 수 없습니다. StateRepository 초기화 실패: {e}")
        return

    print(f"성공적으로 {len(cameras)}개의 카메라와 AI 파이프라인을 초기화했습니다.")

    # 2. 메인 루프 (주기적으로 이미지를 캡처하고 처리)
    try:
        while True:
            print("\n--- 새로운 캡처 및 추론 사이클 시작 ---")
            
            # 2.1. 모든 카메라에서 이미지 일괄 캡처
            captured_images = onvif_repo.capture_images_from_all(cameras)
            print(f"캡처 완료: {len(captured_images)}개의 이미지를 메모리에 로드했습니다.")
            
            if not captured_images:
                print("캡처된 이미지가 없습니다. 다음 사이클까지 대기합니다.")
                time.sleep(10)
                continue

            # (디버깅용) 캡처된 이미지를 파일로 저장합니다.
            for capture in captured_images:
                cv2.imwrite(f"./data/{capture.image_id}.jpg", capture.image_data)
            
            # 2.2. 캡처된 이미지 배치를 AI 추론 파이프라인에 전달
            inference_results = ai_facade.process_batch(captured_images)
            print("AI 추론 완료.")

            # 2.3. [최종 수정] 결과 데이터와 메타데이터를 Redis에 저장
            # 이전의 복잡한 출력 로직 대신, StateRepository에 모든 것을 위임합니다.
            state_repo.save_batch_results(captured_images, inference_results)
            
            print("----------------------------------------")
            # 다음 사이클까지 대기
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n메인 루프에서 예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    main()