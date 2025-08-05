# producer_main.py

import time

# --- 데이터 모델 및 리포지토리 임포트 ---
from cctv_event_detector.core.models import Camera
# [수정] OnvifRepository 대신 ImageRepository 임포트
from cctv_event_detector.repository.image_repository import ImageRepository
from cctv_event_detector.repository.state_repository import StateRepository
from cctv_event_detector.inference.facade import AIInferenceFacade
# [수정] 새로 추가한 REDIS_IMAGE_CHANNEL 임포트
from config import CAMERA_INFO, REDIS_IMAGE_CHANNEL


def main():
    """
    저장된 이미지 파일을 순차적으로 불러와 AI 추론을 수행하고,
    결과를 Redis에 저장하는 메인 함수입니다.
    """
    print("===== CCTV Event Detector Producer (Image File Mode) 시작 =====")

    # 1. 설정 및 객체 초기화
    cameras = [Camera(**data) for data in CAMERA_INFO]
    # [수정] OnvifRepository 대신 ImageRepository 인스턴스 생성
    image_repo = ImageRepository()
    ai_facade = AIInferenceFacade()
    
    try:
        state_repo = StateRepository()
    except Exception as e:
        print(f"프로그램을 시작할 수 없습니다. StateRepository 초기화 실패: {e}")
        return

    print(f"성공적으로 {len(cameras)}개의 카메라와 AI 파이프라인을 초기화했습니다.")

    # 2. 메인 루프 (이미지 시퀀스를 순차적으로 처리)
    total_frames = image_repo.sync_frame_count
    if total_frames == 0:
        print("처리할 동기화된 이미지가 없습니다. 프로그램을 종료합니다.")
        return
        
    print(f"\n총 {total_frames}개의 동기화된 프레임에 대한 처리를 시작합니다.")
    
    try:
        # [수정] while 루프 대신 for 루프로 변경하여 정해진 프레임만큼만 실행
        for frame_index in range(total_frames):
            # frame_index = frame_index + 336 # [수정] 프레임 인덱스 시작을 조정
            print(f"\n--- 프레임 {frame_index + 1}/{total_frames} 처리 시작 ---")
            
            # 2.1. 현재 프레임 인덱스에 해당하는 모든 카메라의 이미지 일괄 로드
            captured_images = image_repo.get_images_for_frame_index(frame_index)
            
            if not captured_images:
                print("이미지를 더 이상 가져올 수 없습니다. 처리를 중단합니다.")
                break

            print(f"로드 완료: {len(captured_images)}개의 이미지를 메모리에 로드했습니다.")
            
            # 2.2. 로드된 이미지 배치를 AI 추론 파이프라인에 전달
            inference_results = ai_facade.process_batch(captured_images)
            print("AI 추론 완료.")

            # 2.3. 결과 데이터와 메타데이터를 Redis에 저장 (수정된 채널 이름 전달)
            state_repo.save_batch_results(captured_images, inference_results, REDIS_IMAGE_CHANNEL)
            
            print("----------------------------------------")

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n메인 루프에서 예상치 못한 오류 발생: {e}")
    finally:
        print("\n===== CCTV Event Detector Producer 종료 =====")

if __name__ == "__main__":
    main()