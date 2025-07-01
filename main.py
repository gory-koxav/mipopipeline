from cctv_event_detector.core.models import Camera
from config import CAMERA_INFO

def main():
    # 설정 데이터를 기반으로 Camera 객체 리스트 생성
    cameras = [Camera(**data) for data in CAMERA_INFO]

    # 프로그램 로직 실행
    print("성공적으로 카메라 객체를 생성했습니다:")
    for cam in cameras:
        print(f"- {cam}")

if __name__ == "__main__":
    main()