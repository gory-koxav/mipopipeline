import cv2
import time
import requests
import numpy as np
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera
from zeep.transports import Transport
from zeep.exceptions import Fault

# --- 헬퍼 함수 정의 ---

def pan_relative(ptz, profile_token, step, sleep_time):
    """지정된 값만큼 카메라를 상대적으로 회전시킵니다."""
    try:
        request = {
            'ProfileToken': profile_token,
            'Translation': {'PanTilt': {'x': step, 'y': 0.0, 'space': 'http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace'}}
        }
        ptz.RelativeMove(request)
        time.sleep(sleep_time)
    except Fault as e:
        print(f"WARN: PTZ 이동 실패: {e}")

# --- 3단계 전략 함수 ---

def capture_with_snapshot_retry(media_service, profile_token, user, passwd, max_retries=3, timeout=5):
    """1단계: 스냅샷으로 이미지 캡처를 재시도."""
    for attempt in range(max_retries):
        try:
            snapshot_info = media_service.GetSnapshotUri({'ProfileToken': profile_token})
            snapshot_uri = snapshot_info.Uri
            response = requests.get(snapshot_uri, auth=HTTPDigestAuth(user, passwd), timeout=timeout)
            
            if response.status_code == 200 and response.content:
                print("INFO (Snapshot): 캡처 성공")
                # 이미지 데이터를 numpy 배열로 디코딩하여 반환
                image_array = np.frombuffer(response.content, np.uint8)
                return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"WARN (Snapshot attempt {attempt+1}): 예외 발생 - {e}")
        time.sleep(0.5)
    print("ERROR (Snapshot): 최종 실패")
    return None

def capture_with_rtsp_retry(media_service, profile_token, user, passwd, timeout=5):
    """2단계: 안정화된 RTSP 스트림으로 이미지 캡처를 재시도."""
    print("INFO: RTSP 방식으로 전환하여 재시도합니다.")
    cap = None
    start_time = time.time()
    try:
        uri_info = media_service.GetStreamUri({
            'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
            'ProfileToken': profile_token
        })
        stream_uri = uri_info.Uri
        stream_uri_auth = f"rtsp://{user}:{passwd}@{stream_uri.split('//')[1]}"
        
        cap = cv2.VideoCapture(stream_uri_auth)
        if not cap.isOpened():
            print("ERROR (RTSP): 스트림을 열 수 없습니다.")
            return None

        # 타임아웃 시간 내에 유효한 프레임이 잡힐 때까지 계속 시도
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                print("INFO (RTSP): 유효 프레임 확보 성공")
                return frame
            time.sleep(0.1) # 너무 빠른 루프 방지

    except Exception as e:
        print(f"ERROR (RTSP): 캡처 중 예외 발생 - {e}")
    finally:
        if cap:
            cap.release()
    print("ERROR (RTSP): 최종 실패")
    return None

def create_failure_image(file_path, cam_info):
    """3단계: 모든 캡처 실패 시, 실패 증거 이미지를 생성."""
    print(f"FATAL: 모든 캡처 방법 실패. {file_path}에 에러 이미지를 생성합니다.")
    # 검은색 배경 생성 (가로 640, 세로 480)
    img = np.zeros((480, 640, 3), np.uint8)
    
    # 텍스트 설정
    text1 = "CAPTURE FAILED"
    text2 = f"CAM: {cam_info}"
    text3 = time.strftime("%Y-%m-%d %H:%M:%S")

    # 텍스트를 이미지에 추가
    cv2.putText(img, text1, (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(img, text2, (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, text3, (170, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    # 이미지 저장
    cv2.imwrite(file_path, img)

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    CAMERA_INFO = [
        '10.150.160.183,80,admin,Hmd!!2503700',
        '10.150.160.184,80,admin,Hmd!!2503700',
        '10.150.160.193,80,admin,Hmd!!2503700',
        '10.150.160.194,80,admin,Hmd!!2503700',
        '10.150.160.209,80,admin,Hmd!!2503700',
        '10.150.160.211,80,admin,Hmd!!2503700',
        '10.150.160.221,80,admin,Hmd!!2503700',
        '10.150.160.222,80,admin,Hmd!!2503700',
        '10.150.160.224,80,admin,Hmd!!2503700',
        '10.150.160.225,80,admin,Hmd!!2503700',
        '10.150.160.226,80,admin,Hmd!!2503700',
        '10.150.160.227,80,admin,Hmd!!2503700'
    ]

    for i, camera_str in enumerate(CAMERA_INFO):
        ptz_service = None
        profile_token = None
        initial_position = None
        host, port, username, password = camera_str.split(',')

        try:
            print(f"--- 카메라 {i+1}/{len(CAMERA_INFO)} ({host}) 캡처 시작 ---")
            
            mycam = ONVIFCamera(host, int(port), username, password)
            media_service = mycam.create_media_service()
            ptz_service = mycam.create_ptz_service()
            profiles = media_service.GetProfiles()
            media_profile = profiles[0]
            profile_token = media_profile.token
            ptz_status = ptz_service.GetStatus({'ProfileToken': profile_token})
            initial_position = ptz_status.Position

            pan_step_num = 7
            pan_step = -0.015
            pan_relative(ptz_service, profile_token, pan_step * pan_step_num * -1, 1.0)
            
            j = 0
            for j in range(pan_step_num * 2):
                file_path = f'camera_{i:02d}_{j:02d}.jpeg'
                final_image = None
                
                # 1단계 시도
                final_image = capture_with_snapshot_retry(media_service, profile_token, username, password)
                
                # 2단계 시도 (1단계 실패 시)
                if final_image is None:
                    final_image = capture_with_rtsp_retry(media_service, profile_token, username, password)
                
                # 3단계 실행 (1, 2단계 모두 실패 시)
                if final_image is None:
                    create_failure_image(file_path, f"{i+1} ({host})")
                else:
                    cv2.imwrite(file_path, final_image)
                    print(f"SUCCESS: {file_path} 저장 완료")

                pan_relative(ptz_service, profile_token, pan_step, 1.0)

            # (마지막 프레임 캡처 로직도 위와 동일하게 적용)

        except Exception as e:
            print(f"FATAL: 카메라 {i+1} 처리 중 복구 불가능한 에러 발생 - {e}")
        finally:
            try:
                if ptz_service and profile_token and initial_position:
                    ptz_service.AbsoluteMove({'ProfileToken': profile_token, 'Position': initial_position})
                    print(f"INFO: 카메라 {i+1}를 초기 위치로 복귀시켰습니다.")
            except Exception as e:
                print(f"ERROR: 카메라 {i+1} 초기 위치 복귀 실패: {e}")
            print("-" * 50 + "\n")