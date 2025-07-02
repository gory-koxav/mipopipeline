# cctv_event_detector/repository/onvif_repository.py

import cv2
import time
import datetime
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera
from zeep.exceptions import Fault

from cctv_event_detector.core.models import Camera, CapturedImage

class OnvifRepository:
    """ONVIF 프로토콜을 통해 CCTV 이미지를 가져오는 리포지토리"""

    def _validate_camera_credentials(self, camera: Camera):
        """카메라 연결에 필요한 필수 정보가 있는지 확인합니다."""
        required_fields = {
            "host": camera.host,
            "port": camera.port,
            "username": camera.username,
            "password": camera.password
        }
        missing_fields = [key for key, value in required_fields.items() if value is None]
        if missing_fields:
            raise ValueError(f"카메라 연결에 필수적인 정보가 누락되었습니다: {', '.join(missing_fields)}")

    def _capture_with_snapshot_retry(self, camera: Camera, media_service, profile_token, max_retries=3, timeout=5) -> Optional[np.ndarray]:
        # ... (이하 내부 메서드는 변경 없음)
        for attempt in range(max_retries):
            try:
                snapshot_info = media_service.GetSnapshotUri({'ProfileToken': profile_token})
                snapshot_uri = snapshot_info.Uri
                response = requests.get(snapshot_uri, auth=HTTPDigestAuth(camera.username, camera.password), timeout=timeout)
                
                if response.status_code == 200 and response.content:
                    print(f"INFO (Snapshot): [{camera.name}] 캡처 성공")
                    image_array = np.frombuffer(response.content, np.uint8)
                    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"WARN (Snapshot attempt {attempt + 1}): [{camera.name}] 예외 발생 - {e}")
            time.sleep(0.5)
        print(f"ERROR (Snapshot): [{camera.name}] 최종 실패")
        return None

    def _capture_with_rtsp_retry(self, camera: Camera, media_service, profile_token, timeout=5) -> Optional[np.ndarray]:
        # ... (이하 내부 메서드는 변경 없음)
        print(f"INFO: [{camera.name}] RTSP 방식으로 전환하여 재시도합니다.")
        cap = None
        start_time = time.time()
        try:
            uri_info = media_service.GetStreamUri({
                'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                'ProfileToken': profile_token
            })
            stream_uri = uri_info.Uri
            stream_uri_auth = f"rtsp://{camera.username}:{camera.password}@{stream_uri.split('//')[1]}"
            
            cap = cv2.VideoCapture(stream_uri_auth)
            if not cap.isOpened():
                print(f"ERROR (RTSP): [{camera.name}] 스트림을 열 수 없습니다.")
                return None

            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    print(f"INFO (RTSP): [{camera.name}] 유효 프레임 확보 성공")
                    return frame
                time.sleep(0.1)

        except Exception as e:
            print(f"ERROR (RTSP): [{camera.name}] 캡처 중 예외 발생 - {e}")
        finally:
            if cap:
                cap.release()
        print(f"ERROR (RTSP): [{camera.name}] 최종 실패")
        return None

    def _create_failure_image(self, camera: Camera) -> np.ndarray:
        # ... (이하 내부 메서드는 변경 없음)
        print(f"FATAL: [{camera.name}] 모든 캡처 방법 실패. 에러 이미지를 생성합니다.")
        img = np.zeros((480, 640, 3), np.uint8)
        text1 = "CAPTURE FAILED"
        text2 = f"CAM: {camera.name} ({camera.host})"
        text3 = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, text1, (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(img, text2, (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, text3, (170, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        return img

    def _capture_single_image(self, camera: Camera) -> CapturedImage:
        """
        단일 카메라 이미지를 캡처하여 CapturedImage 객체로 반환 (기존 capture_image 로직)
        """
        # ... (기존 capture_image 메서드의 로직 대부분을 이곳으로 이동) ...
        self._validate_camera_credentials(camera)
        
        print(f"--- 카메라 [{camera.name}] 캡처 시작 ---")
        final_image = None
        
        try:
            mycam = ONVIFCamera(camera.host, camera.port, camera.username, camera.password)
            media_service = mycam.create_media_service()
            profile_token = media_service.GetProfiles()[0].token

            final_image = self._capture_with_snapshot_retry(camera, media_service, profile_token)
            
            if final_image is None:
                final_image = self._capture_with_rtsp_retry(camera, media_service, profile_token)

        except Exception as e:
            print(f"FATAL: [{camera.name}] 처리 중 복구 불가능한 에러 발생 - {e}")
        
        if final_image is None:
            final_image = self._create_failure_image(camera)

        captured_at_dt = datetime.datetime.now()
        image_id = f"{camera.name}_{captured_at_dt.strftime('%Y%m%d%H%M%S')}"
        
        print(f"--- 카메라 [{camera.name}] 캡처 완료 ---")

        return CapturedImage(
            image_id=image_id,
            camera_name=camera.name,
            image_data=final_image,
            captured_at=captured_at_dt
        )

    def capture_images_from_all(self, cameras: List[Camera]) -> List[CapturedImage]:
        """
        주어진 모든 카메라에서 이미지를 병렬적으로 (혹은 순차적으로) 캡처합니다.
        (참고: 실제 병렬 처리를 위해서는 threading 이나 asyncio 같은 기술이 필요하지만, 여기서는 순차 처리로 간소화합니다.)
        
        Args:
            cameras (List[Camera]): 이미지를 캡처할 카메라 객체 리스트

        Returns:
            List[CapturedImage]: 캡처된 이미지 데이터 객체 리스트
        """
        captured_images = []
        for camera in cameras:
            # 각 카메라에 대해 내부 캡처 메서드 호출
            captured_image_obj = self._capture_single_image(camera)
            captured_images.append(captured_image_obj)
        return captured_images