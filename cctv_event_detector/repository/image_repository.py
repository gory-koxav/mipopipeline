# cctv_event_detector/repository/image_repository.py

import os
import cv2
import datetime
from typing import List, Dict
from pathlib import Path

from cctv_event_detector.core.models import Camera, CapturedImage
from config import IMAGE_DATA_DIR, CAMERA_INFO

class ImageRepository:
    """
    로컬 파일 시스템에 저장된 이미지 시퀀스를 관리하고 불러오는 역할을 합니다.
    각 카메라별 디렉토리에서 프레임 번호에 맞춰 이미지를 동기화하여 제공합니다.
    """
    def __init__(self):
        """
        ImageRepository를 초기화하고, 이미지 파일들을 스캔하여 메모리에 로드할 준비를 합니다.
        """
        self.image_data_dir = Path(IMAGE_DATA_DIR)
        self.camera_names = [cam["name"] for cam in CAMERA_INFO]
        self.camera_image_files: Dict[str, List[Path]] = {}
        self.sync_frame_count = 0
        
        print("--- ImageRepository 초기화 시작 ---")
        self._scan_image_files()
        self._calculate_sync_frames()
        print(f"✅ ImageRepository 초기화 완료. 동기화 가능한 프레임: {self.sync_frame_count}개")

    def _scan_image_files(self):
        """
        설정된 카메라 이름에 해당하는 디렉토리에서 이미지 파일 목록을 스캔하고 정렬합니다.
        """
        for name in self.camera_names:
            camera_dir = self.image_data_dir / name
            if not camera_dir.is_dir():
                print(f"⚠️ 경고: '{name}' 카메라의 디렉토리({camera_dir})를 찾을 수 없습니다. 건너뜁니다.")
                self.camera_image_files[name] = []
                continue

            # 이미지 파일만 필터링하여 리스트에 추가하고, 파일 이름 순으로 정렬합니다.
            files = sorted([p for p in camera_dir.glob("*.png")])
            self.camera_image_files[name] = files
            print(f"📁 '{name}' 카메라에서 {len(files)}개의 이미지 파일을 찾았습니다.")

    def _calculate_sync_frames(self):
        """
        모든 카메라에 공통적으로 존재하는 이미지의 최소 개수(동기화 가능한 프레임 수)를 계산합니다.
        """
        if not self.camera_image_files:
            self.sync_frame_count = 0
            return
        
        # 각 카메라별 이미지 파일 개수를 리스트로 만듭니다.
        frame_counts = [len(files) for files in self.camera_image_files.values()]
        
        # 리스트가 비어있지 않다면, 가장 작은 값을 동기화 프레임 수로 설정합니다.
        if frame_counts:
            self.sync_frame_count = min(frame_counts)
        else:
            self.sync_frame_count = 0
            
    def get_images_for_frame_index(self, frame_index: int) -> List[CapturedImage]:
        """
        주어진 프레임 인덱스에 해당하는 모든 카메라의 이미지를 불러와 리스트로 반환합니다.

        :param frame_index: 불러올 이미지의 순서 (0부터 시작)
        :return: 해당 프레임의 CapturedImage 객체 리스트
        """
        if frame_index >= self.sync_frame_count:
            print(f"요청된 프레임 인덱스({frame_index})가 동기화 가능한 최대치({self.sync_frame_count})를 초과했습니다.")
            return []

        captured_images = []
        # 현재 시간을 기준으로 각 프레임에 대한 가상 촬영 시각을 생성합니다.
        base_timestamp = datetime.datetime.now()

        for camera_name, image_files in self.camera_image_files.items():
            if frame_index < len(image_files):
                image_path = image_files[frame_index]
                image_data = cv2.imread(str(image_path))

                if image_data is None:
                    print(f"⚠️ 오류: {image_path} 파일을 읽을 수 없습니다.")
                    continue
                
                # 파일 이름(확장자 제외)을 프레임 ID로 사용합니다.
                frame_id = image_path.stem 
                
                # CapturedImage 객체 생성
                captured_image = CapturedImage(
                    image_id=f"{camera_name}_{frame_id}",
                    camera_name=camera_name,
                    image_data=image_data,
                    # 프레임 인덱스만큼 초를 더하여 고유한 타임스탬프를 부여합니다.
                    captured_at=base_timestamp + datetime.timedelta(seconds=frame_index)
                )
                captured_images.append(captured_image)

        return captured_images