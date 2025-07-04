# cctv_event_detector/repository/image_repository.py

import os
from datetime import datetime
from pathlib import Path
from typing import List

class ImageRepository:
    def __init__(self, base_dir: Path, camera_ids: List[str]):
        self._base_dir = base_dir
        self._camera_ids = camera_ids

    def get_latest_images(self, count: int) -> List[Path]:
        """지정된 모든 카메라 폴더에서 최신 이미지 파일 경로를 N개 가져옵니다."""
        all_image_paths = []
        for cam_id in self._camera_ids:
            cam_path = self._base_dir / cam_id
            if not cam_path.exists():
                continue
            
            for file_name in os.listdir(cam_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(cam_path / file_name)

        # 파일명에서 시간 정보를 파싱하여 최신순으로 정렬
        # 실제 환경에서는 에러 처리(시간 파싱 실패 등)가 중요합니다.
        try:
            all_image_paths.sort(key=lambda p: datetime.strptime(p.stem.split('_')[-1], "%Y%m%d%H%M%S"), reverse=True)
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse timestamp from some filenames. {e}")
            # 정렬 없이 그냥 반환하거나 다른 로직을 수행할 수 있음
            
        print(f"Found {len(all_image_paths)} images. Returning latest {count}.")
        return all_image_paths[:count]