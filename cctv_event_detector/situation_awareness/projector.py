# cctv_event_detector/situation_awareness/projector.py

import numpy as np
import cv2
from typing import Dict, Any, Optional

from cctv_event_detector.core.models import FrameData, ProjectedData
from config import CAMERA_INTRINSICS, CAMERA_EXTRINSICS

class Projector:
    """
    단일 카메라의 프레임 데이터를 받아 절대 좌표계로 투영(사영 변환)하는
    모든 계산을 담당합니다.
    """
    def __init__(self):
        self.fx = CAMERA_INTRINSICS['fx']
        self.fy = CAMERA_INTRINSICS['fy']
        self.cx = CAMERA_INTRINSICS['cx']
        self.cy = CAMERA_INTRINSICS['cy']

    def _rotation_matrix(self, pan_deg: float, tilt_deg: float) -> np.ndarray:
        """Pan(Yaw), Tilt(Pitch) 각도를 사용하여 회전 행렬을 생성합니다."""
        pan = np.deg2rad(pan_deg)
        tilt = np.deg2rad(tilt_deg)
        R_tilt = np.array([
            [1, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt),  np.cos(tilt)]
        ])
        R_pan = np.array([
            [np.cos(pan), -np.sin(pan), 0],
            [np.sin(pan),  np.cos(pan), 0],
            [0, 0, 1]
        ])
        return R_pan @ R_tilt

    def _project_pixel_to_ground(self, u: float, v: float, R: np.ndarray, t: np.ndarray) -> Optional[np.ndarray]:
        """픽셀 좌표 (u, v)를 지면(z=0) 좌표로 투영합니다."""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        ray_cam = np.array([x, y, -1.0])
        ray_world = R @ ray_cam
        
        if np.abs(ray_world[2]) < 1e-6: return None
        lam = -t[2] / ray_world[2]
        if lam < 0: return None
        
        intersect = t + lam * ray_world
        return intersect[:2]

    def project(self, frame_data: FrameData) -> ProjectedData:
        """
        하나의 FrameData를 받아 이미지, 마스크, 객체 박스를 투영하고
        ProjectedData 객체로 반환합니다.
        """
        cam_config = CAMERA_EXTRINSICS.get(frame_data.camera_name)
        if not cam_config:
            print(f"경고: {frame_data.camera_name}에 대한 카메라 설정을 찾을 수 없습니다. 건너뜁니다.")
            return ProjectedData(camera_name=frame_data.camera_name, is_valid=False, warped_image=None, warped_masks=[], projected_boxes=[], extent=[], clip_polygon=np.array([]))

        R = self._rotation_matrix(cam_config['pan'], cam_config['tilt'])
        t = cam_config['coord']
        h_img, w_img = frame_data.image_shape

        # 이미지의 네 꼭짓점 픽셀 좌표
        src_corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32)
        
        # 꼭짓점을 지면으로 투영
        projected_corners = [self._project_pixel_to_ground(u, v, R, t) for u, v in src_corners]

        if any(corner is None for corner in projected_corners):
            return ProjectedData(camera_name=frame_data.camera_name, is_valid=False, warped_image=None, warped_masks=[], projected_boxes=[], extent=[], clip_polygon=np.array([]))

        dst_corners = np.array(projected_corners, dtype=np.float32)

        # Homography 행렬 계산
        H_mat, _ = cv2.findHomography(src_corners, dst_corners, method=0)
        
        # 원본 이미지 로드 및 뒤집기 (기존 코드 로직 유지)
        img = cv2.imread(frame_data.image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"경고: 이미지 로드 실패 - {frame_data.image_path}")
            img = np.zeros((*frame_data.image_shape, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)

        # 투영될 이미지의 범위(extent) 계산
        min_x, max_x = np.min(dst_corners[:, 0]), np.max(dst_corners[:, 0])
        min_y, max_y = np.min(dst_corners[:, 1]), np.max(dst_corners[:, 1])
        extent = [min_x, max_x, max_y, min_y]

        # 이미지 워핑
        # OpenCV의 warpPerspective는 출력 이미지 크기를 픽셀 단위로 받으므로,
        # 투영된 좌표계의 크기를 적절한 해상도로 변환해야 합니다.
        # 여기서는 시각화의 편의를 위해 직접 변환 행렬을 사용하지 않고,
        # matplotlib의 기능(imshow의 extent)을 활용합니다.
        
        # 객체 경계 상자 투영
        projected_boxes = []
        for det in frame_data.detections:
            x_min, y_min, w_box, h_box = det['bbox_xywh']
            x_max, y_max = x_min + w_box, y_min + h_box
            
            # BBox의 네 꼭짓점을 원본 이미지 좌표계 기준으로 정의
            # (좌상단, 우상단, 우하단, 좌하단)
            box_corners_orig = np.array([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ], dtype=np.float32)

            # 기존 코드처럼 이미지 flip에 맞춰 좌표 변환
            box_corners_flipped = box_corners_orig.copy()
            box_corners_flipped[:, 0] = w_img - 1 - box_corners_flipped[:, 0]
            box_corners_flipped[:, 1] = h_img - 1 - box_corners_flipped[:, 1]
            
            # 변환된 BBox 꼭짓점을 절대 좌표계로 투영
            projected_box_vertices = [self._project_pixel_to_ground(u, v, R, t) for u, v in box_corners_flipped]

            if not any(v is None for v in projected_box_vertices):
                projected_boxes.append(np.array(projected_box_vertices))

        # 마스크 데이터 뒤집기
        warped_masks = []
        for mask in frame_data.boundary_masks:
            flipped_mask = cv2.flip(mask, 0)
            flipped_mask = cv2.flip(flipped_mask, 1)
            warped_masks.append(flipped_mask)

        return ProjectedData(
            camera_name=frame_data.camera_name,
            warped_image=img,
            warped_masks=warped_masks,
            projected_boxes=projected_boxes,
            extent=extent,
            clip_polygon=dst_corners,
            is_valid=True
        )