# cctv_event_detector/situation_awareness/projector.py

import numpy as np
import cv2  # cv2로 import 되었으므로 cv2를 사용해야 합니다.
from typing import List, Dict, Any, Optional, Set
from cctv_event_detector.core.models import FrameData, ProjectedData
from config import CAMERA_INTRINSICS, CAMERA_EXTRINSICS

class Projector:
    """
    단일 카메라의 프레임 데이터를 받아 절대 좌표계로 투영(사영 변환)하는
    모든 계산을 담당합니다. (경계 상자 투영 로직 수정)
    """
    def __init__(self, pixels_per_meter: int = 10, target_classes: Optional[List[str]] = None):
        """
        Projector를 초기화합니다.

        :param pixels_per_meter: 월드 좌표계 1미터를 몇 픽셀로 표현할지 결정하는 해상도 값
        :param target_classes: 평면도에 투영할 객체의 class_name 목록. None이면 모든 객체를 투영합니다.
        """
        self.fx = CAMERA_INTRINSICS['fx']
        self.fy = CAMERA_INTRINSICS['fy']
        self.cx = CAMERA_INTRINSICS['cx']
        self.cy = CAMERA_INTRINSICS['cy']
        self.pixels_per_meter = pixels_per_meter
        # target_classes가 주어지면 set으로 만들어 조회 성능을 높입니다.
        self.target_classes: Optional[Set[str]] = set(target_classes) if target_classes else None

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
        하나의 FrameData를 받아 이미지, 마스크, 객체 박스를 평면도(Bird's-eye view)로 투영하고
        ProjectedData 객체로 반환합니다.
        """
        cam_config = CAMERA_EXTRINSICS.get(frame_data.camera_name)
        if not cam_config:
            print(f"경고: {frame_data.camera_name}에 대한 카메라 설정을 찾을 수 없습니다.")
            return ProjectedData(camera_name=frame_data.camera_name, is_valid=False)

        # 1. 원본 이미지 로드
        img = cv2.imread(frame_data.image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"경고: 이미지 로드 실패 - {frame_data.image_path}")
            return ProjectedData(camera_name=frame_data.camera_name, is_valid=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        R = self._rotation_matrix(cam_config['pan'], cam_config['tilt'])
        t = np.array(cam_config['coord'])

        # 2. 이미지의 네 꼭짓점을 월드 좌표계로 투영
        src_corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32)
        projected_corners = [self._project_pixel_to_ground(u, v, R, t) for u, v in src_corners]

        if any(corner is None for corner in projected_corners):
            print(f"경고: {frame_data.camera_name}의 일부 영역이 지평선 너머로 투영되어 변환할 수 없습니다.")
            return ProjectedData(camera_name=frame_data.camera_name, is_valid=False)

        dst_corners = np.array(projected_corners, dtype=np.float32)

        # 3. Homography 및 최종 Warp 행렬 계산
        H_mat, _ = cv2.findHomography(src_corners, dst_corners)
        if H_mat is None:
            print(f"경고: {frame_data.camera_name}의 Homography 행렬 계산에 실패했습니다.")
            return ProjectedData(camera_name=frame_data.camera_name, is_valid=False)

        # --- 핵심 수정 사항 ---
        # warped_image가 180도 회전되었으므로, extent(프레임)도 회전된 이미지 기준으로 계산해야 함
        # dst_corners = [proj(0,0), proj(w,0), proj(w,h), proj(0,h)]
        # Flipped(0,0) -> Orig(w,h) -> proj(w,h) -> dst_corners[2]
        # Flipped(w,0) -> Orig(0,h) -> proj(0,h) -> dst_corners[3]
        # Flipped(w,h) -> Orig(0,0) -> proj(0,0) -> dst_corners[0]
        # Flipped(0,h) -> Orig(w,0) -> proj(w,0) -> dst_corners[1]
        dst_corners_for_flipped = np.array([
            dst_corners[2], dst_corners[3], dst_corners[0], dst_corners[1]
        ])
        
        # 회전된 이미지에 맞는 새로운 extent(프레임) 계산
        min_x = np.min(dst_corners_for_flipped[:, 0])
        max_x = np.max(dst_corners_for_flipped[:, 0])
        min_y = np.min(dst_corners_for_flipped[:, 1])
        max_y = np.max(dst_corners_for_flipped[:, 1])
        
        warp_width = int(np.ceil((max_x - min_x) * self.pixels_per_meter))
        warp_height = int(np.ceil((max_y - min_y) * self.pixels_per_meter))

        if not (0 < warp_width < 8000 and 0 < warp_height < 8000):
            print(f"경고: {frame_data.camera_name}의 변환 결과 크기({warp_width}x{warp_height})가 너무 큽니다.")
            return ProjectedData(camera_name=frame_data.camera_name, is_valid=False)

        T_matrix = np.array([
            [self.pixels_per_meter, 0, -min_x * self.pixels_per_meter],
            [0, self.pixels_per_meter, -min_y * self.pixels_per_meter],
            [0, 0, 1]
        ], dtype=np.float64)
        
        H_warp = T_matrix @ H_mat

        # 4. 이미지와 마스크를 실제로 Warp(변환)
        img_flipped = cv2.flip(cv2.flip(img, 0), 1)
        warped_image = cv2.warpPerspective(img_flipped, H_warp, (warp_width, warp_height), flags=cv2.INTER_LINEAR)
        
        warped_masks = []
        for mask in frame_data.boundary_masks:
            mask_flipped = cv2.flip(cv2.flip(mask, 0), 1)
            warped_mask = cv2.warpPerspective(mask_flipped, H_warp, (warp_width, warp_height), flags=cv2.INTER_NEAREST)
            warped_masks.append(warped_mask)

        # 5. 객체 경계 상자 투영 (지난번 수정과 동일하게 유지)
        projected_boxes = []
        for det in frame_data.detections:
            class_name = det.get('class_name')
            if self.target_classes and class_name not in self.target_classes:
                continue
            
            x_min, y_min, w_box, h_box = det['bbox_xywh']
            x_max, y_max = x_min + w_box, y_min + h_box
            
            box_corners_pix = np.array([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ], dtype=np.float32)

            flipped_box_corners_pix = np.array([
                [w_img - 1 - u, h_img - 1 - v] for u, v in box_corners_pix
            ], dtype=np.float32)
            
            transformed_corners = cv2.perspectiveTransform(flipped_box_corners_pix.reshape(1, -1, 2), H_mat)

            if transformed_corners is not None:
                projected_boxes.append(transformed_corners.reshape(4, 2))

        return ProjectedData(
            camera_name=frame_data.camera_name,
            is_valid=True,
            warped_image=warped_image,
            warped_masks=warped_masks,
            projected_boxes=projected_boxes,
            extent=[min_x, max_x, max_y, min_y],
            clip_polygon=dst_corners # 클리핑 영역은 원본 이미지의 가시 영역이므로 dst_corners 유지
        )