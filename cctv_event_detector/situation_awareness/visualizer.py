# cctv_event_detector/situation_awareness/visualizer.py

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import cv2
from typing import List, Dict, Any

from cctv_event_detector.core.models import ProjectedData
from config import (
    FACTORY_WIDTH, FACTORY_HEIGHT, VISUALIZATION_CONFIG, PROJECTION_OUTPUT_DIR
)

class Visualizer:
    """
    투영된 데이터(ProjectedData) 리스트를 받아 Matplotlib를 사용하여
    하나의 절대 좌표 평면에 시각화하고 파일로 저장합니다.
    """
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.output_dir = PROJECTION_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fig, self.ax = plt.subplots(figsize=(20, 10), dpi=VISUALIZATION_CONFIG['dpi'])
        plt.style.use('seaborn-v0_8-darkgrid')
        self._setup_background()

    def _setup_background(self):
        """공장 윤곽, 그리드 등 정적인 배경 요소를 그립니다."""
        self.ax.set_title(f"Situation Awareness Map (Batch ID: {self.batch_id})", fontsize=16)
        
        # 공장 윤곽
        factory_rect = plt.Rectangle(
            (0, 0), FACTORY_WIDTH, FACTORY_HEIGHT, 
            fill=None, edgecolor='black', linewidth=2, label='Factory Contour', zorder=1
        )
        self.ax.add_patch(factory_rect)

        # 그리드
        grid_interval = VISUALIZATION_CONFIG['grid_interval']
        for x in np.arange(0, FACTORY_WIDTH + grid_interval, grid_interval):
            self.ax.plot([x, x], [0, FACTORY_HEIGHT], color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        for y in np.arange(0, FACTORY_HEIGHT + grid_interval, grid_interval):
            self.ax.plot([0, FACTORY_WIDTH], [y, y], color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        self.ax.set_xlim(-10, FACTORY_WIDTH + 10)
        self.ax.set_ylim(FACTORY_HEIGHT + 10, -10) # y축을 뒤집어 위쪽이 y가 큰 값이 되도록 함
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

    def draw(self, projected_data_list: List[ProjectedData]):
        """모든 카메라의 투영 결과를 캔버스에 그립니다."""
        
        # --- 색상 변경 로직 시작 ---
        
        # 각 데이터 소스에 고유한 색상을 할당하기 위한 설정
        # 다양한 색상 팔레트 (RGB 튜플, 0-255)
        color_palette = [
            (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
            (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
            (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)
        ]
        
        # Pinjig 마스크를 위한 별도 색상 팔레트
        pinjig_color_palette = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
            (255, 100, 255), (100, 255, 255), (200, 150, 100), (150, 100, 200)
        ]
        
        # Matplotlib의 `edgecolor`에서 사용하기 위해 0-1 범위로 정규화된 색상
        normalized_palette = [(r/255, g/255, b/255) for r, g, b in color_palette]
        
        color_map: Dict[Any, Dict[str, Any]] = {}
        
        # ProjectedData 객체에 cctv_id와 같은 고유 식별자가 있다고 가정합니다.
        # 이 식별자를 기준으로 고유한 데이터 소스를 찾습니다.
        unique_ids = []
        for data in projected_data_list:
            # data 객체에 'cctv_id' 속성이 있다고 가정합니다.
            unique_id = getattr(data, 'cctv_id', id(data))
            if unique_id not in unique_ids:
                unique_ids.append(unique_id)

        # 고유 식별자별로 색상을 매핑합니다.
        for i, uid in enumerate(unique_ids):
            color_index = i % len(color_palette)
            pinjig_color_index = i % len(pinjig_color_palette)
            color_map[uid] = {
                'mask_color': color_palette[color_index],
                'pinjig_color': pinjig_color_palette[pinjig_color_index],
                'box_color': normalized_palette[color_index]
            }
        
        # --- 색상 변경 로직 끝 ---

        for data in projected_data_list:
            if not data.is_valid:
                continue
            
            # --- 할당된 색상 가져오기 ---
            # data의 고유 식별자를 기준으로 색상을 가져옵니다.
            unique_id = getattr(data, 'cctv_id', id(data))
            
            if unique_id in color_map:
                mask_color = color_map[unique_id]['mask_color']
                pinjig_color = color_map[unique_id]['pinjig_color']
                box_color = color_map[unique_id]['box_color']
            else:
                # 만약의 경우를 대비한 기본값 설정
                mask_color = VISUALIZATION_CONFIG.get('mask_color', (255, 0, 0))
                pinjig_color = (255, 200, 100)  # 기본 pinjig 색상
                box_color = VISUALIZATION_CONFIG.get('box_color', (1, 0, 0))
            # --- 색상 가져오기 끝 ---

            # 클리핑을 위한 폴리곤 생성
            clip_polygon = Polygon(data.clip_polygon, closed=True, facecolor='none', edgecolor='none')
            self.ax.add_patch(clip_polygon)

            # 1. 워핑된 원본 이미지 표시
            if VISUALIZATION_CONFIG.get('show_warped_image', True) and data.warped_image is not None:
                im = self.ax.imshow(
                    data.warped_image, 
                    extent=data.extent, 
                    alpha=0.6, 
                    interpolation='antialiased',
                    zorder=2
                )
                im.set_clip_path(clip_polygon)
            
            # 2. 워핑된 boundary 마스크 표시
            if VISUALIZATION_CONFIG.get('show_boundary_masks', True):
                for mask in data.warped_masks:
                    # 마스크를 RGBA 이미지로 변환
                    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    alpha = VISUALIZATION_CONFIG.get('mask_alpha', 0.5)
                    # 동적으로 할당된 색상 사용
                    mask_rgba[mask == 255] = [*mask_color, int(alpha * 255)]
                    
                    mask_im = self.ax.imshow(
                        mask_rgba, 
                        extent=data.extent, 
                        interpolation='nearest',
                        zorder=3
                    )
                    mask_im.set_clip_path(clip_polygon)
            
            # 3. 워핑된 pinjig 마스크 표시 (새로 추가)
            if VISUALIZATION_CONFIG.get('show_pinjig_masks', True):  # 기본값 True
                for mask in data.warped_pinjig_masks:
                    # pinjig 마스크를 RGBA 이미지로 변환
                    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    alpha = VISUALIZATION_CONFIG.get('pinjig_mask_alpha', 0.4)
                    # pinjig 전용 색상 사용
                    mask_rgba[mask == 255] = [*pinjig_color, int(alpha * 255)]
                    
                    mask_im = self.ax.imshow(
                        mask_rgba, 
                        extent=data.extent, 
                        interpolation='nearest',
                        zorder=3.5  # boundary mask보다 약간 위에 표시
                    )
                    mask_im.set_clip_path(clip_polygon)

            # 4. 감지된 객체(사각형) 표시
            if VISUALIZATION_CONFIG.get('show_object_boxes', True):
                for box_vertices in data.projected_boxes:
                    poly = Polygon(
                        box_vertices, 
                        closed=True, 
                        fill=False, 
                        edgecolor=box_color, # 동적으로 할당된 색상 사용
                        linewidth=VISUALIZATION_CONFIG.get('box_linewidth', 2),
                        zorder=4
                    )
                    self.ax.add_patch(poly)

    def save_and_close(self):
        """결과 이미지를 파일로 저장하고 plot을 닫습니다."""
        output_path = self.output_dir / f"projection_{self.batch_id}.png"
        self.fig.savefig(output_path, bbox_inches='tight')
        plt.close(self.fig)
        print(f"🖼️  상황 인식 맵이 '{output_path}'에 저장되었습니다.")