# cctv_event_detector/situation_awareness/visualizer.py

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import cv2
from typing import List

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
        for data in projected_data_list:
            if not data.is_valid:
                continue

            # 클리핑을 위한 폴리곤 생성
            clip_polygon = Polygon(data.clip_polygon, closed=True, facecolor='none', edgecolor='none')
            self.ax.add_patch(clip_polygon)

            # 1. 워핑된 원본 이미지 표시
            if VISUALIZATION_CONFIG['show_warped_image'] and data.warped_image is not None:
                im = self.ax.imshow(
                    data.warped_image, 
                    extent=data.extent, 
                    alpha=0.6, 
                    interpolation='antialiased',
                    zorder=2
                )
                im.set_clip_path(clip_polygon)
            
            # 2. 워핑된 마스크 표시
            if VISUALIZATION_CONFIG['show_boundary_masks']:
                for mask in data.warped_masks:
                    # 마스크를 RGBA 이미지로 변환
                    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    color = VISUALIZATION_CONFIG['mask_color']
                    alpha = VISUALIZATION_CONFIG['mask_alpha']
                    mask_rgba[mask == 255] = [*color, int(alpha * 255)]
                    
                    mask_im = self.ax.imshow(
                        mask_rgba, 
                        extent=data.extent, 
                        interpolation='nearest',
                        zorder=3
                    )
                    mask_im.set_clip_path(clip_polygon)

            # 3. 감지된 객체(사각형) 표시
            if VISUALIZATION_CONFIG['show_object_boxes']:
                for box_vertices in data.projected_boxes:
                    poly = Polygon(
                        box_vertices, 
                        closed=True, 
                        fill=False, 
                        edgecolor=VISUALIZATION_CONFIG['box_color'], 
                        linewidth=VISUALIZATION_CONFIG['box_linewidth'],
                        zorder=4
                    )
                    self.ax.add_patch(poly)

    def save_and_close(self):
        """결과 이미지를 파일로 저장하고 plot을 닫습니다."""
        output_path = self.output_dir / f"projection_{self.batch_id}.png"
        self.fig.savefig(output_path, bbox_inches='tight')
        plt.close(self.fig)
        print(f"🖼️  상황 인식 맵이 '{output_path}'에 저장되었습니다.")