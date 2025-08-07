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
    하나의 절대 좌표 평면에 세 가지 버전으로 시각화하고 파일로 저장합니다.
    """
    def __init__(self, batch_id: str):
        """
        Visualizer 초기화.
        세 개의 세로 서브플롯을 생성하고 각각의 배경을 설정합니다.
        """
        self.batch_id = batch_id
        self.output_dir = PROJECTION_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 세로로 3개의 서브플롯 생성
        # figsize의 높이를 3배로 늘려 각 플롯이 적절한 크기를 갖도록 함
        self.fig, self.axs = plt.subplots(3, 1, figsize=(20, 30), dpi=VISUALIZATION_CONFIG['dpi'])
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 각 서브플롯에 대한 배경 설정
        self._setup_background(
            self.axs[0], 
            title=f"Situation Awareness Map (Images Only) - Batch ID: {self.batch_id}"
        )
        self._setup_background(
            self.axs[1], 
            title=f"Situation Awareness Map (Full Visualization) - Batch ID: {self.batch_id}"
        )
        self._setup_background(
            self.axs[2], 
            title=f"Situation Awareness Map (Masks, Boxes & Merged Regions) - Batch ID: {self.batch_id}"
        )
        
        # 플롯 간 간격 조절
        self.fig.tight_layout(pad=4.0)

    def _setup_background(self, ax: plt.Axes, title: str):
        """
        주어진 축(ax)에 공장 윤곽, 그리드 등 정적인 배경 요소를 그립니다.
        
        Args:
            ax (plt.Axes): 배경을 그릴 Matplotlib 축 객체
            title (str): 해당 축에 설정할 제목
        """
        ax.set_title(title, fontsize=16)
        
        # 공장 윤곽
        factory_rect = plt.Rectangle(
            (0, 0), FACTORY_WIDTH, FACTORY_HEIGHT, 
            fill=None, edgecolor='black', linewidth=2, label='Factory Contour', zorder=1
        )
        ax.add_patch(factory_rect)

        # 그리드
        grid_interval = VISUALIZATION_CONFIG['grid_interval']
        for x in np.arange(0, FACTORY_WIDTH + grid_interval, grid_interval):
            ax.plot([x, x], [0, FACTORY_HEIGHT], color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        for y in np.arange(0, FACTORY_HEIGHT + grid_interval, grid_interval):
            ax.plot([0, FACTORY_WIDTH], [y, y], color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        ax.set_xlim(-10, FACTORY_WIDTH + 10)
        ax.set_ylim(FACTORY_HEIGHT + 10, -10) # y축을 뒤집어 위쪽이 y가 큰 값이 되도록 함
        ax.set_aspect('equal')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

    def draw(self, projected_data_list: List[ProjectedData]):
        """
        모든 카메라의 투영 결과를 세 개의 플롯에 각각 그립니다.
        """
        
        # --- 색상 매핑 로직 (한 번만 실행) ---
        color_palette = [
            (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
            (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
            (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)
        ]
        pinjig_color_palette = [
            (255, 0, 255),
        ]
        normalized_palette = [(r/255, g/255, b/255) for r, g, b in color_palette]
        
        color_map: Dict[Any, Dict[str, Any]] = {}
        unique_ids = []
        for data in projected_data_list:
            unique_id = getattr(data, 'cctv_id', id(data))
            if unique_id not in unique_ids:
                unique_ids.append(unique_id)

        for i, uid in enumerate(unique_ids):
            color_index = i % len(color_palette)
            pinjig_color_index = i % len(pinjig_color_palette)
            color_map[uid] = {
                'mask_color': color_palette[color_index],
                'pinjig_color': pinjig_color_palette[pinjig_color_index],
                'box_color': normalized_palette[color_index]
            }
        # --- 색상 매핑 로직 끝 ---

        # 위쪽 플롯: 이미지만
        self._draw_on_ax(self.axs[0], projected_data_list, color_map, 
                         show_image=True, show_masks=False, show_boxes=False, show_labels=False, show_merged=False)
        
        # 중간 플롯: 이미지, 마스크, 경계 상자, 라벨 모두
        self._draw_on_ax(self.axs[1], projected_data_list, color_map, 
                         show_image=True, show_masks=True, show_boxes=True, show_labels=True, show_merged=False)
        
        # 아래쪽 플롯: 마스크, 경계 상자, 라벨, 병합된 박스
        self._draw_on_ax(self.axs[2], projected_data_list, color_map, 
                         show_image=False, show_masks=True, show_boxes=True, show_labels=True, show_merged=True)

    def _draw_on_ax(self, ax: plt.Axes, projected_data_list: List[ProjectedData], color_map: Dict, 
                    show_image: bool, show_masks: bool, show_boxes: bool, show_labels: bool, show_merged: bool):
        """
        지정된 축(ax)에 모든 투영 데이터를 그리는 헬퍼 함수.

        Args:
            ax (plt.Axes): 그림을 그릴 축.
            projected_data_list (List[ProjectedData]): 시각화할 데이터 리스트.
            color_map (Dict): CCTV ID별 색상 정보 맵.
            show_image (bool): 워핑된 원본 이미지를 표시할지 여부.
            show_masks (bool): 마스크를 표시할지 여부.
            show_boxes (bool): 경계 상자를 표시할지 여부.
            show_labels (bool): assembly 라벨을 표시할지 여부.
            show_merged (bool): 병합된 박스를 표시할지 여부.
        """
        for data in projected_data_list:
            if not data.is_valid:
                continue
            
            unique_id = getattr(data, 'cctv_id', id(data))
            colors = color_map.get(unique_id, {
                'mask_color': VISUALIZATION_CONFIG.get('mask_color', (255, 0, 0)),
                'pinjig_color': (255, 200, 100),
                'box_color': VISUALIZATION_CONFIG.get('box_color', (1, 0, 0))
            })

            clip_polygon = Polygon(data.clip_polygon, closed=True, facecolor='none', edgecolor='none')
            ax.add_patch(clip_polygon)

            # 1. 워핑된 원본 이미지 표시
            if show_image and VISUALIZATION_CONFIG.get('show_warped_image', True) and data.warped_image is not None:
                im = ax.imshow(
                    data.warped_image, 
                    extent=data.extent, 
                    alpha=0.6, 
                    interpolation='antialiased',
                    zorder=2
                )
                im.set_clip_path(clip_polygon)
            
            # 2. 워핑된 boundary 마스크 표시
            if show_masks and VISUALIZATION_CONFIG.get('show_boundary_masks', True):
                for mask in data.warped_masks:
                    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    alpha = VISUALIZATION_CONFIG.get('mask_alpha', 0.5)
                    mask_rgba[mask == 255] = [*colors['mask_color'], int(alpha * 255)]
                    
                    mask_im = ax.imshow(
                        mask_rgba, 
                        extent=data.extent, 
                        interpolation='nearest',
                        zorder=3
                    )
                    mask_im.set_clip_path(clip_polygon)
            
            # 3. 워핑된 pinjig 마스크 표시
            if show_masks and VISUALIZATION_CONFIG.get('show_pinjig_masks', True):
                for mask in data.warped_pinjig_masks:
                    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    alpha = VISUALIZATION_CONFIG.get('pinjig_mask_alpha', 0.4)
                    mask_rgba[mask == 255] = [*colors['pinjig_color'], int(alpha * 255)]
                    
                    mask_im = ax.imshow(
                        mask_rgba, 
                        extent=data.extent, 
                        interpolation='nearest',
                        zorder=3.5
                    )
                    mask_im.set_clip_path(clip_polygon)

            # 4. 감지된 객체(사각형) 표시
            if show_boxes and VISUALIZATION_CONFIG.get('show_object_boxes', True):
                for box_vertices in data.projected_boxes:
                    poly = Polygon(
                        box_vertices, 
                        closed=True, 
                        fill=False, 
                        edgecolor=colors['box_color'],
                        linewidth=VISUALIZATION_CONFIG.get('box_linewidth', 2),
                        zorder=4
                    )
                    ax.add_patch(poly)
                
                # # Assembly classification 박스 표시 (점선으로 구분)
                # for box_vertices in data.projected_assembly_boxes:
                #     poly = Polygon(
                #         box_vertices, 
                #         closed=True, 
                #         fill=False, 
                #         edgecolor=colors['box_color'],
                #         linewidth=VISUALIZATION_CONFIG.get('box_linewidth', 2),
                #         linestyle='--',  # 점선으로 구분
                #         zorder=4
                #     )
                #     ax.add_patch(poly)
            
            # 5. Assembly classification 라벨 표시
            if show_labels and VISUALIZATION_CONFIG.get('show_assembly_labels', True):
                for label_info in data.projected_assembly_labels:
                    x, y = label_info['position']
                    label_text = label_info['label']
                    confidence = label_info['confidence']
                    
                    # 라벨 텍스트 생성 (신뢰도 포함)
                    display_text = f"{label_text}\n({confidence:.2f})"
                    
                    # 텍스트 표시 (배경 박스와 함께)
                    ax.text(
                        x, y, display_text,
                        color='white',
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(
                            boxstyle='round,pad=0.3',
                            facecolor=colors['box_color'],
                            edgecolor='white',
                            alpha=0.8
                        ),
                        zorder=5
                    )
        
        # 6. 병합된 박스 표시 (세 번째 플롯에서만)
        if show_merged:
            # 모든 ProjectedData에서 merged_boxes 수집 (중복 제거)
            all_merged_boxes = []
            for data in projected_data_list:
                if hasattr(data, 'merged_boxes') and data.merged_boxes:
                    # 첫 번째로 유효한 merged_boxes를 사용 (모든 ProjectedData가 동일한 merged_boxes를 가짐)
                    all_merged_boxes = data.merged_boxes
                    break
            
            # 병합된 박스들을 빨간색 굵은 선으로 표시
            for merged_box in all_merged_boxes:
                poly = Polygon(
                    merged_box,
                    closed=True,
                    fill=False,
                    edgecolor='red',  # 빨간색으로 강조
                    linewidth=3,  # 굵은 선
                    linestyle='-',  # 실선
                    alpha=0.9,
                    zorder=6  # 다른 요소들보다 위에 표시
                )
                ax.add_patch(poly)
                
                # 병합 박스에 "MERGED" 라벨 추가
                center_x = np.mean(merged_box[:, 0])
                center_y = np.mean(merged_box[:, 1])
                ax.text(
                    center_x, center_y, "MERGED",
                    color='white',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='red',
                        edgecolor='white',
                        alpha=0.8
                    ),
                    zorder=7
                )
    
    def save_and_close(self):
        """결과 이미지를 파일로 저장하고 plot을 닫습니다."""
        output_path = self.output_dir / f"projection_{self.batch_id}.png"
        self.fig.savefig(output_path, bbox_inches='tight')
        plt.close(self.fig)
        print(f"🖼️  상황 인식 맵이 '{output_path}'에 저장되었습니다.")