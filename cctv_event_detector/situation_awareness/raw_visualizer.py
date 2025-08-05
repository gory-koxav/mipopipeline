# cctv_event_detector/situation_awareness/raw_visualizer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import List, Dict, Any
from pathlib import Path

from cctv_event_detector.core.models import FrameData
from config import PROJECTION_OUTPUT_DIR, PROJECTION_TARGET_CLASSES

class RawDataVisualizer:
    """
    원본 프레임 데이터를 변환 없이 개별적으로 시각화하는 클래스.
    각 카메라의 이미지, 마스크, 검출된 객체를 원본 상태로 표시합니다.
    """
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.output_dir = PROJECTION_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_classes = set(PROJECTION_TARGET_CLASSES) if PROJECTION_TARGET_CLASSES else None
    
    def visualize_frame(self, frame_data: FrameData, index: int):
        """
        단일 프레임 데이터를 시각화합니다.
        
        Args:
            frame_data: 시각화할 프레임 데이터
            index: 프레임 인덱스 (파일명에 사용)
        """
        # 이미지 로드
        img = cv2.imread(frame_data.image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"경고: 이미지 로드 실패 - {frame_data.image_path}")
            return
        
        # BGR to RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]
        
        # 서브플롯 설정: 이미지, 마스크, 객체 박스를 나란히 표시
        num_masks = len(frame_data.boundary_masks)
        total_cols = 2 + num_masks  # 원본 이미지, 객체 박스가 있는 이미지, 마스크들
        
        fig, axes = plt.subplots(1, total_cols, figsize=(6*total_cols, 6))
        fig.suptitle(f'Camera: {frame_data.camera_name} - Frame: {frame_data.image_id}', fontsize=16)
        
        # 단일 축인 경우 리스트로 변환
        if total_cols == 1:
            axes = [axes]
        
        # 1. 원본 이미지 표시
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 2. 객체 박스가 그려진 이미지
        img_with_boxes = img.copy()
        ax_boxes = axes[1]
        ax_boxes.imshow(img_with_boxes)
        ax_boxes.set_title('Detected Objects')
        ax_boxes.axis('off')
        
        # 객체 박스 그리기
        detection_count = 0
        for det in frame_data.detections:
            class_name = det.get('class_name')
            
            # target_classes가 지정된 경우 필터링
            if self.target_classes and class_name not in self.target_classes:
                continue
            
            x_min, y_min, w_box, h_box = det['bbox_xywh']
            confidence = det.get('confidence', 0)
            
            # 박스 그리기
            rect = patches.Rectangle(
                (x_min, y_min), w_box, h_box,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax_boxes.add_patch(rect)
            
            # 라벨 추가
            label = f'{class_name}: {confidence:.2f}'
            ax_boxes.text(
                x_min, y_min - 5, label,
                color='red', fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, pad=2)
            )
            
            detection_count += 1
        
        # 검출된 객체 수 표시
        ax_boxes.text(
            10, 30, f'Detected: {detection_count} objects',
            color='green', fontsize=12, weight='bold',
            bbox=dict(facecolor='white', alpha=0.8, pad=5)
        )
        
        # 3. 바운더리 마스크들 표시
        for i, mask in enumerate(frame_data.boundary_masks):
            ax_mask = axes[2 + i]
            
            # 마스크를 컬러맵으로 표시
            mask_colored = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            mask_colored[mask > 0] = [255, 0, 0]  # 빨간색으로 마스크 표시
            
            # 원본 이미지와 마스크 오버레이
            overlay = img.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + mask_colored[mask > 0] * 0.5
            
            ax_mask.imshow(overlay)
            ax_mask.set_title(f'Boundary Mask {i+1}')
            ax_mask.axis('off')
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 파일로 저장
        output_filename = f"raw_data_{self.batch_id}_{index:03d}_{frame_data.camera_name}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"📸 원본 데이터 시각화 저장: {output_path}")
    
    def visualize_all_frames(self, all_frames_data: List[FrameData]):
        """
        모든 프레임 데이터를 개별적으로 시각화합니다.
        
        Args:
            all_frames_data: 시각화할 모든 프레임 데이터 리스트
        """
        print(f"\n🎨 {len(all_frames_data)}개 프레임의 원본 데이터 시각화를 시작합니다...")
        
        for idx, frame_data in enumerate(all_frames_data):
            self.visualize_frame(frame_data, idx)
        
        print(f"✅ 모든 원본 데이터 시각화가 완료되었습니다.")
        
        # 요약 이미지도 생성 (모든 카메라의 원본 이미지를 한 번에 보기)
        self._create_summary_image(all_frames_data)
    
    def _create_summary_image(self, all_frames_data: List[FrameData]):
        """
        모든 카메라의 이미지를 하나의 요약 이미지로 생성합니다.
        """
        num_frames = len(all_frames_data)
        if num_frames == 0:
            return
        
        # 그리드 레이아웃 계산
        cols = min(4, num_frames)  # 최대 4열
        rows = (num_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        fig.suptitle(f'All Cameras Summary - Batch: {self.batch_id}', fontsize=20)
        
        # 축을 2D 배열로 변환
        if num_frames == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, frame_data in enumerate(all_frames_data):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # 이미지 로드
            img = cv2.imread(frame_data.image_path, cv2.IMREAD_COLOR)
            if img is None:
                ax.text(0.5, 0.5, 'Image Load Failed', ha='center', va='center')
                ax.set_title(f'{frame_data.camera_name}')
                ax.axis('off')
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 이미지에 검출된 객체 표시
            for det in frame_data.detections:
                class_name = det.get('class_name')
                if self.target_classes and class_name not in self.target_classes:
                    continue
                
                x_min, y_min, w_box, h_box = det['bbox_xywh']
                rect = patches.Rectangle(
                    (x_min, y_min), w_box, h_box,
                    linewidth=1, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
            
            ax.imshow(img)
            ax.set_title(f'{frame_data.camera_name}', fontsize=12)
            ax.axis('off')
        
        # 빈 서브플롯 숨기기
        for idx in range(num_frames, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 요약 이미지 저장
        summary_path = self.output_dir / f"raw_data_summary_{self.batch_id}.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"📊 요약 이미지 저장: {summary_path}")