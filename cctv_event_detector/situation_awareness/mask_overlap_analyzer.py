# cctv_event_detector/situation_awareness/mask_overlap_analyzer.py

import numpy as np
import cv2
from typing import List, Dict, Tuple, Set
from cctv_event_detector.core.models import ProjectedData
from itertools import combinations

class MaskOverlapAnalyzer:
    """
    서로 다른 카메라의 ProjectedData에서 warped_masks 간의 IOU를 계산하고,
    특정 threshold 이상으로 겹치는 마스크들을 병합하는 기능을 제공합니다.
    마스크의 실제 픽셀 영역을 기반으로 정확한 IOU를 계산합니다.
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        MaskOverlapAnalyzer를 초기화합니다.
        
        :param iou_threshold: IOU가 이 값 이상일 때 마스크들을 병합할 threshold
        """
        self.iou_threshold = iou_threshold
        print(f"🎭 MaskOverlapAnalyzer 초기화 완료 (IOU Threshold: {self.iou_threshold})")
        print(f"   마스크 픽셀 기반 정확한 IOU 계산을 수행합니다.")
    
    def _transform_mask_to_common_space(self, mask: np.ndarray, extent: List[float], 
                                       target_extent: Tuple[float, float, float, float],
                                       target_shape: Tuple[int, int]) -> np.ndarray:
        """
        마스크를 공통 좌표 공간으로 변환합니다.
        
        :param mask: 원본 마스크
        :param extent: 원본 마스크의 extent [min_x, max_x, max_y, min_y]
        :param target_extent: 목표 공간의 extent (min_x, max_x, min_y, max_y)
        :param target_shape: 목표 마스크의 shape (height, width)
        :return: 변환된 마스크
        """
        # extent를 [min_x, max_x, min_y, max_y] 형식으로 정규화
        src_min_x, src_max_x, src_max_y, src_min_y = extent
        tgt_min_x, tgt_max_x, tgt_min_y, tgt_max_y = target_extent
        
        # 원본 마스크 크기
        src_h, src_w = mask.shape
        tgt_h, tgt_w = target_shape
        
        # 스케일 계산
        src_scale_x = src_w / (src_max_x - src_min_x)
        src_scale_y = src_h / (src_max_y - src_min_y)
        tgt_scale_x = tgt_w / (tgt_max_x - tgt_min_x)
        tgt_scale_y = tgt_h / (tgt_max_y - tgt_min_y)
        
        # 변환 행렬 계산
        # 1. 원본 좌표계에서 월드 좌표계로
        # 2. 월드 좌표계에서 목표 좌표계로
        offset_x = (src_min_x - tgt_min_x) * tgt_scale_x
        offset_y = (src_min_y - tgt_min_y) * tgt_scale_y
        scale_x = src_scale_x / tgt_scale_x
        scale_y = src_scale_y / tgt_scale_y
        
        # 아핀 변환 행렬
        M = np.array([
            [scale_x, 0, offset_x],
            [0, scale_y, offset_y]
        ], dtype=np.float32)
        
        # 마스크 변환
        transformed_mask = cv2.warpAffine(mask, M, (tgt_w, tgt_h), 
                                         flags=cv2.INTER_NEAREST,
                                         borderValue=0)
        
        return transformed_mask
    
    def _calculate_mask_iou(self, mask1: np.ndarray, extent1: List[float],
                           mask2: np.ndarray, extent2: List[float]) -> Tuple[float, np.ndarray]:
        """
        두 마스크 간의 IOU를 계산합니다.
        서로 다른 좌표 공간의 마스크들을 공통 공간으로 변환한 후 계산합니다.
        
        :param mask1: 첫 번째 마스크
        :param extent1: 첫 번째 마스크의 extent
        :param mask2: 두 번째 마스크
        :param extent2: 두 번째 마스크의 extent
        :return: (IOU 값, 병합된 마스크)
        """
        # 공통 좌표 공간 계산 (두 extent를 포함하는 최소 영역)
        min_x = min(extent1[0], extent2[0])
        max_x = max(extent1[1], extent2[1])
        # extent는 [min_x, max_x, max_y, min_y] 형식
        min_y = min(extent1[3], extent2[3])
        max_y = max(extent1[2], extent2[2])
        
        common_extent = (min_x, max_x, min_y, max_y)
        
        # 공통 공간의 해상도 결정 (더 높은 해상도 선택)
        resolution = 10  # pixels per meter (projector의 기본값)
        common_width = int((max_x - min_x) * resolution)
        common_height = int((max_y - min_y) * resolution)
        common_shape = (common_height, common_width)
        
        # 크기 제한 (메모리 절약)
        max_size = 4000
        if common_width > max_size or common_height > max_size:
            scale = max_size / max(common_width, common_height)
            common_width = int(common_width * scale)
            common_height = int(common_height * scale)
            common_shape = (common_height, common_width)
        
        # 두 마스크를 공통 공간으로 변환
        try:
            transformed_mask1 = self._transform_mask_to_common_space(
                mask1, extent1, common_extent, common_shape
            )
            transformed_mask2 = self._transform_mask_to_common_space(
                mask2, extent2, common_extent, common_shape
            )
        except Exception as e:
            print(f"    ⚠️ 마스크 변환 실패: {e}")
            return 0.0, None
        
        # 이진화 (안전을 위해)
        binary_mask1 = (transformed_mask1 > 128).astype(np.uint8)
        binary_mask2 = (transformed_mask2 > 128).astype(np.uint8)
        
        # IOU 계산
        intersection = np.logical_and(binary_mask1, binary_mask2)
        union = np.logical_or(binary_mask1, binary_mask2)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        if union_area == 0:
            return 0.0, None
        
        iou = intersection_area / union_area
        
        # 병합된 마스크 생성 (255 값으로)
        merged_mask = union.astype(np.uint8) * 255
        
        return iou, merged_mask
    
    def _get_bounding_box_from_mask(self, mask: np.ndarray, extent: List[float]) -> np.ndarray:
        """
        마스크로부터 경계 박스를 추출합니다.
        
        :param mask: 마스크 이미지
        :param extent: 마스크의 extent [min_x, max_x, max_y, min_y]
        :return: 경계 박스의 꼭짓점 좌표 (월드 좌표계)
        """
        # 마스크에서 0이 아닌 픽셀 찾기
        points = cv2.findNonZero(mask)
        
        if points is None:
            return np.array([])
        
        # 최소 경계 사각형 찾기
        x, y, w, h = cv2.boundingRect(points)
        
        # 픽셀 좌표를 월드 좌표로 변환
        min_x_world = extent[0]
        max_x_world = extent[1]
        min_y_world = extent[3]
        max_y_world = extent[2]
        
        mask_h, mask_w = mask.shape
        
        # 스케일 계산
        scale_x = (max_x_world - min_x_world) / mask_w
        scale_y = (max_y_world - min_y_world) / mask_h
        
        # 박스 꼭짓점을 월드 좌표로 변환
        box_min_x = min_x_world + x * scale_x
        box_max_x = min_x_world + (x + w) * scale_x
        box_min_y = min_y_world + y * scale_y
        box_max_y = min_y_world + (y + h) * scale_y
        
        # 박스 꼭짓점 생성
        box = np.array([
            [box_min_x, box_min_y],
            [box_max_x, box_min_y],
            [box_max_x, box_max_y],
            [box_min_x, box_max_y]
        ])
        
        return box
    
    def _create_enclosing_box_from_masks(self, mask_indices: List[Tuple[int, int]], 
                                        valid_data_list: List[ProjectedData]) -> np.ndarray:
        """
        여러 마스크들을 모두 포함하는 최소 경계 상자를 생성합니다.
        
        :param mask_indices: (카메라 인덱스, 마스크 인덱스) 튜플 리스트
        :param valid_data_list: 유효한 ProjectedData 리스트
        :return: 모든 마스크를 포함하는 최소 경계 상자의 꼭짓점 좌표
        """
        all_boxes = []
        
        for cam_idx, mask_idx in mask_indices:
            data = valid_data_list[cam_idx]
            mask = data.warped_masks[mask_idx]
            extent = data.extent
            
            box = self._get_bounding_box_from_mask(mask, extent)
            if box.size > 0:
                all_boxes.append(box)
        
        if not all_boxes:
            return np.array([])
        
        # 모든 박스의 꼭짓점을 하나의 배열로 결합
        all_points = np.vstack(all_boxes)
        
        # 최소/최대 좌표 찾기
        min_x = np.min(all_points[:, 0])
        max_x = np.max(all_points[:, 0])
        min_y = np.min(all_points[:, 1])
        max_y = np.max(all_points[:, 1])
        
        # 새로운 경계 상자의 꼭짓점 생성
        enclosing_box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])
        
        return enclosing_box
    
    def analyze_overlaps(self, projected_data_list: List[ProjectedData]) -> List[ProjectedData]:
        """
        ProjectedData 리스트를 분석하여 서로 다른 카메라 간의 겹치는 마스크들을 찾고,
        병합된 박스 정보를 추가합니다.
        
        :param projected_data_list: 분석할 ProjectedData 리스트
        :return: 병합된 박스 정보가 추가된 ProjectedData 리스트
        """
        print("\n" + "="*80)
        print("🎭 Mask-based Overlap Analysis 시작")
        print("="*80)
        
        # 유효한 ProjectedData만 필터링
        valid_data_list = [data for data in projected_data_list if data.is_valid]
        
        if len(valid_data_list) < 2:
            print("⚠️ 유효한 카메라 데이터가 2개 미만이므로 overlap 분석을 수행하지 않습니다.")
            return projected_data_list
        
        print(f"✅ 총 {len(valid_data_list)}개의 유효한 카메라 데이터를 분석합니다.")
        
        # 병합된 박스들을 저장할 리스트
        merged_boxes = []
        
        # 모든 카메라 쌍에 대해 비교
        for i, j in combinations(range(len(valid_data_list)), 2):
            data1 = valid_data_list[i]
            data2 = valid_data_list[j]
            
            print(f"\n--- 카메라 '{data1.camera_name}' vs '{data2.camera_name}' 마스크 비교 시작 ---")
            print(f"  🎭 {data1.camera_name}: {len(data1.warped_masks)}개의 boundary 마스크")
            print(f"  🎭 {data2.camera_name}: {len(data2.warped_masks)}개의 boundary 마스크")
            
            if len(data1.warped_masks) == 0 or len(data2.warped_masks) == 0:
                print(f"  ⏭️ 마스크가 없는 카메라가 있어 비교를 건너뜁니다.")
                continue
            
            # 각 마스크 쌍에 대해 IOU 계산
            overlapping_groups = []  # IOU threshold를 넘는 마스크 인덱스 그룹들
            
            for idx1, mask1 in enumerate(data1.warped_masks):
                for idx2, mask2 in enumerate(data2.warped_masks):
                    # 마스크 기반 정확한 IOU 계산
                    iou, merged_mask = self._calculate_mask_iou(
                        mask1, data1.extent, mask2, data2.extent
                    )
                    
                    if iou > 0:  # IOU가 0보다 크면 로그 출력
                        print(f"    🔍 Mask[{data1.camera_name}:{idx1}] ↔ Mask[{data2.camera_name}:{idx2}]: IOU = {iou:.4f}")
                    
                    if iou >= self.iou_threshold:
                        print(f"    ✨ Threshold 초과! Mask[{data1.camera_name}:{idx1}] ↔ Mask[{data2.camera_name}:{idx2}]: IOU = {iou:.4f}")
                        
                        # 마스크 인덱스를 저장
                        mask_index1 = (i, idx1)
                        mask_index2 = (j, idx2)
                        
                        # 기존 그룹에 추가할지 새 그룹을 만들지 결정
                        added_to_group = False
                        for group in overlapping_groups:
                            if mask_index1 in group or mask_index2 in group:
                                if mask_index1 not in group:
                                    group.append(mask_index1)
                                if mask_index2 not in group:
                                    group.append(mask_index2)
                                added_to_group = True
                                break
                        
                        if not added_to_group:
                            overlapping_groups.append([mask_index1, mask_index2])
            
            # 각 그룹에 대해 병합 박스 생성
            for group_idx, group in enumerate(overlapping_groups):
                merged_box = self._create_enclosing_box_from_masks(group, valid_data_list)
                if merged_box.size > 0:
                    merged_boxes.append(merged_box)
                    print(f"  🎯 그룹 {group_idx + 1}: {len(group)}개의 마스크를 병합하여 큰 경계 상자 생성")
                    print(f"     병합된 박스 범위: X[{merged_box[:, 0].min():.2f}, {merged_box[:, 0].max():.2f}], "
                          f"Y[{merged_box[:, 1].min():.2f}, {merged_box[:, 1].max():.2f}]")
        
        print(f"\n🏁 Mask-based Overlap Analysis 완료: 총 {len(merged_boxes)}개의 병합 박스 생성")
        print("="*80)
        
        # 병합된 박스들을 ProjectedData에 추가
        # 모든 ProjectedData에 merged_boxes 속성 추가
        for data in projected_data_list:
            if not hasattr(data, 'merged_boxes'):
                data.merged_boxes = []
        
        # 병합된 박스들을 저장 (visualizer에서 사용하기 위해)
        if merged_boxes and projected_data_list:
            # 모든 ProjectedData에 동일한 merged_boxes 추가
            for data in projected_data_list:
                data.merged_boxes = merged_boxes.copy()
        
        return projected_data_list