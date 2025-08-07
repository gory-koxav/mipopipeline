# cctv_event_detector/situation_awareness/overlap_analyzer.py

import numpy as np
from typing import List, Dict, Tuple, Set
from cctv_event_detector.core.models import ProjectedData
from itertools import combinations

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
    print("✅ Shapely 라이브러리를 사용하여 정확한 다각형 IOU를 계산합니다.")
except ImportError:
    SHAPELY_AVAILABLE = False
    print("⚠️ Shapely 라이브러리가 설치되지 않았습니다. 대체 방법을 사용합니다.")
    print("   정확한 IOU 계산을 위해 'pip install shapely'를 실행하세요.")

class OverlapAnalyzer:
    """
    서로 다른 카메라의 ProjectedData에서 객체 경계 상자들 간의 IOU를 계산하고,
    특정 threshold 이상으로 겹치는 박스들을 병합하는 기능을 제공합니다.
    사다리꼴 형태의 다각형을 정확하게 처리합니다.
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        OverlapAnalyzer를 초기화합니다.
        
        :param iou_threshold: IOU가 이 값 이상일 때 박스들을 병합할 threshold
        """
        self.iou_threshold = iou_threshold
        print(f"🔍 OverlapAnalyzer 초기화 완료 (IOU Threshold: {self.iou_threshold})")
        if not SHAPELY_AVAILABLE:
            print("   ⚠️ 대체 알고리즘을 사용합니다. 정확도가 떨어질 수 있습니다.")
    
    def _calculate_polygon_iou_shapely(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """
        Shapely를 사용하여 두 다각형의 정확한 IOU를 계산합니다.
        
        :param poly1: 첫 번째 다각형의 꼭짓점 좌표 (N x 2)
        :param poly2: 두 번째 다각형의 꼭짓점 좌표 (N x 2)
        :return: IOU 값 (0.0 ~ 1.0)
        """
        try:
            # Shapely Polygon 객체 생성
            polygon1 = ShapelyPolygon(poly1)
            polygon2 = ShapelyPolygon(poly2)
            
            # 유효한 다각형인지 확인하고 필요시 수정
            if not polygon1.is_valid:
                polygon1 = make_valid(polygon1)
            if not polygon2.is_valid:
                polygon2 = make_valid(polygon2)
            
            # 교집합과 합집합 계산
            intersection = polygon1.intersection(polygon2)
            union = polygon1.union(polygon2)
            
            # IOU 계산
            if union.area == 0:
                return 0.0
            
            iou = intersection.area / union.area
            return iou
            
        except Exception as e:
            print(f"   ⚠️ Shapely IOU 계산 중 오류: {e}")
            return 0.0
    
    def _calculate_polygon_iou_fallback(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """
        Shapely가 없을 때 사용하는 대체 IOU 계산 방법.
        Sutherland-Hodgman 알고리즘을 사용하여 다각형 교집합을 구합니다.
        
        :param poly1: 첫 번째 다각형의 꼭짓점 좌표 (N x 2)
        :param poly2: 두 번째 다각형의 꼭짓점 좌표 (N x 2)
        :return: IOU 값 (0.0 ~ 1.0)
        """
        def polygon_area(vertices):
            """Shoelace formula를 사용한 다각형 면적 계산"""
            n = len(vertices)
            if n < 3:
                return 0.0
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i][0] * vertices[j][1]
                area -= vertices[j][0] * vertices[i][1]
            return abs(area) / 2.0
        
        def line_intersection(p1, p2, p3, p4):
            """두 선분의 교점 계산"""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return None
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return [x, y]
            return None
        
        def sutherland_hodgman_clip(subject_polygon, clip_polygon):
            """Sutherland-Hodgman 알고리즘으로 다각형 교집합 구하기"""
            def inside_edge(point, edge_start, edge_end):
                """점이 엣지의 왼쪽(내부)에 있는지 확인"""
                return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - 
                       (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0
            
            output_list = list(subject_polygon)
            
            for i in range(len(clip_polygon)):
                if len(output_list) == 0:
                    break
                    
                input_list = output_list
                output_list = []
                
                edge_start = clip_polygon[i]
                edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
                
                for j in range(len(input_list)):
                    current_vertex = input_list[j]
                    previous_vertex = input_list[j - 1]
                    
                    if inside_edge(current_vertex, edge_start, edge_end):
                        if not inside_edge(previous_vertex, edge_start, edge_end):
                            intersection = line_intersection(
                                previous_vertex, current_vertex,
                                edge_start, edge_end
                            )
                            if intersection:
                                output_list.append(intersection)
                        output_list.append(current_vertex)
                    elif inside_edge(previous_vertex, edge_start, edge_end):
                        intersection = line_intersection(
                            previous_vertex, current_vertex,
                            edge_start, edge_end
                        )
                        if intersection:
                            output_list.append(intersection)
            
            return output_list
        
        # 교집합 다각형 구하기
        intersection_vertices = sutherland_hodgman_clip(poly1.tolist(), poly2.tolist())
        
        if len(intersection_vertices) < 3:
            return 0.0
        
        # 면적 계산
        area1 = polygon_area(poly1)
        area2 = polygon_area(poly2)
        intersection_area = polygon_area(intersection_vertices)
        
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        두 경계 상자(사다리꼴) 간의 IOU를 계산합니다.
        
        :param box1: 첫 번째 박스의 꼭짓점 좌표 (4 x 2)
        :param box2: 두 번째 박스의 꼭짓점 좌표 (4 x 2)
        :return: IOU 값 (0.0 ~ 1.0)
        """
        if SHAPELY_AVAILABLE:
            return self._calculate_polygon_iou_shapely(box1, box2)
        else:
            return self._calculate_polygon_iou_fallback(box1, box2)
    
    def _create_enclosing_box(self, boxes: List[np.ndarray]) -> np.ndarray:
        """
        여러 박스들을 모두 포함하는 최소 경계 상자를 생성합니다.
        
        :param boxes: 병합할 박스들의 리스트
        :return: 모든 박스를 포함하는 최소 경계 상자의 꼭짓점 좌표
        """
        # 모든 박스의 꼭짓점을 하나의 배열로 결합
        all_points = np.vstack(boxes)
        
        # 최소/최대 좌표 찾기
        min_x = np.min(all_points[:, 0])
        max_x = np.max(all_points[:, 0])
        min_y = np.min(all_points[:, 1])
        max_y = np.max(all_points[:, 1])
        
        # 새로운 경계 상자의 꼭짓점 생성 (직사각형)
        enclosing_box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])
        
        return enclosing_box
    
    def analyze_overlaps(self, projected_data_list: List[ProjectedData]) -> List[ProjectedData]:
        """
        ProjectedData 리스트를 분석하여 서로 다른 카메라 간의 겹치는 박스들을 찾고,
        병합된 박스 정보를 추가합니다.
        
        :param projected_data_list: 분석할 ProjectedData 리스트
        :return: 병합된 박스 정보가 추가된 ProjectedData 리스트
        """
        print("\n" + "="*80)
        print("📊 Overlap Analysis 시작")
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
            
            print(f"\n--- 카메라 '{data1.camera_name}' vs '{data2.camera_name}' 비교 시작 ---")
            print(f"  📦 {data1.camera_name}: {len(data1.projected_boxes)}개의 객체 경계 박스")
            print(f"  📦 {data2.camera_name}: {len(data2.projected_boxes)}개의 객체 경계 박스")
            
            if len(data1.projected_boxes) == 0 or len(data2.projected_boxes) == 0:
                print(f"  ⏭️ 박스가 없는 카메라가 있어 비교를 건너뜁니다.")
                continue
            
            # 각 박스 쌍에 대해 IOU 계산
            overlapping_groups = []  # IOU threshold를 넘는 박스 그룹들
            
            for idx1, box1 in enumerate(data1.projected_boxes):
                for idx2, box2 in enumerate(data2.projected_boxes):
                    # 사다리꼴 형태의 정확한 IOU 계산
                    iou = self._calculate_iou(box1, box2)
                    
                    if iou > 0:  # IOU가 0보다 크면 로그 출력
                        print(f"    🔍 Box[{data1.camera_name}:{idx1}] ↔ Box[{data2.camera_name}:{idx2}]: IOU = {iou:.4f}")
                    
                    if iou >= self.iou_threshold:
                        print(f"    ✨ Threshold 초과! Box[{data1.camera_name}:{idx1}] ↔ Box[{data2.camera_name}:{idx2}]: IOU = {iou:.4f}")
                        
                        # 기존 그룹에 추가할지 새 그룹을 만들지 결정
                        added_to_group = False
                        for group in overlapping_groups:
                            # 현재 박스들이 그룹의 기존 박스들과 연결되어 있는지 확인
                            if any(np.array_equal(box, box1) or np.array_equal(box, box2) for box in group):
                                if not any(np.array_equal(box, box1) for box in group):
                                    group.append(box1)
                                if not any(np.array_equal(box, box2) for box in group):
                                    group.append(box2)
                                added_to_group = True
                                break
                        
                        if not added_to_group:
                            overlapping_groups.append([box1, box2])
            
            # 각 그룹에 대해 병합 박스 생성
            for group_idx, group in enumerate(overlapping_groups):
                merged_box = self._create_enclosing_box(group)
                merged_boxes.append(merged_box)
                print(f"  🎯 그룹 {group_idx + 1}: {len(group)}개의 박스를 병합하여 큰 경계 상자 생성")
                print(f"     병합된 박스 범위: X[{merged_box[:, 0].min():.2f}, {merged_box[:, 0].max():.2f}], "
                      f"Y[{merged_box[:, 1].min():.2f}, {merged_box[:, 1].max():.2f}]")
        
        print(f"\n🏁 Overlap Analysis 완료: 총 {len(merged_boxes)}개의 병합 박스 생성")
        print("="*80)
        
        # 병합된 박스들을 ProjectedData에 추가
        # 모든 ProjectedData에 merged_boxes 속성 추가
        for data in projected_data_list:
            if not hasattr(data, 'merged_boxes'):
                data.merged_boxes = []
        
        # 병합된 박스들을 저장 (visualizer에서 사용하기 위해)
        # 마지막 ProjectedData에 모든 병합 박스 추가 (또는 별도 처리)
        if merged_boxes and projected_data_list:
            # 모든 ProjectedData에 동일한 merged_boxes 추가
            for data in projected_data_list:
                data.merged_boxes = merged_boxes.copy()
        
        return projected_data_list