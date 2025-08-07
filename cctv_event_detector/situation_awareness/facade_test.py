# cctv_event_detector/situation_awareness/facade_test.py

import redis
import time

# ✅ 요구사항: config에서 시각화 대상 클래스 목록을 가져옵니다.
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, PROJECTION_TARGET_CLASSES, PINJIG_TARGET_CLASSES
from .data_aggregator import DataAggregator
from .projector import Projector
from .overlap_analyzer import OverlapAnalyzer  # 박스 기반 분석
from .mask_overlap_analyzer import MaskOverlapAnalyzer  # 마스크 기반 분석
from .visualizer import Visualizer
from .raw_visualizer import RawDataVisualizer  # 새로 추가

class SituationAwarenessFacade:
    """
    상황 인식 시스템의 전체 워크플로우를 관리하고 조정하는 퍼사드 클래스.
    (배치 테스트 버전)
    """
    def __init__(self, iou_threshold: float = 0.3, use_mask_analyzer: bool = False):
        """
        퍼사드 클래스를 초기화합니다. Redis 클라이언트 및 데이터 처리 모듈을 설정합니다.
        
        Args:
            iou_threshold (float): IOU 임계값 (기본값: 0.3)
            use_mask_analyzer (bool): True면 마스크 기반 분석, False면 박스 기반 분석 (기본값: False)
        """
        try:
            # Redis 클라이언트 초기화
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB
            )
            self.redis_client.ping()
            print("✅ Redis 클라이언트가 성공적으로 연결되었습니다.")
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Redis 연결 실패: {e}")
            raise
        
        # 데이터 집계 및 사영 변환 모듈 초기화
        self.aggregator = DataAggregator(self.redis_client)
        
        # ✅ 요구사항: Projector 생성 시 시각화 대상 클래스 목록을 전달합니다.
        self.projector = Projector(target_classes=PROJECTION_TARGET_CLASSES)
        
        # ✅ 분석 방법 선택: 마스크 기반 또는 박스 기반
        self.use_mask_analyzer = use_mask_analyzer
        if use_mask_analyzer:
            print("🎭 마스크 기반 Overlap 분석을 사용합니다.")
            self.overlap_analyzer = MaskOverlapAnalyzer(iou_threshold=iou_threshold)
        else:
            print("📦 박스 기반 Overlap 분석을 사용합니다.")
            self.overlap_analyzer = OverlapAnalyzer(iou_threshold=iou_threshold)

    def process_batch(self, batch_id: str):
        """
        지정된 단일 배치 ID에 대한 전체 처리 로직을 수행합니다.
        
        Args:
            batch_id (str): 처리할 데이터의 배치 ID.
        """
        if not batch_id or batch_id == "여기에_테스트할_배치ID를_입력하세요":
            print("❌ 오류: 유효한 batch_id가 main 함수에 지정되지 않았습니다. 코드를 확인해주세요.")
            return

        print(f"\n🚀 Batch ID '{batch_id}' 처리를 시작합니다...")
        print(f"📋 Detection 타겟 클래스: {PROJECTION_TARGET_CLASSES}")
        print(f"📋 Pinjig 타겟 클래스: {PINJIG_TARGET_CLASSES}")
        print(f"🔍 분석 방법: {'마스크 기반' if self.use_mask_analyzer else '박스 기반'} IOU 계산")
        start_time = time.time()
        
        # 1. Redis에서 데이터 집계 (pinjig 데이터 포함)
        all_frames_data = self.aggregator.get_batch_data(batch_id)
        if not all_frames_data:
            print(f"데이터를 찾을 수 없습니다. Batch ID '{batch_id}'에 해당하는 데이터가 Redis에 있는지 확인해주세요.")
            return

        # Pinjig 데이터 통계 출력
        total_pinjig_masks = sum(len(frame.pinjig_masks) for frame in all_frames_data)
        total_boundary_masks = sum(len(frame.boundary_masks) for frame in all_frames_data)
        total_detections = sum(len(frame.detections) for frame in all_frames_data)
        
        print(f"📊 데이터 통계:")
        print(f"   - 총 {total_detections}개의 객체 감지")
        print(f"   - 총 {total_boundary_masks}개의 boundary 마스크")
        print(f"   - 총 {total_pinjig_masks}개의 pinjig/hbeamjig 마스크")

        # # 🆕 1-1. 원본 데이터 시각화 (flip 및 좌표 변환 없이)
        # print("\n📸 원본 데이터 시각화를 수행합니다...")
        # raw_visualizer = RawDataVisualizer(batch_id)
        # raw_visualizer.visualize_all_frames(all_frames_data)

        # 2. 각 카메라 데이터에 대해 사영 변환 수행 (pinjig 포함)
        projected_results = []
        for frame_data in all_frames_data:
            projected_data = self.projector.project(frame_data)
            projected_results.append(projected_data)
        
        # 유효한 투영 결과 개수 확인
        valid_projections = sum(1 for p in projected_results if p.is_valid)
        print(f"✅ {len(projected_results)}개 카메라 중 {valid_projections}개의 사영 변환을 성공적으로 완료했습니다.")

        # 3. ✅ 새로운 단계: 서로 다른 카메라 간 IOU 계산 및 병합 박스 생성
        print(f"\n🔄 {'마스크' if self.use_mask_analyzer else '박스'} 기반 Overlap 분석을 시작합니다...")
        projected_results = self.overlap_analyzer.analyze_overlaps(projected_results)
        
        # 4. 투영된 결과들을 시각화하고 파일로 저장 (pinjig 및 병합 박스 포함)
        visualizer = Visualizer(batch_id)
        visualizer.draw(projected_results)
        visualizer.save_and_close()
        
        end_time = time.time()
        print(f"🏁 Batch ID '{batch_id}' 처리를 완료했습니다. (총 소요 시간: {end_time - start_time:.2f}초)")