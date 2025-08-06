# cctv_event_detector/situation_awareness/data_aggregator.py

import redis
import json
import numpy as np
from typing import List, Dict, Any, Tuple

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, IMAGE_SHAPE, PINJIG_TARGET_CLASSES, PROJECTION_TARGET_CLASSES
from cctv_event_detector.core.models import FrameData

class DataAggregator:
    """
    Redis에서 특정 배치 ID와 관련된 모든 데이터를 수집하고
    처리하기 쉬운 데이터 모델(FrameData)로 파싱하는 역할을 합니다.
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.image_shape = IMAGE_SHAPE
        self.pinjig_target_classes = PINJIG_TARGET_CLASSES  # config에서 가져온 pinjig 타겟 클래스
        self.projection_target_classes = PROJECTION_TARGET_CLASSES  # detection 타겟 클래스

    def _decompress_mask(self, compressed_mask: bytes, shape: Tuple[int, int]) -> np.ndarray:
        """
        압축된 바이너리 마스크 데이터를 NumPy 배열로 복원합니다.
        state_repository.py의 _compress_mask와 정확히 반대 동작을 합니다.
        """
        try:
            # 바이너리 데이터를 uint8 배열로 변환
            packed_array = np.frombuffer(compressed_mask, dtype=np.uint8)
            # 비트 단위로 풀기
            unpacked_array = np.unpackbits(packed_array)
            # 원본 이미지 크기에 맞게 자르기
            num_elements = shape[0] * shape[1]
            unpacked_array = unpacked_array[:num_elements]
            # 2D 형태로 재구성
            mask = unpacked_array.reshape(shape)
            # 시각화를 위해 0과 255 값으로 변환
            return (mask * 255).astype(np.uint8)
        except Exception as e:
            print(f"Error decompressing mask: {e}")
            # 오류 발생 시 빈 마스크 반환
            return np.zeros(shape, dtype=np.uint8)

    def get_batch_data(self, batch_id: str) -> List[FrameData]:
        """
        주어진 batch_id에 해당하는 모든 카메라의 데이터를 Redis에서 가져와
        FrameData 객체 리스트로 반환합니다.
        """
        batch_set_key = f"batch:{batch_id}"
        result_keys = self.redis_client.smembers(batch_set_key)
        
        if not result_keys:
            print(f"경고: Batch ID '{batch_id}'에 해당하는 결과 키를 찾을 수 없습니다.")
            return []

        all_frames_data: List[FrameData] = []
        for key_bytes in result_keys:
            key = key_bytes.decode('utf-8')
            
            # 1. 메인 데이터(Hash) 가져오기
            raw_data = self.redis_client.hgetall(key)
            if not raw_data:
                print(f"경고: 키 '{key}'에 대한 데이터를 찾을 수 없습니다.")
                continue

            # 바이트를 문자열로 디코딩
            data = {k.decode('utf-8'): v.decode('utf-8') for k, v in raw_data.items()}
            
            # 2. 추론 데이터(JSON) 파싱
            inference_data_str = data.get("inference_data", "{}")
            try:
                inference_data = json.loads(inference_data_str)
            except json.JSONDecodeError:
                print(f"경고: 키 '{key}'의 inference_data 파싱 실패.")
                inference_data = {}

            # 3. 바운더리 마스크 데이터 가져오기 및 복원
            reconstructed_masks = []
            if 'boundary_masks' in inference_data and isinstance(inference_data['boundary_masks'], list):
                for mask_info in inference_data['boundary_masks']:
                    mask_key = mask_info.get('segmentation_mask')
                    if not mask_key:
                        continue
                    
                    compressed_mask = self.redis_client.get(mask_key)
                    if compressed_mask:
                        decompressed_mask = self._decompress_mask(compressed_mask, self.image_shape)
                        reconstructed_masks.append(decompressed_mask)

            # 4. Pinjig 마스크 데이터 가져오기 및 복원
            reconstructed_pinjig_masks = []
            filtered_pinjig_classifications = []
            
            if 'pinjig_masks' in inference_data and isinstance(inference_data['pinjig_masks'], list):
                pinjig_masks = inference_data.get('pinjig_masks', [])
                pinjig_classifications = inference_data.get('pinjig_classifications', [])
                
                # pinjig_classifications의 top1_class가 target_classes에 포함되는 것만 필터링
                for classification in pinjig_classifications:
                    if classification.get('top1_class') in self.pinjig_target_classes:
                        mask_index = classification.get('mask_index')
                        
                        # 해당 인덱스의 마스크 찾기
                        if mask_index is not None and mask_index < len(pinjig_masks):
                            mask_info = pinjig_masks[mask_index]
                            mask_key = mask_info.get('segmentation')
                            
                            if mask_key:
                                compressed_mask = self.redis_client.get(mask_key)
                                if compressed_mask:
                                    decompressed_mask = self._decompress_mask(compressed_mask, self.image_shape)
                                    reconstructed_pinjig_masks.append(decompressed_mask)
                                    filtered_pinjig_classifications.append(classification)

            # 5. Assembly classifications 가져오기 (필터링 없이 모든 요소 포함)
            detections = inference_data.get("detections", [])
            assembly_classifications = inference_data.get("assembly_classifications", [])

            # 6. FrameData 객체 생성
            frame = FrameData(
                image_id=data.get("image_id", ""),
                camera_name=data.get("camera_name", ""),
                image_path=data.get("image_path", ""),
                captured_at=data.get("captured_at", ""),
                image_shape=self.image_shape,
                detections=detections,
                boundary_masks=reconstructed_masks,
                pinjig_masks=reconstructed_pinjig_masks,
                pinjig_classifications=filtered_pinjig_classifications,
                assembly_classifications=assembly_classifications  # 모든 assembly classifications 포함
            )
            all_frames_data.append(frame)

        print(f"✅ Batch ID '{batch_id}'에서 {len(all_frames_data)}개 카메라 데이터를 성공적으로 수집했습니다.")
        
        # Assembly classifications 통계 출력
        total_assembly_classifications = sum(len(frame.assembly_classifications) for frame in all_frames_data)
        if total_assembly_classifications > 0:
            print(f"   - 총 {total_assembly_classifications}개의 assembly classifications")
        
        return all_frames_data