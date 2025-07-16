import redis
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple

from config import REDIS_HOST, REDIS_PORT, REDIS_DB
from cctv_event_detector.core.models import CapturedImage

# JSON이 Numpy 데이터 타입을 처리할 수 있도록 돕는 변환기 클래스입니다.
class NumpyEncoder(json.JSONEncoder):
    """ Numpy 타입을 파이썬 기본 타입으로 변환하여 JSON으로 직렬화할 수 있도록 돕습니다. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class StateRepository:
    """
    CCTV 이벤트 데이터의 상태를 Redis에 저장하고 관리하는 책임을 가집니다.
    """
    def __init__(self):
        """
        StateRepository를 초기화하고 Redis 서버에 연결합니다.
        """
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=False  # 중요: 바이너리 데이터(마스크) 저장을 위해 False로 설정
            )
            self.redis_client.ping()
            print(f"✅ Redis client connected successfully to {REDIS_HOST}:{REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise

    def _compress_mask(self, mask: np.ndarray) -> bytes:
        """
        불리언(boolean) 타입의 Numpy 배열 마스크를 압축된 바이트 문자열로 변환합니다.
        np.packbits는 8개의 불리언 값을 1바이트로 압축하여 저장 효율을 극대화합니다.
        """
        return np.packbits(mask).tobytes()

    def _prepare_data_for_storage(
        self,
        inference_result: Dict[str, Any],
        batch_id: str,
        image_id: str
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        """
        추론 결과에서 큰 마스크 데이터를 분리하고, 저장할 데이터들을 준비합니다.
        
        :return: (메인 데이터로 저장될 딕셔너리, 별도로 저장될 마스크 (Key, 데이터) 리스트)
        """
        # 원본 수정을 방지하기 위해 깊은 복사는 아니지만, 최상위 레벨을 복사합니다.
        # 내부 리스트나 딕셔너리는 수정되므로 주의가 필요하지만, 여기서는 괜찮습니다.
        result_copy = dict(inference_result)
        masks_to_store = []

        # 처리할 마스크 타입과 해당 키 이름을 정의합니다.
        mask_types_to_process = {
            "boundary_masks": ("boundary", "segmentation_mask"),
            "pinjig_masks": ("pinjig", "segmentation"),
        }

        for key_in_result, (type_name, mask_field) in mask_types_to_process.items():
            if key_in_result in result_copy:
                # 리스트를 복사하여 순회 중 수정 문제를 방지합니다.
                original_masks = list(result_copy[key_in_result])
                for i, mask_data in enumerate(original_masks):
                    if mask_field in mask_data and isinstance(mask_data[mask_field], np.ndarray):
                        # 1. 마스크 데이터 분리 및 압축
                        mask_array = mask_data.pop(mask_field)
                        compressed_mask = self._compress_mask(mask_array)
                        
                        # 2. 마스크를 저장할 고유 키 생성
                        mask_key = f"mask:{batch_id}:{image_id}:{type_name}:{i}"
                        masks_to_store.append((mask_key, compressed_mask))
                        
                        # 3. 원본 데이터 위치에 마스크 키(주소)를 삽입
                        mask_data[mask_field] = mask_key

        return result_copy, masks_to_store

    def save_batch_results(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]):
        """
        한 번의 캡처/추론 사이클(배치)에서 나온 모든 결과를 Redis에 원자적으로 저장합니다.
        Redis의 pipeline을 사용하여 여러 명령을 한 번의 네트워크 요청으로 처리하여 성능을 최적화합니다.
        """
        if not captured_images:
            return

        batch_id = time.strftime('%Y%m%d-%H%M%S')
        batch_set_key = f"batch:{batch_id}"
        batch_result_keys = []

        print(f"\n--- Storing results to Redis with Batch ID: {batch_id} ---")

        try:
            # 파이프라인 시작
            with self.redis_client.pipeline() as pipe:
                for capture in captured_images:
                    image_id = capture.image_id
                    if image_id not in inference_results:
                        continue

                    # 1. 데이터 준비 (마스크 분리 및 주소 대체)
                    prepared_inference_data, masks_to_store = self._prepare_data_for_storage(
                        inference_results[image_id], batch_id, image_id
                    )

                    # 2. 메인 데이터 Hash 저장 준비
                    main_key = f"result:{batch_id}:{image_id}"
                    batch_result_keys.append(main_key)
                    
                    main_data_payload = {
                        "image_id": image_id,
                        "camera_name": capture.camera_name,
                        "captured_at": capture.captured_at.isoformat(),
                        "image_path": f"./data/{image_id}.jpg", # 원본 이미지 저장 경로
                        "inference_data": json.dumps(prepared_inference_data, cls=NumpyEncoder)
                    }
                    pipe.hset(main_key, mapping=main_data_payload)

                    # 3. 분리된 마스크 데이터 저장 준비
                    for mask_key, compressed_mask in masks_to_store:
                        pipe.set(mask_key, compressed_mask)

                # 4. 배치 그룹(Set)에 모든 메인 키 저장 준비
                if batch_result_keys:
                    pipe.sadd(batch_set_key, *batch_result_keys)

                # 5. 모든 명령을 한 번에 실행
                results = pipe.execute()
                print(f"✅ Successfully stored {len(captured_images)} results and {sum(len(m[1]) for _, m in enumerate(masks_to_store, 1))} masks.")
                print(f"   - Batch Set Key: {batch_set_key}")

        except redis.exceptions.RedisError as e:
            print(f"❌ An error occurred while saving data to Redis: {e}")