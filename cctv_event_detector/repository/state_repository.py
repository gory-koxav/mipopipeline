# cctv_event_detector/repository/state_repository.py

import redis
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import atexit  # [핵심 추가] 애플리케이션 종료 시 실행할 함수를 등록하기 위해 임포트합니다.

from config import IMAGE_DATA_DIR, REDIS_HOST, REDIS_PORT, REDIS_DB
from cctv_event_detector.core.models import CapturedImage

# Redis 연결 풀을 생성합니다.
pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=False,
    health_check_interval=30
)

# [핵심 추가] 애플리케이션 종료 시 Redis 연결 풀을 닫는 함수입니다.
def disconnect_redis_pool():
    """애플리케이션 종료 시 Redis 연결 풀의 모든 연결을 종료합니다."""
    print("👋 Shutting down... Disconnecting all Redis connections from the pool.")
    pool.disconnect()

# [핵심 추가] 애플리케이션이 정상적으로 종료될 때 disconnect_redis_pool 함수가 자동으로 호출되도록 등록합니다.
atexit.register(disconnect_redis_pool)


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
            # 공유된 연결 풀을 사용하고, 시간 초과를 설정합니다.
            self.redis_client = redis.Redis(
                connection_pool=pool,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            print(f"✅ Redis client connected successfully using connection pool to {REDIS_HOST}:{REDIS_PORT}")
        except redis.exceptions.TimeoutError as e:
            print(f"❌ Redis connection timed out: {e}")
            raise
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise

    def _compress_mask(self, mask: np.ndarray) -> bytes:
        return np.packbits(mask).tobytes()

    def _prepare_data_for_storage(
        self,
        inference_result: Dict[str, Any],
        batch_id: str,
        image_id: str
    ) -> Tuple[Dict[str, Any], List[Tuple[str, bytes]]]:
        result_copy = dict(inference_result)
        masks_to_store = []
        mask_types_to_process = {
            "boundary_masks": ("boundary", "segmentation_mask"),
            "pinjig_masks": ("pinjig", "segmentation"),
        }
        for key_in_result, (type_name, mask_field) in mask_types_to_process.items():
            if key_in_result in result_copy:
                original_masks = list(result_copy[key_in_result])
                for i, mask_data in enumerate(original_masks):
                    if mask_field in mask_data and isinstance(mask_data[mask_field], np.ndarray):
                        mask_array = mask_data.pop(mask_field)
                        compressed_mask = self._compress_mask(mask_array)
                        mask_key = f"mask:{batch_id}:{image_id}:{type_name}:{i}"
                        masks_to_store.append((mask_key, compressed_mask))
                        mask_data[mask_field] = mask_key
        return result_copy, masks_to_store

    def _get_inference_summary(self, result: Dict[str, Any]) -> str:
        summary_lines = []
        for key, value in result.items():
            if isinstance(value, list):
                summary_lines.append(f"      - {key}: {len(value)} items")
            elif isinstance(value, dict):
                summary_lines.append(f"      - {key}: {len(value.keys())} keys")
            else:
                summary_lines.append(f"      - {key}: (type: {type(value).__name__})")
        return "\n".join(summary_lines)
        
    def save_batch_results(
        self, 
        captured_images: List[CapturedImage], 
        inference_results: Dict[str, Any], 
        redis_channel: str
    ):
        if not captured_images:
            print("⚠️ 경고: 저장할 captured_images 데이터가 없습니다.")
            return

        batch_id = time.strftime('%Y%m%d-%H%M%S')
        batch_set_key = f"batch:{batch_id}"
        batch_result_keys = []
        processed_count = 0

        print(f"\n--- 🕵️  Redis 저장 전 데이터 상세 디버깅 시작 (Batch ID: {batch_id}) ---")
        print(f"   - 총 {len(captured_images)}개의 캡처 데이터와 {len(inference_results)}개의 추론 결과를 받았습니다.\n")

        try:
            with self.redis_client.pipeline() as pipe:
                for i, capture in enumerate(captured_images):
                    image_id = capture.image_id
                    
                    print(f"-[{i+1:02d}/{len(captured_images)}] 📄 ID: {image_id}")
                    
                    if hasattr(capture, 'image_data') and capture.image_data is not None:
                        img_size_mb = capture.image_data.nbytes / (1024 * 1024)
                        print(f"    [CAPTURE] Status: ✅ Image Present | Size: {img_size_mb:.2f} MB | Time: {capture.captured_at}")
                    else:
                        print(f"    [CAPTURE] Status: ❌ Image Missing")
                    
                    inference_result = inference_results.get(image_id)
                    if inference_result:
                        print(f"    [INFERENCE] Status: ✅ Found")
                        summary = self._get_inference_summary(inference_result)
                        print(summary)
                    else:
                        print(f"    [INFERENCE] Status: ❌ Not Found")
                    print("-" * 60)

                    main_key = f"result:{batch_id}:{image_id}"
                    batch_result_keys.append(main_key)
                    
                    main_data_payload = {
                        "image_id": image_id,
                        "camera_name": capture.camera_name,
                        "captured_at": capture.captured_at.isoformat(),
                        "image_path": f"{IMAGE_DATA_DIR}/{capture.camera_name}/{image_id.split('_')[-1]}.png",
                    }

                    if inference_result:
                        prepared_data, masks_to_store = self._prepare_data_for_storage(
                            inference_result, batch_id, image_id
                        )
                        main_data_payload["inference_data"] = json.dumps(prepared_data, cls=NumpyEncoder)
                        
                        for mask_key, compressed_mask in masks_to_store:
                            pipe.set(mask_key, compressed_mask)
                    else:
                        error_payload = {
                            "status": "error",
                            "message": "AI inference result not found or failed."
                        }
                        main_data_payload["inference_data"] = json.dumps(error_payload)

                    pipe.hset(main_key, mapping=main_data_payload)
                    processed_count += 1
                
                print("\n--- ✅ 디버깅 종료. Redis 파이프라인 실행 ---")

                if batch_result_keys:
                    pipe.sadd(batch_set_key, *batch_result_keys)
                    
                message_payload = json.dumps({"batch_id": batch_id, "status": "completed", "processed_count": processed_count})
                pipe.publish(redis_channel, message_payload)
                
                pipe.execute()
                
                print(f"✅ {processed_count}개 카메라 상태 저장 완료. '{redis_channel}' 채널에 알림 전송.")
                print(f"   - Batch Set Key: {batch_set_key}")

        except redis.exceptions.RedisError as e:
            print(f"❌ Redis에 데이터를 저장하는 중 오류 발생: {e}")
        except Exception as e:
            print(f"❌ 저장 로직 중 예상치 못한 오류 발생: {e}")