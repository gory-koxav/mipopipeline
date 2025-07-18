# cctv_event_detector/situation_awareness/facade.py

import redis
import json
import time

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_IMAGE_CHANNEL
from .data_aggregator import DataAggregator
from .projector import Projector
from .visualizer import Visualizer

class SituationAwarenessFacade:
    """
    상황 인식 시스템의 전체 워크플로우를 관리하고 조정하는 퍼사드 클래스.
    Redis 구독, 데이터 집계, 사영 변환, 시각화의 흐름을 제어합니다.
    """
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB
            )
            self.redis_client.ping()
            print("✅ Redis 클라이언트가 성공적으로 연결되었습니다.")
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Redis 연결 실패: {e}")
            raise
            
        self.pubsub = self.redis_client.pubsub()
        self.aggregator = DataAggregator(self.redis_client)
        self.projector = Projector()

    def _process_batch(self, batch_id: str):
        """단일 배치 ID에 대한 전체 처리 로직을 수행합니다."""
        print(f"\n🚀 Batch ID '{batch_id}' 처리를 시작합니다...")
        start_time = time.time()
        
        # 1. Redis에서 데이터 집계
        all_frames_data = self.aggregator.get_batch_data(batch_id)
        if not all_frames_data:
            print("처리할 데이터가 없어 작업을 중단합니다.")
            return

        # 2. 각 카메라 데이터에 대해 사영 변환 수행
        projected_results = []
        for frame_data in all_frames_data:
            projected_data = self.projector.project(frame_data)
            projected_results.append(projected_data)
        
        print(f"✅ {len(projected_results)}개 카메라 데이터의 사영 변환을 완료했습니다.")

        # 3. 투영된 결과들을 시각화하고 파일로 저장
        visualizer = Visualizer(batch_id)
        visualizer.draw(projected_results)
        visualizer.save_and_close()
        
        end_time = time.time()
        print(f"🏁 Batch ID '{batch_id}' 처리를 완료했습니다. (총 소요 시간: {end_time - start_time:.2f}초)")

    def start_listening(self):
        """
        Redis 채널을 구독하고, 메시지를 기다리는 메인 루프를 시작합니다.
        """
        self.pubsub.subscribe(REDIS_IMAGE_CHANNEL)
        print(f"\n🎧 '{REDIS_IMAGE_CHANNEL}' 채널을 구독합니다. 메시지를 기다립니다...")
        
        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    print(f"\n📬 새로운 메시지 수신: {message['data']}")
                    try:
                        payload = json.loads(message['data'])
                        batch_id = payload.get('batch_id')
                        if batch_id:
                            self._process_batch(batch_id)
                        else:
                            print("경고: 메시지에 'batch_id'가 없습니다.")
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"오류: 메시지 파싱 실패 - {e}")
                    finally:
                        print(f"\n🎧 다시 메시지를 기다립니다...")

        except KeyboardInterrupt:
            print("\n👋 사용자에 의해 리스너가 종료되었습니다.")
        except Exception as e:
            print(f"\n❌ 리스닝 중 심각한 오류 발생: {e}")
        finally:
            self.pubsub.close()
            print("Redis PubSub 연결이 종료되었습니다.")