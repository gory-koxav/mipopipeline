# situation_awareness_main.py

from cctv_event_detector.situation_awareness.facade_test import SituationAwarenessFacade

def main():
    """
    상황 인식 및 융합 시스템의 메인 실행 함수입니다. (배치 테스트 버전)
    지정된 배치 ID에 대한 처리를 수행합니다.
    """
    print("===== 👁️  상황 인식 융합 시스템 (배치 테스트) 시작 =====")
    
    # =======================================================
    # ✅ 테스트할 배치 ID를 여기에 지정하세요.
    # 예: "batch_20231026_153000"
    TARGET_BATCH_ID = "20250805-173454"
    # =======================================================

    try:
        # 퍼사드 객체를 생성합니다.
        facade = SituationAwarenessFacade()
        
        # 지정된 배치 ID로 처리를 실행합니다.
        facade.process_batch(TARGET_BATCH_ID)
        
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류가 발생했습니다: {e}")
    finally:
        print("\n===== 👁️  상황 인식 융합 시스템 (배치 테스트) 종료 =====")

if __name__ == "__main__":
    main()