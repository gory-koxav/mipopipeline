# situation_awareness_main.py

from cctv_event_detector.situation_awareness.facade import SituationAwarenessFacade

def main():
    """
    상황 인식 및 융합 시스템의 메인 실행 함수입니다.
    """
    print("===== 👁️  상황 인식 융합 시스템 (Subscriber) 시작 =====")
    
    try:
        # 퍼사드 객체를 생성하고 리스닝을 시작합니다.
        facade = SituationAwarenessFacade()
        facade.start_listening()
    except Exception as e:
        print(f"프로그램을 시작하지 못했습니다: {e}")
    finally:
        print("===== 👁️  상황 인식 융합 시스템 종료 =====")

if __name__ == "__main__":
    main()