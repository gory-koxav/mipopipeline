"""
프로젝트 내 설정 파일입니다.
이 파일은 이미지 데이터 디렉토리, 출력 디렉토리, 모델 경로 등 주요 설정을 포함합니다.
"""
from pathlib import Path

# --- 기본 경로 설정 ---
# 이 파일(config.py)이 위치한 디렉토리를 기준으로 프로젝트의 기본 경로를 설정합니다.
BASE_DIR = Path(__file__).resolve().parent

# --- 데이터 경로 ---
# 원본 이미지가 저장된 디렉토리입니다.
IMAGE_DATA_DIR = BASE_DIR / "data" / "mipo"


# --- SAM (Segment Anything Model) 관련 설정 ---
# SAM을 통해 분할 및 분류된 결과(크롭 이미지, 오버레이)가 저장될 부모 디렉토리입니다.
# SAM_CLASSIFICATION_OUTPUT_DIR = BASE_DIR / "output" / "sam_classification_results"
SAM_CLASSIFICATION_OUTPUT_DIR = BASE_DIR / "output"

# 학습된 SAM 모델의 체크포인트 파일 경로입니다.
SAM_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "segmentation_sam" / "sam_vit_h_4b8939.pth"

# 사용할 SAM 모델의 타입입니다. (예: "vit_h", "vit_l", "vit_b")
SAM_MODEL_TYPE = "vit_h"

# SAM으로 분할된 이미지를 분류하는 데 사용할 YOLO 분류 모델의 경로입니다.
SAM_YOLO_CLS_MODEL_PATH = BASE_DIR / "checkpoints" / "segmentation_yolo_cls" / "best.pt"


# --- YOLO Object Detection 모델 설정 ---
# 객체 탐지를 위한 YOLO 모델의 경로입니다.
YOLO_OD_MODEL_PATH = BASE_DIR / "checkpoints" / "objectdetection_yolo" / "best.pt"

# --- SAM Object Boundary Finder 설정 ---
# 경계선 탐지를 수행할 객체 클래스 이름 목록 # 빈 리스트면 모든 클래스에 대해 경계선 탐지
BOUNDARY_TARGET_CLASSES = ["panel", "block", "HP_block"]
# BOUNDARY_TARGET_CLASSES = []

# 경계선 탐지 시각화 결과가 저장될 디렉토리
# SAM_BOUNDARY_OUTPUT_DIR = BASE_DIR / "output" / "sam_boundary_results"
SAM_BOUNDARY_OUTPUT_DIR = BASE_DIR / "output"

# 카메라 정보 리스트
CAMERA_INFO = [
    {"bay": "3bay_north", "name": "C02", "host": "10.150.160.183", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "C04", "host": "10.150.160.184", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "C06", "host": "10.150.160.193", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "C07", "host": "10.150.160.194", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "C09", "host": "10.150.160.209", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "C11", "host": "10.150.160.211", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "D02", "host": "10.150.160.221", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "D04", "host": "10.150.160.222", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "D06", "host": "10.150.160.224", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "D07", "host": "10.150.160.225", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "D09", "host": "10.150.160.226", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    {"bay": "3bay_north", "name": "D11", "host": "10.150.160.227", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
    ]

NUM_LATEST_IMAGES = 12

# Redis 서버 정보
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Redis Producer와 Consumer가 통신할 Redis 채널
REDIS_CHANNEL = "cctv_state_updates"