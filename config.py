"""
프로젝트 내 설정 파일입니다.
이 파일은 이미지 데이터 디렉토리, 출력 디렉토리, 카메라 ID, Redis 서버 정보 등을 포함합니다.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DATA_DIR = BASE_DIR / "data" / "mipo"
OUTPUT_DIR = BASE_DIR / "output"

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

# CAMERA_INFO = [
#     {"bay": "3bay_north", "name": "C02", "host": "10.150.160.183", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "C04", "host": "10.150.160.184", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "C06", "host": "10.150.160.193", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "C07", "host": "10.150.160.194", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "C09", "host": "10.150.160.209", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "C11", "host": "10.150.160.211", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "D02", "host": "10.150.160.221", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "D04", "host": "10.150.160.222", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "D06", "host": "10.150.160.224", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "D07", "host": "10.150.160.225", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "D09", "host": "10.150.160.226", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     {"bay": "3bay_north", "name": "D11", "host": "10.150.160.227", "port": 80, "username": "admin", "password": "Hmd!!2503700"},
#     ]

NUM_LATEST_IMAGES = 12

# Redis 서버 정보
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Redis Producer와 Consumer가 통신할 Redis 채널
REDIS_CHANNEL = "cctv_state_updates"