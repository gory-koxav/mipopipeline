import cv2
import numpy as np

# 원본 이미지 파일 경로 (실제 파일 경로로 변경하세요)
image_path = '/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/mipo/timelapse_250224-250321_pinjig/D_6/00000583.png'

# 이미지 읽기
original_image = cv2.imread(image_path)

# 이미지가 제대로 로드되었는지 확인
if original_image is None:
    print(f"'{image_path}' 파일을 찾을 수 없거나 열 수 없습니다.")
else:
    # 이미지 높이 가져오기
    height = original_image.shape[:2][0]

    # 상단 20% 영역의 y좌표 범위 계산
    top_20_percent_height = int(height * 0.2)
    top_region = original_image[:top_20_percent_height, :]

    # 하단 20% 영역의 y좌표 범위 계산 (80%부터)
    bottom_80_percent_start = int(height * 0.8)
    bottom_region = original_image[-int(height * 0.2):, :] # 또는 original_image:bottom_80_percent_start:, :

    # 회색 컬러 값 (BGR 순서)
    gray_color = [114, 114, 114]

    # 상단 20% 영역을 회색으로 칠하기
    original_image[:top_20_percent_height, :] = gray_color

    # 하단 20% 영역을 회색으로 칠하기
    original_image[-int(height * 0.2):, :] = gray_color # 또는 original_image:bottom_80_percent_start:, :] = gray_color

    # 수정된 이미지 보여주기
    cv2.imshow('Modified Image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()