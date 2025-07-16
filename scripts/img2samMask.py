import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import glob

# ==============================================================================
# 1. 변수 및 경로 설정
# ==============================================================================

# 기본 경로
HOME_PATH = '/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/mipo/timelapse_250224-250321'
PROJECT_BASE_DIR = '/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv'

# SAM (Segment Anything Model) 모델 설정
SAM_CHECKPOINT_PATH = os.path.join(PROJECT_BASE_DIR, "checkpoints", "segmentation_sam", "sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 처리할 카메라 폴더 이름 목록
CAMERA_NAMES = ['C_2', 'C_4', 'C_6', 'C_7', 'C_9', 'C_11', 'D_2', 'D_4', 'D_6', 'D_7', 'D_9', 'D_11']


# ==============================================================================
# 2. 기능별 함수 정의
# ==============================================================================

def preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    SAM 모델에 이미지를 입력하기 전에 전처리를 수행합니다.
    Bilateral 필터를 적용하여 노이즈를 줄이면서 객체의 경계선은 보존합니다.
    """
    return cv2.bilateralFilter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75)

def filter_masks(masks: list, image_shape: tuple) -> list:
    """
    (수정된 로직) SAM이 생성한 여러 마스크들을 참고 코드와 동일한 기준으로 필터링합니다.
    1. 중첩 필터링: 면적이 큰 마스크 내부에 대부분 포함되는 작은 마스크를 제거합니다.
    2. 크기 필터링: 바운딩 박스의 가로/세로가 최소 100픽셀 이상인 마스크만 남깁니다.

    Args:
        masks (list): SamAutomaticMaskGenerator가 생성한 마스크 목록.
        image_shape (tuple): 원본 이미지의 형태 (참고용으로 남겨둠).

    Returns:
        list: 모든 필터링을 통과한 최종 마스크 목록.
    """
    if not masks:
        return []

    # 1. 중첩 필터링
    # 마스크를 면적(area) 기준으로 내림차순(큰 것부터) 정렬합니다.
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    # 유지할 마스크의 인덱스를 관리하는 리스트. 처음에는 모든 마스크를 포함합니다.
    kept_indices = list(range(len(sorted_masks)))
    nesting_threshold = 0.9

    for i in range(len(sorted_masks)):
        if i not in kept_indices:
            continue

        mask_i_seg = sorted_masks[i]['segmentation']
        # i번째 마스크와 그보다 작은 j번째 마스크들을 비교합니다.
        for j in range(i + 1, len(sorted_masks)):
            if j not in kept_indices:
                continue

            mask_j_seg = sorted_masks[j]['segmentation']

            # 두 마스크 간의 교집합(intersection) 면적을 계산합니다.
            intersection = np.logical_and(mask_i_seg, mask_j_seg).sum()
            
            # 교집합 면적이 작은 마스크(j) 면적의 90% 이상이면, 작은 마스크는 큰 마스크에 포함된 것으로 간주합니다.
            if intersection / sorted_masks[j]['area'] > nesting_threshold:
                # 포함된 것으로 간주된 작은 마스크(j)를 최종 목록에서 제거합니다.
                kept_indices.remove(j)

    # 2. 크기 필터링
    # 중첩 필터링에서 살아남은 마스크들을 대상으로 최소 크기 검사를 수행합니다.
    final_masks = []
    min_width = 100
    min_height = 100

    for i in kept_indices:
        mask = sorted_masks[i]
        bbox_width = mask['bbox'][2]   # 바운딩 박스의 너비
        bbox_height = mask['bbox'][3]  # 바운딩 박스의 높이

        # 너비와 높이가 모두 100픽셀 이상인 경우에만 최종 목록에 추가합니다.
        if bbox_width >= min_width and bbox_height >= min_height:
            final_masks.append(mask)
            
    return final_masks


def crop_and_save_mask(original_image: np.ndarray, mask_ann: dict, output_path: str):
    """
    마스크 정보를 이용해 원본 이미지에서 객체 영역을 잘라내고,
    배경을 회색으로 채워서 지정된 경로에 저장합니다.
    """
    try:
        x, y, w, h = map(int, mask_ann['bbox'])

        if w <= 0 or h <= 0:
            return

        rect_crop = original_image[y:y+h, x:x+w]
        segmentation_mask_cropped = mask_ann['segmentation'][y:y+h, x:x+w]
        
        gray_background = np.full_like(rect_crop, (114, 114, 114), dtype=np.uint8)
        gray_background[segmentation_mask_cropped] = rect_crop[segmentation_mask_cropped]
        
        cv2.imwrite(output_path, gray_background)

    except Exception as e:
        print(f"\n[오류] 마스크 처리 중 예외 발생: {e}, 파일: {output_path}")


# ==============================================================================
# 3. 메인 실행 로직
# ==============================================================================

def main():
    """
    전체 프로세스를 실행하는 메인 함수
    """
    print("--- 마스크 추출 스크립트 시작 (필터링 로직 수정됨) ---")
    print(f"사용 장치(DEVICE): {DEVICE}")

    # SAM 모델 로드
    print(f"SAM 모델 로딩 중... 경로: {SAM_CHECKPOINT_PATH}")
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"[오류] SAM 체크포인트 파일을 찾을 수 없습니다. 경로를 확인하세요: {SAM_CHECKPOINT_PATH}")
        return
        
    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.92,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        print("✅ SAM 모델 로드 완료.")
    except Exception as e:
        print(f"[오류] SAM 모델 로드에 실패했습니다: {e}")
        return

    # 각 카메라 폴더 순회
    for camera_name in CAMERA_NAMES:
        image_folder = os.path.join(HOME_PATH, camera_name)
        output_folder = os.path.join(HOME_PATH, f"{camera_name}_masks")
        
        print(f"\n▶️ '{camera_name}' 폴더 처리 시작...")
        
        if not os.path.isdir(image_folder):
            print(f"  - [경고] 이미지 폴더를 찾을 수 없습니다: '{image_folder}'. 다음으로 넘어갑니다.")
            continue
            
        os.makedirs(output_folder, exist_ok=True)
        print(f"  - 결과 저장 위치: {output_folder}")
        
        image_paths = glob.glob(os.path.join(image_folder, '*.jpg')) + \
                      glob.glob(os.path.join(image_folder, '*.png')) + \
                      glob.glob(os.path.join(image_folder, '*.jpeg'))
                      
        if not image_paths:
            print("  - [경고] 처리할 이미지가 없습니다.")
            continue
        
        for image_path in tqdm(image_paths, desc=f"  - '{camera_name}' 이미지 처리 중", leave=False, unit="개"):
            try:
                image_bgr = cv2.imread(image_path)
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                # 전처리 -> 마스크 생성 -> 수정된 로직으로 필터링
                preprocessed_image = preprocess_image(image_rgb)
                masks = mask_generator.generate(preprocessed_image)
                filtered_masks = filter_masks(masks, image_rgb.shape)
                
                original_filename, ext = os.path.splitext(os.path.basename(image_path))
                for i, mask_ann in enumerate(filtered_masks):
                    output_filename = f"{original_filename}_mask_{i:03d}{ext}"
                    output_path = os.path.join(output_folder, output_filename)
                    crop_and_save_mask(image_bgr, mask_ann, output_path)

            except Exception as e:
                print(f"\n[오류] 이미지 처리 중 예외 발생: {e}, 파일: {image_path}")
                
        tqdm.write(f"✅ '{camera_name}' 폴더 처리 완료.")

    print("\n--- 모든 작업이 성공적으로 완료되었습니다. ---")

if __name__ == "__main__":
    main()