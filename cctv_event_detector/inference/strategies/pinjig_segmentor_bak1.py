import os
import sys

# --- 패키지 루트를 sys.path에 추가하는 로직 ---
# 현재 파일의 절대 경로
current_file_path = os.path.abspath(__file__)
# 'cctv_event_detector' 패키지의 루트 디렉토리를 찾음
# project_root = os.path.join(os.path.dirname(current_file_path), '..', '..', '..') # 3단계 위로

# A more robust way: find the specific marker for your project root
# assuming 'cctv_event_detector' is directly inside your project root
temp_path = os.path.dirname(current_file_path) # strategies
temp_path = os.path.dirname(temp_path) # inference
temp_path = os.path.dirname(temp_path) # cctv_event_detector
project_root = os.path.dirname(temp_path) # L2025022_mipo_operationsystem_uv (the directory containing cctv_event_detector)

project_root = os.path.normpath(project_root)

# project_root가 sys.path에 없으면 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------
# cctv_event_detector/inference/strategies/pinjig_segmenter.py
import torch
import numpy as np
from typing import Dict, Any, List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage # 추가

class SamPinjigSegmenter(InferenceStrategy):
    """
    Segment Anything Model (SAM)을 사용하여 'everything' 모드로 이미지 분할을 수행하는 전략 클래스입니다.
    """
    def __init__(self, model_type: str = "vit_h", checkpoint: str = "sam_vit_h_4b8939.pth"):
        """
        SAM 모델을 초기화하고 메모리에 로드합니다.

        Args:
            model_type (str): 사용할 SAM 모델의 종류 (예: 'vit_h', 'vit_l', 'vit_b').
            checkpoint (str): 다운로드한 SAM 모델 체크포인트 파일의 경로.
        """
        print("Initializing SAM model...")
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 지정된 타입의 SAM 모델을 레지스트리에서 찾아 체크포인트를 로드합니다.
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)

        # 'everything' 모드를 위한 SamAutomaticMaskGenerator 객체를 생성합니다.
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM model loaded successfully.")

    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력된 각 이미지에 대해 'everything' 모드 분할을 실행하고 결과를 시각화합니다.
        """
        print("Running SAM Pinjig Segmenter...")

        for capture in captured_images:
            image_data = capture.image_data
            
            # cv2는 BGR, SAM은 RGB 형식의 이미지를 사용하므로 변환이 필요합니다.
            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            # rgb_image = cv2.bilateralFilter(rgb_image, 64, 10, 10) # 이미지 전처리 (선택적) (바이레터럴 블러)
            rgb_image = cv2.GaussianBlur(rgb_image, (31, 31), 0) # 이미지 전처리 (선택적) (가우시안 블러)

            # 마스크 생성 실행
            print(f"Generating masks for {capture.image_id}...")
            masks = self.mask_generator.generate(rgb_image)
            print(f"Found {len(masks)} masks.")

            # 결과 시각화
            visualized_image = self._draw_masks(image_data.copy(), masks) # 원본 이미지를 복사하여 마스크를 그립니다.
            # visualized_image = self._draw_masks(rgb_image, masks) # 전처리된 이미지를 복사하여 마스크를 그립니다.
            cv2.imshow(f"SAM Segmentation - {capture.image_id}", visualized_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        return inference_results

    def _draw_masks(self, image: np.ndarray, masks: List[Dict[str, Any]]) -> np.ndarray:
        """
        생성된 마스크들을 원본 이미지 위에 색상으로 그려주는 헬퍼 함수입니다.
        """
        if len(masks) == 0:
            return image
        
        # 마스크들을 영역(area) 크기 순으로 정렬합니다.
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

        # 각 마스크 위에 랜덤 색상으로 칠할 오버레이 이미지를 생성합니다.
        overlay = image.copy()
        for ann in sorted_masks:
            m = ann['segmentation']
            # 랜덤 색상 생성 (B, G, R)
            color = np.random.randint(0, 256, (1, 3)).tolist()[0]
            # 마스크 영역(m=True)에 해당 색상을 칠합니다.
            overlay[m] = color
        
        # 원본 이미지와 마스크 오버레이를 50%씩 혼합하여 반투명 효과를 줍니다.
        # cv2.addWeighted(src1, alpha, src2, beta, gamma)
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)





if __name__ == "__main__":
    import cv2
    import datetime
    # --- 실행 전 설정 ---
    # 1. 테스트할 이미지 파일 경로를 지정하세요.
    image_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/D04_20250703165331.jpg"
    
    # 2. 다운로드한 SAM 모델 체크포인트 파일의 경로를 지정하세요.
    checkpoint_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/checkpoints/segmentation/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # --- 테스트 코드 ---
    print("Loading test image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        # 실제 Segmenter 객체 생성
        segmenter = SamPinjigSegmenter(model_type=model_type, checkpoint=checkpoint_path)

        # 테스트용 이미지 객체 생성
        test_images = [
            CapturedImage(
                image_id="test_img_01", 
                camera_name="test_cam", 
                image_data=image, 
                captured_at=datetime.datetime.now()
            )
        ]
        
        # 분할 실행
        segmenter.run(test_images, {})

# if __name__ == "__main__":
#     # 테스트용 코드
#     import cv2
#     import datetime

#     image_data = cv2.imread("/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/operationsystem_샘플이미지/C02_20250702_130727.jpg")  # 테스트 이미지 경로

#     mock_segmenter = MockPinjigSegmenter()
#     test_images = [CapturedImage(image_id="img1", camera_name="cam1", image_data=image_data, captured_at=datetime.datetime.now())]
#     results = mock_segmenter.run(test_images, {})
#     print(results)

# class MockObjectDetector(InferenceStrategy):
#     def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
#         print("Running Mock Object Detector...")
#         # 실제 모델은 여기서 이미지 배치([img.image_data for img in captured_images])를 받아 추론합니다.
        
#         for capture in captured_images:
#             # 결과를 해당 이미지의 image_id 아래에 저장
#             if capture.image_id not in inference_results:
#                 inference_results[capture.image_id] = {}

#             # 시뮬레이션을 위해 랜덤 객체 생성
#             num_objects = random.randint(1, 3)
#             detected_objects = []
#             for j in range(num_objects):
#                 detected_objects.append({
#                     "object_id": f"obj_{capture.camera_name}_{j}",
#                     "class": random.choice(["part_A", "part_B", "tool_X"]),
#                     "bbox": [random.randint(10, 400), random.randint(10, 400), 50, 50]
#                 })
            
#             inference_results[capture.image_id]["detections"] = detected_objects
#         return inference_results


# class MockPinjigSegmenter(InferenceStrategy):
#     def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
#         print("Running Mock Pinjig Segmenter...")
#         # 실제 모델은 여기서 이미지 배치([img.image_data for img in captured_images])를 받아 추론합니다.

#         for capture in captured_images:
#             if capture.image_id not in inference_results:
#                 image_data = capture.image_data
#                 # cv2 로 시각화
#                 import cv2
#                 cv2.imshow(f"Pinjig Segmenter - {capture.image_id}", image_data)
#                 cv2.waitKey(0)  # 키 입력을 기다림
#                 cv2.destroyAllWindows()
#                 inference_results[capture.image_id] = {}
        
        
#         return inference_results