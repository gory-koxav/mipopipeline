# cctv_event_detector/inference/strategies/assembly_classifier.py
import numpy as np
import cv2
from typing import Dict, Any, List
from ultralytics import YOLO

from cctv_event_detector.inference.strategies.base import InferenceStrategy
from cctv_event_detector.core.models import CapturedImage
from config import ASSEMBLY_CLS_MODEL_PATH, BOUNDARY_TARGET_CLASSES, SAM_CLASSIFICATION_OUTPUT_DIR

class AssemblyClassifier(InferenceStrategy):
    """조립 상태 분류 모델 (Strategy 구현체)"""
    
    def __init__(self):
        """
        YOLO classification 모델을 초기화하고 설정을 로드합니다.
        """
        print("Initializing Assembly Classifier...")
        self.cls_model = YOLO(ASSEMBLY_CLS_MODEL_PATH)
        print(f"✅ Assembly Classification model loaded from: {ASSEMBLY_CLS_MODEL_PATH}")
        
        # BOUNDARY_TARGET_CLASSES를 사용하여 처리할 클래스 설정
        self.target_classes = set(BOUNDARY_TARGET_CLASSES) if BOUNDARY_TARGET_CLASSES else set()
        if self.target_classes:
            print(f"🎯 Target classes for assembly classification: {self.target_classes}")
        else:
            print("🎯 No target classes specified, processing all detected objects.")
        
        # 결과 이미지 저장 디렉토리 설정
        self.output_dir = SAM_CLASSIFICATION_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Assembly classification output dir: {self.output_dir}")
    
    def run(self, captured_images: List[CapturedImage], inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력된 이미지들에 대해 조립 상태 분류를 수행하고 결과를 저장합니다.
        """
        print("Running Assembly Classifier...")
        
        for capture in captured_images:
            image_id = capture.image_id
            original_image = capture.image_data
            
            # YOLO Object Detection 결과 가져오기
            detections = inference_results.get(image_id, {}).get("detections", [])
            if not detections:
                continue
            
            # 대상 클래스에 해당하는 detection만 필터링
            target_detections = []
            for detection in detections:
                # target_classes가 비어있거나, 해당 클래스가 target에 포함되면 처리
                should_process = not self.target_classes or detection['class_name'] in self.target_classes
                if should_process:
                    target_detections.append(detection)
            
            if not target_detections:
                print(f"No target class objects found in '{image_id}'")
                continue
            
            print(f"Processing {len(target_detections)} target objects in '{image_id}'")
            
            # 각 대상 객체에 대해 분류 수행
            assembly_classifications = []
            for idx, detection in enumerate(target_detections):
                classification_result = self._classify_single_object(
                    original_image, 
                    detection, 
                    image_id, 
                    idx
                )
                if classification_result:
                    assembly_classifications.append(classification_result)
            
            # 결과를 inference_results에 추가
            if assembly_classifications:
                # 전체 이미지의 assembly_status 결정 (가장 복잡한 상태로 설정)
                assembly_status = self._determine_overall_status(assembly_classifications)
                inference_results[image_id]["assembly_status"] = assembly_status
                
                # 상세 분류 결과도 저장 (필요시 사용)
                inference_results[image_id]["assembly_classifications"] = assembly_classifications
                
                # 시각화 결과 저장
                self._save_visualization(
                    original_image, 
                    target_detections, 
                    assembly_classifications, 
                    image_id
                )
                
                print(f"✅ Assembly status for '{image_id}': {assembly_status}")
        
        return inference_results
    
    def _classify_single_object(self, image: np.ndarray, detection: Dict[str, Any], 
                               image_id: str, idx: int) -> Dict[str, Any]:
        """
        단일 객체를 크롭하고 분류를 수행합니다.
        """
        # BBOX 좌표 추출 (object_detector는 [x_min, y_min, width, height] 형식 사용)
        x_min, y_min, width, height = detection['bbox_xywh']
        
        # 이미지 크기 확인
        img_h, img_w = image.shape[:2]
        
        # BBOX 좌표 보정 (이미지 경계를 벗어나지 않도록)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_min + width)
        y_max = min(img_h, y_min + height)
        
        # 유효한 크기인지 확인
        if x_max <= x_min or y_max <= y_min:
            print(f"⚠️ Invalid bbox for object {idx}: skipping")
            return None
        
        # 이미지 크롭
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # (512, 512) 크기로 리사이즈
        resized_image = cv2.resize(cropped_image, (512, 512))
        
        # YOLO classification 모델로 추론
        results = self.cls_model(resized_image, verbose=False)
        result = results[0]
        
        # Top1과 Top5 결과 추출
        top1_idx = result.probs.top1
        top1_class = self.cls_model.names[top1_idx]
        top1_conf = result.probs.top1conf.item()
        
        # Top5 클래스와 확률
        top5_indices = result.probs.top5
        top5_classes = [self.cls_model.names[int(idx)] for idx in top5_indices]
        top5_probs = result.probs.top5conf.tolist()
        
        # 개별 크롭 이미지 저장 (시각화 포함)
        self._save_cropped_result(resized_image, result, image_id, idx, detection['class_name'])
        
        return {
            "object_index": idx,
            "object_class": detection['class_name'],
            "bbox": detection['bbox_xywh'],
            "top1_class": top1_class,
            "top1_confidence": top1_conf,
            "top5_classes": top5_classes,
            "top5_probs": top5_probs
        }
    
    def _determine_overall_status(self, classifications: List[Dict[str, Any]]) -> str:
        """
        전체 이미지의 assembly_status를 결정합니다.
        분류 결과에 따라 simple_assembly 또는 complex_assembly로 결정
        """
        # 예시 로직: top1_class 기준으로 판단
        # 실제 사용시에는 비즈니스 로직에 맞게 수정 필요
        complex_indicators = ["complex", "assembled", "combined"]
        
        for classification in classifications:
            top1_class = classification['top1_class'].lower()
            for indicator in complex_indicators:
                if indicator in top1_class:
                    return "complex_assembly"
        
        # 여러 객체가 있으면 complex로 간주
        if len(classifications) > 1:
            return "complex_assembly"
        
        return "simple_assembly"
    
    def _save_cropped_result(self, image: np.ndarray, result, image_id: str, 
                            idx: int, original_class: str):
        """
        크롭된 이미지에 분류 결과를 시각화하여 저장합니다.
        """
        # 타임스탬프별 디렉토리 생성
        timed_output_dir = self.output_dir / image_id
        timed_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지에 분류 결과 어노테이션
        annotated_image = self._annotate_image(image, result)
        
        # 파일명 생성 및 저장
        top1_class = self.cls_model.names[result.probs.top1]
        filename = f"assembly_{original_class}_{idx:03d}_{top1_class}.png"
        output_path = timed_output_dir / filename
        
        cv2.imwrite(str(output_path), annotated_image)
        print(f"  💾 Saved cropped result: {output_path}")
    
    def _annotate_image(self, image: np.ndarray, result) -> np.ndarray:
        """
        이미지에 Top5 분류 결과를 텍스트로 표시합니다.
        (mask_classifier.py의 _annotate_image 함수 참고)
        """
        annotated_image = image.copy()
        pos_y = 40
        
        for j, idx in enumerate(result.probs.top5):
            text = f"{j+1}. {self.cls_model.names[int(idx)]}: {result.probs.top5conf[j]:.2f}"
            # 검은색 외곽선
            cv2.putText(annotated_image, text, (20, pos_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            # 흰색 텍스트
            cv2.putText(annotated_image, text, (20, pos_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            pos_y += 40
        
        return annotated_image
    
    def _save_visualization(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                          classifications: List[Dict[str, Any]], image_id: str):
        """
        전체 이미지에 대한 시각화 결과를 저장합니다.
        BBOX와 분류 결과를 함께 표시
        """
        timed_output_dir = self.output_dir / image_id
        timed_output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_image = image.copy()
        
        # 각 detection과 classification 결과를 시각화
        for detection, classification in zip(detections, classifications):
            # BBOX 그리기
            x_min, y_min, width, height = detection['bbox_xywh']
            x_max = x_min + width
            y_max = y_min + height
            
            # 색상 결정 (complex면 빨간색, simple이면 초록색)
            if "complex" in classification['top1_class'].lower():
                color = (0, 0, 255)  # Red for complex
            else:
                color = (0, 255, 0)  # Green for simple
            
            # BBOX 그리기
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # 레이블 텍스트 생성
            label = f"{detection['class_name']}: {classification['top1_class']} ({classification['top1_confidence']:.2f})"
            
            # 텍스트 배경 및 텍스트 그리기
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, (x_min, y_min - 25), (x_min + label_w, y_min), color, -1)
            cv2.putText(vis_image, label, (x_min, y_min - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 전체 시각화 결과 저장
        output_path = timed_output_dir / f"_ASSEMBLY_OVERVIEW.jpg"
        cv2.imwrite(str(output_path), vis_image)
        print(f"✅ Assembly visualization saved: {output_path}")


# MockAssemblyClassifier를 실제 구현체로 교체
# 다른 파일에서 MockAssemblyClassifier를 import하는 경우를 위해 별칭 제공
MockAssemblyClassifier = AssemblyClassifier