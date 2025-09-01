import os
import shutil
import glob
from pathlib import Path

def copy_detect_images():
    # 기본 경로 설정
    base_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv"
    source_path = os.path.join(base_path, "runs", "detect")
    destination_base_path = os.path.join(base_path, "runs", "collected_detect_images")
    
    # 목적지 기본 폴더 생성 (존재하지 않으면)
    os.makedirs(destination_base_path, exist_ok=True)
    
    # 복사된 파일 개수 카운트
    copied_count = 0
    
    print(f"소스 경로: {source_path}")
    print(f"목적지 기본 경로: {destination_base_path}")
    print("="*60)
    
    # detect 폴더 내의 모든 하위 폴더 순회
    for root, dirs, files in os.walk(source_path):
        # 현재 폴더명 추출
        folder_name = os.path.basename(root)
        
        # 루트 폴더 자체는 스킵
        if root == source_path:
            continue
            
        print(f"처리 중인 폴더: {folder_name}")
        
        # 해당 폴더에서 image*.jpg 패턴의 파일 찾기
        image_files = glob.glob(os.path.join(root, "image*.jpg"))
        
        if image_files:
            for file_path in image_files:
                # 원본 파일명과 확장자 분리
                original_filename = os.path.basename(file_path)
                filename_without_ext = os.path.splitext(original_filename)[0]  # image0, image11 등
                
                # 목적지 폴더 경로 (이미지명과 동일한 폴더)
                image_folder_path = os.path.join(destination_base_path, filename_without_ext)
                os.makedirs(image_folder_path, exist_ok=True)
                
                # 새로운 파일명 생성 (중복 방지를 위해 폴더명 추가)
                new_filename = f"{filename_without_ext}_{folder_name}.jpg"
                
                # 목적지 전체 경로
                destination_file = os.path.join(image_folder_path, new_filename)
                
                try:
                    # 파일 복사
                    shutil.copy2(file_path, destination_file)
                    copied_count += 1
                    print(f"  복사 완료: {original_filename} -> {filename_without_ext}/{new_filename}")
                    
                except Exception as e:
                    print(f"  복사 실패: {file_path} -> 오류: {e}")
        else:
            print(f"  -> image*.jpg 파일 없음")
    
    print("="*60)
    print(f"총 {copied_count}개의 파일이 복사되었습니다.")
    print(f"파일들이 각각의 이미지별 폴더에 저장되었습니다: {destination_base_path}")
    
    # 생성된 폴더 목록 출력
    if os.path.exists(destination_base_path):
        created_folders = [f for f in os.listdir(destination_base_path) 
                          if os.path.isdir(os.path.join(destination_base_path, f))]
        if created_folders:
            print(f"생성된 이미지 폴더들: {', '.join(sorted(created_folders))}")

if __name__ == "__main__":
    copy_detect_images()