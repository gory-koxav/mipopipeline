import os
import shutil
import glob
import re
from pathlib import Path

def copy_overlay_files():
    # 기본 경로 설정
    base_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv"
    source_path = os.path.join(base_path, "output")
    destination_base_path = os.path.join(base_path, "collected_overlay_results")
    
    # 복사된 파일 개수 카운트
    copied_count = 0
    
    # 숫자 8자리 패턴 (정확히 8자리 숫자로 끝나는 폴더)
    eight_digit_pattern = re.compile(r'^(.+)_(\d{8})$')
    
    print(f"소스 경로: {source_path}")
    print(f"목적지 기본 경로: {destination_base_path}")
    print("="*60)
    
    # source_path 폴더에서만 직접 하위 폴더들 확인
    try:
        # source_path의 직접 하위 항목들만 가져오기
        items = os.listdir(source_path)
        
        for item in items:
            item_path = os.path.join(source_path, item)
            
            # 폴더인지 확인
            if os.path.isdir(item_path):
                # 폴더명이 8자리 숫자로 끝나는지 확인 (예: C_2_00000000, D_4_00000000)
                match = eight_digit_pattern.match(item)
                if match:
                    # 앞부분 추출 (예: C_2, D_4)
                    prefix = match.group(1)
                    eight_digits = match.group(2)
                    
                    print(f"8자리 숫자 폴더 발견: {item} (prefix: {prefix})")
                    
                    # 해당 prefix 폴더 생성
                    destination_path = os.path.join(destination_base_path, prefix)
                    os.makedirs(destination_path, exist_ok=True)
                    
                    # 해당 폴더에서 _OVERLAY_RESULT.jpg 파일 찾기
                    overlay_files = glob.glob(os.path.join(item_path, "*_OVERLAY_RESULT.jpg"))
                    
                    if overlay_files:
                        for file_path in overlay_files:
                            # 원본 파일명
                            original_filename = os.path.basename(file_path)
                            
                            # 새로운 파일명 생성 (폴더명_원본파일명)
                            new_filename = f"{item}_{original_filename}"
                            
                            # 목적지 전체 경로
                            destination_file = os.path.join(destination_path, new_filename)
                            
                            try:
                                # 파일 복사
                                shutil.copy2(file_path, destination_file)
                                copied_count += 1
                                print(f"복사 완료: {original_filename} -> {prefix}/{new_filename}")
                                
                            except Exception as e:
                                print(f"복사 실패: {file_path} -> 오류: {e}")
                    else:
                        print(f"  -> _OVERLAY_RESULT.jpg 파일 없음")
                else:
                    # 8자리 숫자가 아닌 폴더는 무시 (예: D06_20250708105259)
                    print(f"무시된 폴더 (8자리 숫자 아님): {item}")
            else:
                # 파일인 경우는 무시
                pass
                
    except FileNotFoundError:
        print(f"오류: 소스 경로를 찾을 수 없습니다: {source_path}")
        return
    except PermissionError:
        print(f"오류: 소스 경로에 접근 권한이 없습니다: {source_path}")
        return
    
    print("="*60)
    print(f"총 {copied_count}개의 파일이 복사되었습니다.")
    print(f"파일들이 각각의 prefix 폴더에 저장되었습니다: {destination_base_path}")

if __name__ == "__main__":
    copy_overlay_files()