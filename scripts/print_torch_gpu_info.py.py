import torch
import platform

def print_torch_gpu_info():
    """
    PyTorch 버전과 사용 가능한 GPU 정보를 출력합니다.
    """
    print("--- PyTorch 및 GPU 정보 ---")

    # 1. PyTorch 버전 출력
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 컴파일 버전: {torch.version.cuda}")
    print(f"cuDNN 버전: {torch.backends.cudnn.version()}")

    # 2. GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        print("\nGPU 사용 가능: True")
        print(f"현재 사용 중인 GPU ID: {torch.cuda.current_device()}")
        print(f"총 GPU 개수: {torch.cuda.device_count()}")

        # 각 GPU의 정보 출력
        for i in range(torch.cuda.device_count()):
            print(f"\n--- GPU {i} 정보 ---")
            print(f"  장치 이름: {torch.cuda.get_device_name(i)}")
            print(f"  메모리 총량: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            print(f"  CUDA 코어 수: {torch.cuda.get_device_properties(i).multi_processor_count}")
            print(f"  CUDA 아키텍처: SM {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            print(f"  현재 할당된 메모리: {torch.cuda.memory_allocated(i) / (1024**2):.2f} MB")
            print(f"  캐시된 메모리: {torch.cuda.memory_cached(i) / (1024**2):.2f} MB")
    else:
        print("\nGPU 사용 가능: False (CUDA를 사용할 수 없습니다.)")
        print("GPU를 사용하려면 CUDA가 설치되어 있고 PyTorch가 CUDA 지원 버전으로 빌드되어 있어야 합니다.")

    # 3. 추가 시스템 정보 (선택 사항)
    print("\n--- 시스템 정보 ---")
    print(f"운영 체제: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"아키텍처: {platform.machine()}")
    print(f"Python 버전: {platform.python_version()}")

if __name__ == "__main__":
    print_torch_gpu_info()