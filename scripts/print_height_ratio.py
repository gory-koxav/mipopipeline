import cv2
import numpy as np

# 비율을 계산하고 화면에 출력하는 함수
def print_height_ratio(y_coord, total_height):
    """
    클릭된 y좌표와 이미지의 전체 높이를 사용하여 높이 비율을 계산하고 출력합니다.

    Args:
        y_coord (int): 마우스 클릭으로 얻은 y좌표.
        total_height (int): 이미지의 전체 높이.
    """
    # 높이 비율 계산
    height_ratio = y_coord / total_height
    
    print(f"이미지 전체 높이: {total_height}px")
    print(f"클릭된 지점의 y좌표: {y_coord}px")
    print(f"원본 대비 높이 비율: {height_ratio:.4f} (상단으로부터 {height_ratio * 100:.2f}%)")
    print("-" * 30)

# 마우스 클릭 이벤트를 처리할 콜백 함수
def mouse_callback(event, x, y, flags, param):
    """
    마우스 이벤트가 발생했을 때 호출되는 함수입니다.
    왼쪽 버튼을 클릭했을 때 좌표와 높이 비율을 출력합니다.
    """
    # 왼쪽 마우스 버튼이 클릭되었을 때 이벤트 처리
    if event == cv2.EVENT_LBUTTONDOWN:
        # param으로 전달된 이미지의 높이를 가져옴
        image_height = param.shape[0]
        # 비율 계산 및 출력 함수 호출
        print_height_ratio(y, image_height)

# --- 메인 코드 실행 부분 ---

# 이미지 파일 경로 (사용할 이미지 파일로 변경하세요)
image_path = '/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/mipo/timelapse_250224-250321_pinjig/D_6/00000583.png'

# 이미지 파일을 읽어옴
# cv2.imread()는 이미지를 NumPy 배열 형태로 불러옵니다.
image = cv2.imread(image_path)

# 이미지를 제대로 불러왔는지 확인
if image is None:
    print(f"'{image_path}' 파일을 찾을 수 없거나 열 수 없습니다.")
    print("파일 경로를 확인해주세요.")
else:
    # 이미지를 보여줄 창 생성
    window_name = 'Image - Click to get height ratio'
    cv2.namedWindow(window_name)
    
    # 마우스 콜백 함수 설정
    # cv2.setMouseCallback(창_이름, 콜백_함수, 콜백_함수에_전달할_인자)
    # 여기서는 'image' 자체를 param으로 전달하여 콜백 함수 내에서 이미지 정보를 사용합니다.
    cv2.setMouseCallback(window_name, mouse_callback, image)
    
    # 생성한 창에 이미지 표시
    cv2.imshow(window_name, image)
    
    print("이미지 창이 열렸습니다. 원하는 지점을 마우스 왼쪽 버튼으로 클릭하세요.")
    print("종료하려면 'q' 키를 누르거나 창을 닫으세요.")
    
    # 'q' 키를 누를 때까지 대기
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 모든 창 닫기
    cv2.destroyAllWindows()