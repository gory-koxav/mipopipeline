# "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/mipo/timelapse_250224-250321/C_7/00000266.png"

import cv2

# 이미지 파일 경로
image_path = "/home/ksoeadmin/Projects/PYPJ/L2025022_mipo_operationsystem_uv/data/mipo/timelapse_250224-250321/C_7/00000266.png"

# 이미지 불러오기
try:
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image. Please check the file path.")
    else:
        # 가우시안 블러 적용 (커널 크기: 21x21)
        blurred_image = cv2.GaussianBlur(image, (21, 21), 0)

        # 블러 처리된 이미지를 파일로 저장
        cv2.imwrite("blurred_image.png", blurred_image)
        print("Gaussian blur applied successfully. The blurred image has been saved as 'blurred_image.png'.")

        # 참고: cv2.imshow()는 GUI가 필요하므로 이 환경에서는 작동하지 않습니다.
        # 이 코드를 사용자의 로컬 환경에서 실행하는 경우 아래 코드를 주석 해제하여
        # 화면에 이미지를 표시할 수 있습니다.
        # cv2.imshow("Original Image", image)
        # cv2.imshow("Blurred Image", blurred_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")