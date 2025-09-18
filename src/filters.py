import cv2
import numpy as np

class Filters:
    """
    다양한 이미지 처리 커널(필터)을 관리하고 적용하는 클래스입니다.
    """
    # 사용 가능한 커널들을 정의한 딕셔너리
    Kernels = {
        "Original" : np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
        "Blur" : np.ones((3,3), dtype=np.float32) / 9.0,
        "Gaussian blur" : np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16.0,
        "Sharpen" : np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
        "Sobel (x)" : np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
        "Sobel (y)" : np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
        "Edge Detection" : np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32),
        "Emboss" : np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32)
    }

    def __init__(self, kernels=Kernels):
        """
        Filters 클래스를 초기화합니다.
        """
        self.kernels = kernels
        # --- 내부 변수 구현 ---
        self.kernel_names = list(self.kernels.keys())
        self.current_kernel_index = 0

    def apply_filter(self, frame, filter_name) -> np.array:
        """
        선택된 이름의 필터 커널을 프레임에 적용합니다.
        """
        if filter_name not in self.kernels:
            # 필터 이름이 없으면 원본 프레임 반환
            return frame
        
        kernel = self.kernels[filter_name]
        return cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

    def apply_current_filter(self, frame) -> np.array:

        current_filter_name = self.get_current_filter_name()
        return self.apply_filter(frame, current_filter_name)

    def get_current_filter_name(self) -> str:

        return self.kernel_names[self.current_kernel_index]

    def switch_next_filter(self):

        num_kernels = len(self.kernel_names)
        self.current_kernel_index = (self.current_kernel_index + 1) % num_kernels

    def switch_previous_filter(self):

        num_kernels = len(self.kernel_names)
        self.current_kernel_index = (self.current_kernel_index - 1) % num_kernels