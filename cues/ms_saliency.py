import numpy as np
import cv2
from utils import IntegralImage

class MSSaliency:
    def __init__(self, scales=[16, 24, 32, 48, 64]):
        self.scales = scales
        self.threshold = 0.5 

    def spectral_residual(self, img_channel: np.ndarray) -> np.ndarray:
       
        # 1. 2d FFT
        f = np.fft.fft2(img_channel)

        # 2. amp and phase
        magnitude = np.abs(f)
        phase = np.angle(f)
        
        # 3. Log spectral Spectrum
        log_amplitude = np.log(magnitude + 0.00000001)  
        
        # 4. Spectral Residual
        avg_log_amplitude = cv2.boxFilter(log_amplitude, ddepth=-1, ksize=(3, 3))
        spectral_residual = log_amplitude - avg_log_amplitude
        
        # 5. Reconstruction
        f_residual = np.exp(spectral_residual + 1j * phase)
        saliency_map = np.abs(np.fft.ifft2(f_residual)) ** 2
        # 6 smoothing
        saliency_map = cv2.GaussianBlur(saliency_map, (3, 3), 0)
        
        min_val, max_val = np.min(saliency_map), np.max(saliency_map)
        if max_val > min_val:
            saliency_map = (saliency_map - min_val) / (max_val - min_val)
            
        return saliency_map

    