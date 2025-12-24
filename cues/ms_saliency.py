import numpy as np
import cv2
from utils import IntegralImage

class MSSaliency:
    def __init__(self, scales=[16, 24, 32, 48, 64]):
        self.scales = scales
        self.thresholds = {s: 0.5 for s in scales} 

    def spectral_residual(self, img_channel: np.ndarray) -> np.ndarray:
        # 1. 2D FFT
        f = np.fft.fft2(img_channel)
        magnitude = np.abs(f)
        phase = np.angle(f)
        
        # 2. Log Spectrum
        log_amplitude = np.log(magnitude + 1e-9)  
        
        # 3. Spectral Residual 
        avg_log_amplitude = cv2.boxFilter(log_amplitude, ddepth=-1, ksize=(3, 3))
        spectral_residual = log_amplitude - avg_log_amplitude
        
        # 4. Reconstruction
        f_residual = np.exp(spectral_residual + 1j * phase)
        saliency_map = np.abs(np.fft.ifft2(f_residual)) ** 2
        saliency_map = cv2.GaussianBlur(saliency_map, (3, 3), 0)
        
        # Normalize
        min_val, max_val = np.min(saliency_map), np.max(saliency_map)
        if max_val > min_val:
            saliency_map = (saliency_map - min_val) / (max_val - min_val)
        return saliency_map

    def get_integral_saliency(self, image: np.ndarray) -> IntegralImage:
    
        h, w = image.shape[:2]
        accumulated_map = np.zeros((h, w), dtype=np.float64)
    
        channels = cv2.split(image)
        
        for scale in self.scales:
            scale_accum = np.zeros((h, w), dtype=np.float64)
            for channel in channels:
                resized = cv2.resize(channel, (scale, scale))
                sal = self.spectral_residual(resized)
                scale_accum += cv2.resize(sal, (w, h))
            
          
            avg_map = scale_accum / 3.0
            
           
            theta = self.thresholds[scale]
            binary_map = (avg_map > theta).astype(np.float64)
            
            
            accumulated_map += binary_map

       
        return IntegralImage(accumulated_map)

    def score_window(self, ii_saliency: IntegralImage, window: tuple) -> float:
      
        r1, c1, r2, c2 = window
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        
        if area <= 0: 
            return 0.0
            
        sum_counts = ii_saliency.get_sum(r1, c1, r2, c2)
        return sum_counts / area