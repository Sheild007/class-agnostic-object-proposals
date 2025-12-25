import numpy as np
import cv2
from utils import Integralimg
from typing import Tuple


class ColorContrast:

    def __init__(self, theta_cc: float = 1.5, num_bins: Tuple = (16, 16, 16)):
      
        self.theta_cc = theta_cc
        self.num_bins = num_bins
        self.lab_img = None
        self.quantized_img = None
        self.integral_hists = []  
        self.total_bins = num_bins[0] * num_bins[1] * num_bins[2]    

    def compute_quantized_lab(self, img: np.ndarray):
  
        self.lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
        L, A, B = cv2.split(self.lab_img)
        
        L_quant = (L.astype(np.float32)*self.num_bins[0]/256.0).astype(np.int32)
        A_quant = (A.astype(np.float32)*self.num_bins[1]/256.0).astype(np.int32)
        B_quant = (B.astype(np.float32)*self.num_bins[2]/256.0).astype(np.int32)

        L_quant = np.clip(L_quant, 0, self.num_bins[0] - 1)
        A_quant = np.clip(A_quant, 0, self.num_bins[1] - 1)
        B_quant = np.clip(B_quant, 0, self.num_bins[2] - 1)
        
       
        self.quantized_img = (L_quant*self.num_bins[1]*self.num_bins[2]+A_quant*self.num_bins[2]+B_quant)
        
        self.integral_hists = []
        for id in range(self.total_bins):
            bin_mask = (self.quantized_img == id).astype(np.float64)
            self.integral_hists.append(Integralimg(bin_mask))
        
        return self.quantized_img
    
    def get_histogram(self, window: Tuple[int, int, int, int]) -> np.ndarray:

        r1, c1, r2, c2 = window
        hist = np.zeros(self.total_bins)
        
        for id in range(self.total_bins):
            hist[id] = self.integral_hists[id].get_sum(r1, c1, r2, c2)

        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        
        return hist
    
    def chi_square_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:

        chi2 = 0.0
        for i in range(len(hist1)):
            if hist1[i] + hist2[i] > 0:
                chi2 += (hist1[i] - hist2[i]) ** 2 / (hist1[i] + hist2[i])
        return chi2
    
    def score_window(self, window: Tuple[int, int, int, int]) -> float:
   
        if self.quantized_img is None:
            raise ValueError("you must call compute_quantized_lab first before scoring windows.")
        
        r1, c1, r2, c2 = window
        h, w = self.quantized_img.shape

        win_h = r2 - r1 + 1
        win_w = c2 - c1 + 1
        
        expand_h = int((win_h * self.theta_cc - win_h) / 2)
        expand_w = int((win_w * self.theta_cc - win_w) / 2)
        
        surr_r1 = max(0, r1 - expand_h)
        surr_c1 = max(0, c1 - expand_w)
        surr_r2 = min(h - 1, r2 + expand_h)
        surr_c2 = min(w - 1, c2 + expand_w)
    
        hist_window = self.get_histogram(window)
        hist_surround = self.get_histogram((surr_r1, surr_c1, surr_r2, surr_c2))
        
        window_area = win_h * win_w
        surround_area = (surr_r2 - surr_r1 + 1) * (surr_c2 - surr_c1 + 1)
        ring_area = surround_area - window_area
        
        if ring_area > 0:
            hist_ring = (hist_surround * surround_area - hist_window * window_area) / ring_area
            if hist_ring < 0:
                hist_ring = np.zeros_like(hist_ring)
            if hist_ring.sum() > 0:
                hist_ring /= hist_ring.sum()
            else:
                hist_ring = np.zeros_like(hist_ring)
        else:
            hist_ring = hist_surround
  
        cc_score = self.chi_square_distance(hist_window, hist_ring)
        
        return cc_score
    
  
