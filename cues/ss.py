"""
Superpixels Straddling (SS) Cue
Measures how well a window fits object boundaries using superpixel segmentation.
Reference: Paper Section 2.4, Assignment Task (Common for both groups)
"""

import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
from utils import IntegralImage
from typing import List, Tuple, Dict


class SuperpixelsStraddling:
    def __init__(self, scale: float = 100, sigma: float = 0.8, min_size: int = 50):
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.superpixel_map = None
        self.integral_images = {} 
        self.superpixel_areas = {} 

    def compute_segmentation(self, image: np.ndarray):
        
        self.superpixel_map = felzenszwalb(
            image, 
            scale=self.scale, 
            sigma=self.sigma, 
            min_size=self.min_size
        )
        
        unique_ids = np.unique(self.superpixel_map)
        
        self.integral_images = {}
        self.superpixel_areas = {}

        for id in unique_ids:
        
            mask = (self.superpixel_map == id).astype(np.float64)
            ii = IntegralImage(mask)
            self.integral_images[id] = ii
            h, w = mask.shape
            total_area = ii.get_sum(0, 0, h-1, w-1)
            self.superpixel_areas[id] = total_area
        
        return self.superpixel_map
    
    def score_window(self, window: Tuple[int, int, int, int]) -> float:
     
        if self.superpixel_map is None:
            raise ValueError("you must call the function compute_segmentation before scoring windows.")
        
        r1, c1, r2, c2 = window
        window_area = (r2 - r1 + 1) * (c2 - c1 + 1)
        
        if window_area <= 0: 
            return 0.0
        
        straddle_penalty = 0.0
        
        for id, ii in self.integral_images.items():
            
            area_inside = ii.get_sum(r1, c1, r2, c2)
            if area_inside != 0:
                total_area = self.superpixel_areas[id]
                area_outside = total_area - area_inside
                straddle_penalty += min(area_inside, area_outside)
        
       
        ss_score = 1.0 - (straddle_penalty / window_area)
        
        return max(0.0, min(1.0, ss_score))