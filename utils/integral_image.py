import numpy as np

class IntegralImage:
    def __init__(self, image: np.ndarray):
        self.original_shape = image.shape
        self.ii = self.compute_integral_image(image)

    def compute_integral_image(self, img: np.ndarray) -> np.ndarray:
      
        # II(x,y) = sum of all pixels above and to the left of (x,y)
       
        # V(x,y) = I(x,y) + V(x-1,y)
        row_integral = np.cumsum(img.astype(np.float64), axis=0)

        # II(x,y) = I(x,y) + II(x-1,y) + II(x,y-1) - II(x-1,y-1)
        ii = np.cumsum(row_integral, axis=1)
        
        # Pading with zero top row and left column to simplify(avoid checks) sum calculations
        pad_width = ((1, 0), (1, 0)) 
        ii_padded = np.pad(ii, pad_width, mode='constant', constant_values=0)
        
        return ii_padded

    def get_sum(self, r1: int, c1: int, r2: int, c2: int) -> float:
      
        
        h, w = self.original_shape
        r1 = max(0, min(r1, h - 1))
        c1 = max(0, min(c1, w - 1))
        r2 = max(0, min(r2, h - 1))
        c2 = max(0, min(c2, w - 1))

        
        if r1 > r2: 
            r1, r2 = r2, r1
        if c1 > c2: 
            c1, c2 = c2, c1

     
        # Using the padded integral image:
        # A = (r2+1, c2+1) -> Bottom Right (row r2, col c2 in original )
        # B = (r1,   c2+1) -> Top Right (row r1-1, col c2 in original )
        # C = (r2+1, c1)   -> Bottom Left (row r2, col c1-1 in original )
        # D = (r1,   c1)   -> Top Left (row r1-1, col c1-1 in original )

        A = self.ii[r2 + 1, c2 + 1]
        B = self.ii[r1,     c2 + 1]
        C = self.ii[r2 + 1, c1]
        D = self.ii[r1,     c1]

        return A - B - C + D