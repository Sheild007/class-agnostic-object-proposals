import numpy as np

def compute_iou( box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1: 
        return 0.0
    
    inter = (x2 - x1 + 1) * (y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    return inter / (area1 + area2 - inter)

def generate_random_windows(img_shape, count=1000):

    h, w = img_shape[:2]
    windows = []
    for i in range(count):
        win_h = np.random.randint(20, h // 2)
        win_w = np.random.randint(20, w // 2)
        r = np.random.randint(0, h - win_h)
        c = np.random.randint(0, w - win_w)
        windows.append((r, c, r + win_h, c + win_w))
    return windows

def bbox_to_window(bbox):
    return (bbox[1], bbox[0], bbox[3], bbox[2])