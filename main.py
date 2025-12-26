import argparse
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

from utils import DataSetLoader, compute_iou, generate_random_windows
from cues import MSSaliency, ColorContrast, SuperpixelsStraddling


PARAMS_FILE = "learned_params.pkl"

def train_ms_threshold(dataset):

    print("\nLearning MS Thresholds")
    ms= MSSaliency()
    best_thresholds= {}

    for scale in ms.scales:
        best_iou= -1.0
        best_t= 0.1
    
        for t in np.arange(0.05, 1.0, 0.05):
            
            total_iou = 0.0
            for image, ann in dataset:
                gt_windows = DataSetLoader.get_gt_windows(ann)
                
                h, w= image.shape[:2]
                channels= cv2.split(image)
                scale_map= np.zeros((h, w), dtype=np.float64)
                for ch in channels:
                    resized= cv2.resize(ch, (scale, scale))
                    sal= ms.spectral_residual(resized)
                    scale_map+= cv2.resize(sal, (w, h))
                
                saliency_map= scale_map / 3.0
                binary_map = (saliency_map > t).astype(np.uint8)
         
                contours, _= cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                image_score = 0.0

                for cnt in contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    blob_win = (y, x, y + bh, x + bw)
                    max_obj_iou = 0.0
                    for gt in gt_windows:
                        iou = compute_iou(blob_win, gt)
                        if iou > max_obj_iou:
                            max_obj_iou = iou
                    image_score += max_obj_iou
                
                total_iou += image_score
            
            if total_iou > best_iou:
                best_iou = total_iou
                best_t = t
        
        best_thresholds[scale] = best_t
        print(f"Best Threshold: {best_t:.2f} for Scale: {scale}")
    
    return best_thresholds

def train_bayesian_thresholds(dataset):
  
    print("\nLearning CC & SS Parameters")
    
    cc_candidates = [1.2, 1.4, 1.6, 1.8, 2.0]
    ss_candidates = [50, 100, 200, 300, 400]
    
    cc_data = {}
    ss_data = {}

    for p in cc_candidates:
        cc_data[p] = {'pos': [], 'neg': []}

    for p in ss_candidates:
        ss_data[p] = {'pos': [], 'neg': []}

    for image, ann in tqdm(dataset, desc="Generating Samples"):
        gt_windows = DataSetLoader.get_gt_windows(ann)
        samples = generate_random_windows(image.shape, count=200)
        
        labeled_samples = []
        for win in samples:
            is_pos = False
            for gt in gt_windows:
                if compute_iou(win, gt) > 0.5:
                    is_pos = True
                    break
            labeled_samples.append((win, is_pos))
            
    
        for theta in cc_candidates:
            cc = ColorContrast(theta_cc=theta, num_bins=(4,4,4))
            cc.compute_quantized_lab(image)
            for win, is_pos in labeled_samples:
                try:
                    score = cc.score_window(win)
                    key = 'pos' if is_pos else 'neg'
                    cc_data[theta][key].append(score)
                except: 
                    continue
    
        for scale in ss_candidates:
            ss = SuperpixelsStraddling(scale=scale, min_size=50)
            ss.compute_segmentation(image)
            for win, is_pos in labeled_samples:
                try:
                    score = ss.score_window(win)
                    key = 'pos' if is_pos else 'neg'
                    ss_data[scale][key].append(score)
                except: 
                    continue

    best_cc = 2.0 
    best_sep = -float('inf')
    for p in cc_candidates:
        pos = np.array(cc_data[p]['pos'])
        neg = np.array(cc_data[p]['neg'])
        if len(pos) > 0 and len(neg) > 0:
            sep = np.mean(pos) - np.mean(neg)
            if sep > best_sep:
                best_sep = sep
                best_cc = p
    print(f"Best CC Theta: {best_cc}")

   
    best_ss = 100
    best_sep = -float('inf')
    for p in ss_candidates:
        pos = np.array(ss_data[p]['pos'])
        neg = np.array(ss_data[p]['neg'])
        if len(pos) > 0 and len(neg) > 0:
            sep = np.mean(pos) - np.mean(neg)
            if sep > best_sep:
                best_sep = sep
                best_ss = p
    print(f"Best SS Scale: {best_ss}")
    
    return best_cc, best_ss

def train(args):
   
    print(f"STARTING TRAINING (Images: {args.num_images})")
    
   
    loader = DataSetLoader(args.data_path)
    dataset = loader.load_dataset(max_images=args.num_images)
    
    if not dataset:
        print("Error: No images loaded. Check data path.")
        return


    ms_thresholds = train_ms_threshold(dataset)
    theta_cc, theta_ss = train_bayesian_thresholds(dataset)

  
    learned_params = {
        'ms_thresholds': ms_thresholds,
        'theta_cc': theta_cc,
        'theta_ss': theta_ss
    }
    
    with open(PARAMS_FILE, 'wb') as f:
        pickle.dump(learned_params, f)
    print(f"\nTraining Complete. Parameters saved to {PARAMS_FILE}")

def test(args):
  
    if not os.path.exists(PARAMS_FILE):
        print(f"Error: {PARAMS_FILE} not found. Run --train comand first.") 
        return

    print(f"Testing (Images: {args.num_images})")
    
  
    with open(PARAMS_FILE, 'rb') as f:
        params = pickle.load(f)
    print(f"Loaded Params: {params}")

    loader = DataSetLoader(args.data_path)
    dataset = loader.load_dataset(max_images=args.num_images)

 
    ms = MSSaliency()
    ms.thresholds = params['ms_thresholds']
 
    out_dir = "Results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, (image, ann) in enumerate(dataset):
        print(f"Processing Image {i+1}/{len(dataset)}---")

        vis_img = image.copy()
  
        cc = ColorContrast(theta_cc=params['theta_cc'], num_bins=(4,4,4))
        cc.compute_quantized_lab(image)
        
        ss = SuperpixelsStraddling(scale=params['theta_ss'], min_size=50)
        ss.compute_segmentation(image)
        
        ii_ms = ms.get_integral_saliency(image)
        windows = generate_random_windows(image.shape, count=500)
        
        scored_windows = []
        
        for win in windows:
            try:
                s_ms = ms.score_window(ii_ms, win)
                s_cc = cc.score_window(win)
                s_ss = ss.score_window(win)
                
                final_score = (s_ms + s_cc + s_ss) / 3.0
                scored_windows.append((final_score, win))
            except:
                continue
        
        
        scored_windows.sort(key=lambda x: x[0], reverse=True)
        top_windows = scored_windows[:5]
        
     
        for score, (r1, c1, r2, c2) in top_windows:
            cv2.rectangle(vis_img, (c1, r1), (c2, r2), (0, 0, 255), 2)
            cv2.putText(vis_img, f"{score:.2f}", (c1, r1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
    
        for obj in ann['objects']:
            gx1, gy1, gx2, gy2 = obj['bbox']
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

      
        save_path = os.path.join(out_dir, f"result_{i}.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"  Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generic Objectness Estimation (Group B)")
    
    
    parser.add_argument('--train', action='store_true', help="Run Training Phase")
    parser.add_argument('--test', action='store_true', help="Run Testing/Visualization Phase")
    
    
    parser.add_argument('--data_path', type=str, default="./data", help="Path to PASCAL VOC 'data' folder")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images to use")
    
    args = parser.parse_args()
    
    if args.train:
        train(args)
    elif args.test: 
        test(args)
    else:
        print("Please specify --train or --test")
        print("Example: python main.py --train --num_images 20")

if __name__ == "__main__":
    main()