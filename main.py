import argparse
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

from utils import DataSetLoader, compute_iou, generate_random_windows
from cues import MSSaliency, ColorContrast, SuperpixelsStraddling


PARAMS_FILE = "learned_params.pkl"

def build_histogram(values, bins=50, range_vals=(0, 1)):
    hist, edges = np.histogram(values, bins=bins, range=range_vals)
    hist = hist.astype(np.float64) + 1.0
    hist /= hist.sum()
    return hist, edges

def get_likelihood(value, hist, edges):

    if value < edges[0]:
        return hist[0]
    if value >= edges[-1]:
        return hist[-1]
    idx = np.searchsorted(edges[:-1], value, side='right') - 1
    idx = max(0, min(len(hist)-1, idx))
    return hist[idx]

def train_ms_threshold(dataset):
    
    print("\n[1/3] Learning MS Thresholds per Scale...")
    ms = MSSaliency()
    best_thresholds = {}

    for scale in ms.scales:
        print(f"Optimizing scale {scale}...")
        best_score = -1.0
        best_t = 0.5
    
        for t in np.arange(0.1, 0.9, 0.05):
            total_score = 0.0
            
            for image, ann in dataset:
                gt_windows = DataSetLoader.get_gt_windows(ann)
                if len(gt_windows) == 0:
                    continue
                
                h, w = image.shape[:2]
                channels = cv2.split(image)
                scale_map = np.zeros((h, w), dtype=np.float64)
               
                for ch in channels:
                    resized = cv2.resize(ch, (scale, scale))
                    sal = ms.spectral_residual(resized)
                    scale_map += cv2.resize(sal, (w, h))
                
                saliency_map = scale_map / 3.0
                binary_map = (saliency_map > t).astype(np.uint8)
         
                
                contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
               
                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:  
                      
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        blob_win = (y, x, y + bh - 1, x + bw - 1)
                        
                        max_iou = 0.0
                        for gt in gt_windows:
                            iou = compute_iou(blob_win, gt)
                            max_iou = max(max_iou, iou)
                        
                        total_score += max_iou
            
            if total_score > best_score:
                best_score = total_score
                best_t = t
        
        best_thresholds[scale] = best_t
        print(f"Best threshold: {best_t:.2f} (score: {best_score:.2f})")
    
    return best_thresholds

def train_bayesian_parameters(dataset, ms_thresholds):
  

    print("\n[2/3] Learning Bayesian Parameters (CC, SS, Combined)")
    
  
    cc_candidates = [1.3, 1.5, 1.7, 2.0, 2.5]
    ss_candidates = [50, 100, 150, 200, 300]
    cc_scores = {}
    for p in cc_candidates:
        cc_scores[p] = {'pos': [], 'neg': []}

    ss_scores = {}
    for p in ss_candidates:
        ss_scores[p] = {'pos': [], 'neg': []}

    ms_scores = {'pos': [], 'neg': []}
   
    ms = MSSaliency()
    ms.thresholds = ms_thresholds
    
    print(" Collecting training samples...")
    total_pos = 0
    total_neg = 0
    
    for img_idx, (image, ann) in enumerate(tqdm(dataset, desc="  Processing images")):
        gt_windows = DataSetLoader.get_gt_windows(ann)
        if len(gt_windows) == 0:
            continue
        
      
        h, w = image.shape[:2]
        candidates = generate_random_windows(image.shape, count=1000)
        
       
        labeled_windows = []
        for win in candidates:
            max_iou = max([compute_iou(win, gt) for gt in gt_windows], default=0)
            is_pos = max_iou > 0.5
            labeled_windows.append((win, is_pos))
            if is_pos:
                total_pos += 1
            else:
                total_neg += 1
        
      
        try:
            ii_ms = ms.get_integral_saliency(image)
            for win, is_pos in labeled_windows:
                try:
                    score = ms.score_window(ii_ms, win)
                    key = 'pos' if is_pos else 'neg'
                    ms_scores[key].append(score)
                except:
                    continue
        except:
            pass
        
       
        for theta in cc_candidates:
            try:
                cc = ColorContrast(theta_cc=theta, num_bins=(8, 8, 8))
                cc.compute_quantized_lab(image)
                for win, is_pos in labeled_windows:
                    try:
                        score = cc.score_window(win)
                        key = 'pos' if is_pos else 'neg'
                        cc_scores[theta][key].append(score)
                    except:
                        continue
            except:
                continue
        
       
        for scale in ss_candidates:
            try:
                ss = SuperpixelsStraddling(scale=scale, min_size=50)
                ss.compute_segmentation(image)
                for win, is_pos in labeled_windows:
                    try:
                        score = ss.score_window(win)
                        key = 'pos' if is_pos else 'neg'
                        ss_scores[scale][key].append(score)
                    except:
                        continue
            except:
                continue
    
    print(f"Collected {total_pos} positive and {total_neg} negative samples")
    

    print("Selecting best CC parameter...")
    best_cc = 2.0
    best_sep = -float('inf')
    for p in cc_candidates:
        pos = np.array(cc_scores[p]['pos'])
        neg = np.array(cc_scores[p]['neg'])
        if len(pos) > 10 and len(neg) > 10:
            sep = np.mean(pos) - np.mean(neg)
            print(f"    CC θ={p}: pos_mean={np.mean(pos):.3f}, neg_mean={np.mean(neg):.3f}, sep={sep:.3f}")
            if sep > best_sep:
                best_sep = sep
                best_cc = p
    print(f"Best CC: θ={best_cc}")
    
    print("Selecting best SS parameter...")
    best_ss = 100
    best_sep = -float('inf')
    for p in ss_candidates:
        pos = np.array(ss_scores[p]['pos'])
        neg = np.array(ss_scores[p]['neg'])
        if len(pos) > 10 and len(neg) > 10:
            sep = np.mean(pos) - np.mean(neg)
            print(f"    SS scale={p}: pos_mean={np.mean(pos):.3f}, neg_mean={np.mean(neg):.3f}, sep={sep:.3f}")
            if sep > best_sep:
                best_sep = sep
                best_ss = p
    print(f"Best SS: scale={best_ss}")
    
    
    print("Building Bayesian likelihoods...")
    
   
    ms_hist_pos, ms_edges = build_histogram(ms_scores['pos'], bins=50, range_vals=(0, 5))
    ms_hist_neg, _ = build_histogram(ms_scores['neg'], bins=50, range_vals=(0, 5))
    

    cc_hist_pos, cc_edges = build_histogram(cc_scores[best_cc]['pos'], bins=50, range_vals=(0, 5))
    cc_hist_neg, _ = build_histogram(cc_scores[best_cc]['neg'], bins=50, range_vals=(0, 5))
    
    
    ss_hist_pos, ss_edges = build_histogram(ss_scores[best_ss]['pos'], bins=50, range_vals=(0, 1))
    ss_hist_neg, _ = build_histogram(ss_scores[best_ss]['neg'], bins=50, range_vals=(0, 1))
    
 
    p_obj = total_pos / (total_pos + total_neg)
    p_bg = 1.0 - p_obj
    
    print(f"  Prior p(obj) = {p_obj:.3f}, p(bg) = {p_bg:.3f}")
    
    bayesian_params = {
        'ms': {'hist_pos': ms_hist_pos, 'hist_neg': ms_hist_neg, 'edges': ms_edges},
        'cc': {'hist_pos': cc_hist_pos, 'hist_neg': cc_hist_neg, 'edges': cc_edges},
        'ss': {'hist_pos': ss_hist_pos, 'hist_neg': ss_hist_neg, 'edges': ss_edges},
        'prior_obj': p_obj,
        'prior_bg': p_bg
    }
    
    return best_cc, best_ss, bayesian_params

def train(args):
  
    print(f"Trainig Start (using {args.num_images} images)")

    loader = DataSetLoader(args.data_path)
    dataset = loader.load_dataset(max_images=args.num_images)
    
    if not dataset:
        print("ERROR: No images loaded. Check data path.")
        return
    
    print(f"Loaded {len(dataset)} images successfully")
    
   
    ms_thresholds = train_ms_threshold(dataset)
    theta_cc, theta_ss, bayesian_params = train_bayesian_parameters(dataset, ms_thresholds)
    
  
    learned_params = {
        'ms_thresholds': ms_thresholds,
        'theta_cc': theta_cc,
        'theta_ss': theta_ss,
        'bayesian': bayesian_params
    }
    
    with open(PARAMS_FILE, 'wb') as f:
        pickle.dump(learned_params, f)
    
 
    print(f"[3/3] Training Complete!")

    print(f"Parameters saved to: {PARAMS_FILE}")
    print(f"\nLearned Parameters:")
    print(f"MS Thresholds: {ms_thresholds}")
    print(f"CC Theta: {theta_cc}")
    print(f"SS Scale: {theta_ss}")
    print(f" Bayesian priors: p(obj)={bayesian_params['prior_obj']:.3f}")
    print(f"\nRun testing with: python main.py --test --num_images 5")

def non_maximum_suppression(windows, overlap_thresh=0.3, max_windows=5, image_shape=None):
    
    if len(windows) == 0:
        return []
    
   
    boxes = []
    scores = []
    for score, (r1, c1, r2, c2) in windows:
        boxes.append([r1, c1, r2, c2])
        scores.append(score)
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
   
    order = np.argsort(scores)[::-1]
    
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        remaining_indices = order[1:]
        ious = np.zeros(len(remaining_indices))
        
        for idx, j in enumerate(remaining_indices):
            iou = compute_iou(
                (int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])),
                (int(boxes[j, 0]), int(boxes[j, 1]), int(boxes[j, 2]), int(boxes[j, 3]))
            )
            ious[idx] = iou
        
        order = remaining_indices[ious < overlap_thresh]
    
    
    final_windows = []
    
    for idx in keep:
        score, (r1, c1, r2, c2) = windows[idx]
        is_diverse = True
        
        for prev_score, (pr1, pc1, pr2, pc2) in final_windows:
           
            iou = compute_iou((r1, c1, r2, c2), (pr1, pc1, pr2, pc2))
            if iou > 0.02: 
                is_diverse = False
                break
            
       
            if image_shape is not None:
                center_r, center_c = (r1 + r2) // 2, (c1 + c2) // 2
                prev_center_r, prev_center_c = (pr1 + pr2) // 2, (pc1 + pc2) // 2
                
              
                img_diag = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
                min_dist = img_diag * 0.35
                dist = np.sqrt((center_r - prev_center_r)**2 + (center_c - prev_center_c)**2)
                
                if dist < min_dist:
                    is_diverse = False
                    break
                
             
                size_ratio = min((r2-r1)*(c2-c1), (pr2-pr1)*(pc2-pc1)) / max((r2-r1)*(c2-c1), (pr2-pr1)*(pc2-pc1))
                if size_ratio > 0.7 and dist < min_dist * 1.5:
                    is_diverse = False
                    break
        
        if is_diverse:
            final_windows.append((score, (r1, c1, r2, c2)))
      
        if len(final_windows) >= max_windows:
            break
    
    return final_windows

def compute_objectness_score(s_ms, s_cc, s_ss, bayesian_params):
   
    p_ms_obj = get_likelihood(s_ms, bayesian_params['ms']['hist_pos'], bayesian_params['ms']['edges'])
    p_ms_bg = get_likelihood(s_ms, bayesian_params['ms']['hist_neg'], bayesian_params['ms']['edges'])
    
    p_cc_obj = get_likelihood(s_cc, bayesian_params['cc']['hist_pos'], bayesian_params['cc']['edges'])
    p_cc_bg = get_likelihood(s_cc, bayesian_params['cc']['hist_neg'], bayesian_params['cc']['edges'])
    
    p_ss_obj = get_likelihood(s_ss, bayesian_params['ss']['hist_pos'], bayesian_params['ss']['edges'])
    p_ss_bg = get_likelihood(s_ss, bayesian_params['ss']['hist_neg'], bayesian_params['ss']['edges'])
    
    # Naive Bayes: p(obj|cues) ∝ p(MS|obj) * p(CC|obj) * p(SS|obj) * p(obj)
    p_obj_numerator = p_ms_obj * p_cc_obj * p_ss_obj * bayesian_params['prior_obj']
    p_bg_numerator = p_ms_bg * p_cc_bg * p_ss_bg * bayesian_params['prior_bg']
    
   
    total = p_obj_numerator + p_bg_numerator
    if total > 0:
        return p_obj_numerator / total
    return 0.0

def test(args):
   
    if not os.path.exists(PARAMS_FILE):
        print(f"Error: {PARAMS_FILE} not found.")
        print("Run training first: python main.py --train --num_images 20")
        return

   
    print(f"Testing Start(Images: {args.num_images})")
  
    
    with open(PARAMS_FILE, 'rb') as f:
        params = pickle.load(f)
    
    print("Loaded parameters:")
    print(f"MS Thresholds: {params['ms_thresholds']}")
    print(f"CC Theta: {params['theta_cc']}")
    print(f"SS Scale: {params['theta_ss']}")

    loader = DataSetLoader(args.data_path)
    dataset = loader.load_dataset(max_images=args.num_images)

    
    ms = MSSaliency()
    ms.thresholds = params['ms_thresholds']
 
    out_dir = "Results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, (image, ann) in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Processing image..")
        vis_img = image.copy()
        
   
        print("Initializing cues...")
        cc = ColorContrast(theta_cc=params['theta_cc'], num_bins=(8,8,8))
        cc.compute_quantized_lab(image)
        
        ss = SuperpixelsStraddling(scale=params['theta_ss'], min_size=50)
        ss.compute_segmentation(image)
        
      
        ii_ms = ms.get_integral_saliency(image)

        print("Generating candidate windows...")
        windows = generate_random_windows(image.shape, count=2000)
        
        print("Scoring windows with Bayesian combinatio")
        scored_windows = []
        
        for win in windows:
            try:
                r1, c1, r2, c2 = win
             
                s_ms = ms.score_window(ii_ms, win)
                s_cc = cc.score_window(win)
                s_ss = ss.score_window(win)
                
              
                objectness = compute_objectness_score(s_ms, s_cc, s_ss, params['bayesian'])
                
                scored_windows.append((objectness, win))
            except Exception as e:
                continue
        
        if len(scored_windows) == 0:
            print("  WARNING: No valid windows scored!")
            continue
        
     
        scored_windows.sort(key=lambda x: x[0], reverse=True)
        
      
        print("Applying non-maximum suppression.")
        top_candidates = scored_windows[:500]
        top_windows = non_maximum_suppression(
            top_candidates, 
            overlap_thresh=0.10,  
            max_windows=3,
            image_shape=image.shape
        )
        
        print(f"  Selected {len(top_windows)} windows after NMS")
        if len(top_windows) > 0:
            print(f"  Score range: [{top_windows[-1][0]:.3f}, {top_windows[0][0]:.3f}]")
        
      
        for idx, (score, (r1, c1, r2, c2)) in enumerate(top_windows):
            cv2.rectangle(vis_img, (c1, r1), (c2, r2), (0, 0, 255), 2)
            label = f"#{idx+1}: {score:.2f}"
            cv2.putText(vis_img, label, (c1, r1-5 if r1 > 20 else r1+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
     
        save_path = os.path.join(out_dir, f"result_{i}.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"  Saved: {save_path}")

    print(f"Testing complete! Results saved in {out_dir}/")
    

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