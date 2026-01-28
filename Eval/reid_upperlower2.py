"""
Final Evaluation Script with Comparison Table
Paper Method: "Pose-Guided Synergistic Attention with Query Expansion"
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from typing import List, Dict
import torchreid
from torchreid.utils import FeatureExtractor
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = Path("./duke/dukemtmc-reid/DukeMTMC-reID") 

# PATHS TO YOUR MODELS
PATH_BODY  = './log/osnet_duke/model/model.pth.tar-50'
PATH_UPPER = './log/osnet_duke_upper/model/model.pth.tar-30' 
PATH_LOWER = './log/osnet_duke_lower/model/model.pth.tar-30' 

GALLERY_LIMIT = None  # Set None for full test
QUERY_LIMIT = None    # Set None for full test

# ==================== 1. DATA & POSE UTILS ====================

@dataclass
class ReIDSample:
    image_path: str
    person_id: int
    camera_id: int
    @staticmethod
    def parse(filename, full_path):
        try:
            if filename.startswith('-1') or filename.startswith('0000'): return None
            parts = filename.replace('.jpg', '').split('_')
            return ReIDSample(full_path, int(parts[0]), int(parts[1].replace('c', '').split('s')[0]))
        except: return None

class PoseEstimator:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
        
    def extract(self, img):
        if img is None: return None
        # Suppress YOLO output
        res = self.model(img, verbose=False, device=0 if DEVICE=='cuda' else 'cpu')[0]
        if res.keypoints is None or res.keypoints.xy.shape[0] == 0: return None
        kps = res.keypoints.xy[0].cpu().numpy()
        # Only return if confidence is decent
        return kps if np.mean(res.keypoints.conf[0].cpu().numpy()) > 0.4 else None

# ==================== 2. EXPERT EXTRACTORS ====================

class OSNetBody:
    def __init__(self):
        print(f"Loading GLOBAL BODY from {PATH_BODY}...")
        self.net = FeatureExtractor('osnet_ain_x1_0', PATH_BODY, device=DEVICE, image_size=(256, 128))
    def extract(self, img):
        # Flatten immediately
        return self.net([img])[0].cpu().numpy().flatten()

class OSNetUpper:
    def __init__(self):
        print(f"Loading UPPER EXPERT from {PATH_UPPER}...")
        self.net = FeatureExtractor('osnet_ain_x1_0', PATH_UPPER, device=DEVICE, image_size=(128, 128))
        
    def extract(self, img, kps):
        if kps is None: return None
        shoulders = kps[5:7]
        hips = kps[11:13]
        
        if np.sum(shoulders[:,0] > 0) == 0 or np.sum(hips[:,0] > 0) == 0:
            return None 
            
        y1 = np.min(shoulders[shoulders[:,0]>0, 1])
        y2 = np.max(hips[hips[:,0]>0, 1])
        
        valid_pts = np.vstack((shoulders, hips))
        valid_pts = valid_pts[valid_pts[:,0]>0]
        center_x = np.mean(valid_pts[:,0])
        width = (np.max(valid_pts[:,0]) - np.min(valid_pts[:,0])) * 1.4 
        
        x1 = int(max(0, center_x - width/2))
        x2 = int(min(img.shape[1], center_x + width/2))
        y1 = int(max(0, y1 - (y2-y1)*0.2)) 
        y2 = int(min(img.shape[0], y2))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20: return None
        
        crop = cv2.resize(crop, (128, 128))
        return self.net([crop])[0].cpu().numpy().flatten()

class OSNetLower:
    def __init__(self):
        print(f"Loading LOWER EXPERT from {PATH_LOWER}...")
        self.net = FeatureExtractor('osnet_ain_x1_0', PATH_LOWER, device=DEVICE, image_size=(128, 128))
        
    def extract(self, img, kps):
        if kps is None: return None
        hips = kps[11:13]
        ankles = kps[15:17]
        
        if np.sum(hips[:,0] > 0) == 0 or np.sum(ankles[:,0] > 0) == 0:
            return None 
            
        y1 = np.min(hips[hips[:,0]>0, 1])
        y2 = np.max(ankles[ankles[:,0]>0, 1])
        
        valid_pts = np.vstack((hips, ankles))
        valid_pts = valid_pts[valid_pts[:,0]>0]
        center_x = np.mean(valid_pts[:,0])
        width = (np.max(valid_pts[:,0]) - np.min(valid_pts[:,0])) * 1.5
        
        x1 = int(max(0, center_x - width/2))
        x2 = int(min(img.shape[1], center_x + width/2))
        y1 = int(max(0, y1))
        y2 = int(min(img.shape[0], y2 + (y2-y1)*0.1)) 
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20: return None
        
        crop = cv2.resize(crop, (128, 128))
        return self.net([crop])[0].cpu().numpy().flatten()

# ==================== 3. ADAPTIVE SYSTEM ====================

class AdaptivePartSystem:
    def __init__(self, use_parts=False):
        self.use_parts = use_parts
        self.pose_model = PoseEstimator()
        self.body_net = OSNetBody()
        
        self.upper_net = OSNetUpper() if use_parts else None
        self.lower_net = OSNetLower() if use_parts else None
        
    def extract(self, img):
        feats = {'body': None, 'upper': None, 'lower': None, 'valid_upper': False, 'valid_lower': False}
        
        feats['body'] = self.body_net.extract(img)
        
        if self.use_parts:
            kps = self.pose_model.extract(img)
            
            feat_up = self.upper_net.extract(img, kps)
            if feat_up is not None:
                feats['upper'] = feat_up
                feats['valid_upper'] = True 
                
            feat_low = self.lower_net.extract(img, kps)
            if feat_low is not None:
                feats['lower'] = feat_low
                feats['valid_lower'] = True 
                
        return feats

    def compute_similarity(self, f1, f2):
        # --- STRATEGY: SYNERGY (80/10/10) ---
        w_body = 0.80
        w_upper = 0.10
        w_lower = 0.10
        
        # 1. Global Baseline Check
        if f1['body'] is None or f2['body'] is None: return 0.0
        
        if not self.use_parts:
            return 1 - cosine(f1['body'], f2['body'])

        # 2. Check Validity & Redistribute Weights
        if not f1['valid_upper'] or not f2['valid_upper'] or f1['upper'] is None or f2['upper'] is None:
            w_upper = 0.0
            w_body += 0.10

        if not f1['valid_lower'] or not f2['valid_lower'] or f1['lower'] is None or f2['lower'] is None:
            w_lower = 0.0
            w_body += 0.10
            
        # 3. Compute Scores
        score = 0
        total_w = 0
        
        # Body
        sim_body = max(0, 1 - cosine(f1['body'], f2['body']))
        score += w_body * sim_body
        total_w += w_body
        
        # Upper
        if w_upper > 0:
            sim_upper = max(0, 1 - cosine(f1['upper'], f2['upper']))
            score += w_upper * sim_upper
            total_w += w_upper
            
        # Lower
        if w_lower > 0:
            sim_lower = max(0, 1 - cosine(f1['lower'], f2['lower']))
            score += w_lower * sim_lower
            total_w += w_lower
            
        return score / total_w if total_w > 0 else 0

# ==================== 4. EVALUATION LOOP ====================

def run_evaluation(system_name, use_parts, gal_samples, query_samples):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {system_name}")
    print(f"{'='*60}")
    
    system = AdaptivePartSystem(use_parts=use_parts)
    
    # 1. Extract Gallery
    g_data = []
    for s in tqdm(gal_samples, desc="Extracting Gallery"):
        img = cv2.imread(s.image_path)
        g_data.append({'s': s, 'f': system.extract(img)})
        
    # 2. Extract Query
    q_data = []
    for s in tqdm(query_samples, desc="Extracting Query"):
        img = cv2.imread(s.image_path)
        q_data.append({'s': s, 'f': system.extract(img)})
        
    # 3. Compute Metrics (With Query Expansion)
    results = []
    for q in tqdm(q_data, desc="Matching"):
        # --- PASS 1: Standard Matching ---
        matches = []
        for g in g_data:
            if q['s'].person_id == g['s'].person_id and q['s'].camera_id == g['s'].camera_id: continue
            
            sim = system.compute_similarity(q['f'], g['f'])
            matches.append({'match': q['s'].person_id == g['s'].person_id, 'sim': sim, 'f': g['f']})
            
        matches.sort(key=lambda x: x['sim'], reverse=True)
        
        # --- PASS 2: Query Expansion ---
        if use_parts and len(matches) > 0:
            top1 = matches[0]['f'] 
            
            expanded_q = {}
            expanded_q['body'] = (q['f']['body'] + top1['body']) / 2.0
            
            if q['f']['valid_upper'] and top1['valid_upper']:
                expanded_q['upper'] = (q['f']['upper'] + top1['upper']) / 2.0
                expanded_q['valid_upper'] = True
            else:
                expanded_q['upper'] = q['f']['upper']
                expanded_q['valid_upper'] = q['f']['valid_upper']
                
            if q['f']['valid_lower'] and top1['valid_lower']:
                expanded_q['lower'] = (q['f']['lower'] + top1['lower']) / 2.0
                expanded_q['valid_lower'] = True
            else:
                expanded_q['lower'] = q['f']['lower']
                expanded_q['valid_lower'] = q['f']['valid_lower']
                
            final_matches = []
            for g in g_data:
                if q['s'].person_id == g['s'].person_id and q['s'].camera_id == g['s'].camera_id: continue
                sim = system.compute_similarity(expanded_q, g['f'])
                final_matches.append({'match': q['s'].person_id == g['s'].person_id, 'sim': sim})
                
            final_matches.sort(key=lambda x: x['sim'], reverse=True)
            results.append(final_matches)
        else:
            clean_matches = [{'match': m['match'], 'sim': m['sim']} for m in matches]
            results.append(clean_matches)

    # --- CALCULATE CMC METRICS ---
    def get_rank_k(k, res_list):
        return np.mean([1 if any(m['match'] for m in r[:k]) else 0 for r in res_list if r])

    r1  = get_rank_k(1, results)
    r5  = get_rank_k(5, results)
    r10 = get_rank_k(10, results)
    
    # Calculate mAP
    aps = []
    for r in results:
        y_true = [m['match'] for m in r]
        if not any(y_true): continue
        cum = 0
        prec = 0
        for i, m in enumerate(y_true):
            if m:
                cum += 1
                prec += cum / (i + 1)
        aps.append(prec / sum(y_true))
    map_score = np.mean(aps) if aps else 0.0
    
    # Return Dictionary for Table
    return {
        'name': system_name,
        'r1': r1,
        'r5': r5,
        'r10': r10,
        'map': map_score
    }

# ==================== 5. MAIN ====================
if __name__ == "__main__":
    print("Parsing Dataset...")
    gal_files = sorted(list((DATASET_PATH / "bounding_box_test").glob("*.jpg")))[:GALLERY_LIMIT]
    q_files = sorted(list((DATASET_PATH / "query").glob("*.jpg")))[:QUERY_LIMIT]
    
    gal_samples = [ReIDSample.parse(f.name, str(f)) for f in gal_files]
    q_samples = [ReIDSample.parse(f.name, str(f)) for f in q_files]
    gal_samples = [s for s in gal_samples if s is not None]
    q_samples = [s for s in q_samples if s is not None]
    
    # Store results
    final_results = []

    # 1. Baseline
    res_base = run_evaluation("Body Only", False, gal_samples, q_samples)
    final_results.append(res_base)
    
    # 2. Synergistic + QE
    res_parts = run_evaluation("Body + Upper + Lower", True, gal_samples, q_samples)
    final_results.append(res_parts)

    # --- GENERATE TABLE ---
    print("\n" + "="*85)
    print(f"{'METHOD':<30} | {'Rank-1':<10} | {'Rank-5':<10} | {'Rank-10':<10} | {'mAP':<10}")
    print("-" * 85)
    
    for res in final_results:
        print(f"{res['name']:<30} | {res['r1']*100:>8.2f}% | {res['r5']*100:>8.2f}% | {res['r10']*100:>8.2f}% | {res['map']*100:>8.2f}%")
    
    print("="*85 + "\n")