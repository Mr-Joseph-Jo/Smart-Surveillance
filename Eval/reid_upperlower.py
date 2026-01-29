"""
Occlusion-Robust Evaluation Script
System: Global Body + Pose-Guided Upper/Lower Experts
Features: Auto-switching weights based on part visibility
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

# PATHS TO YOUR NEW EXPERTS
PATH_BODY  = './log/osnet_duke/model/model.pth.tar-50'
PATH_UPPER = './log/osnet_duke_upper/model/model.pth.tar-30' # Your 30-epoch model
PATH_LOWER = './log/osnet_duke_lower/model/model.pth.tar-30' # Your 30-epoch model

GALLERY_LIMIT = 800  # Set None for full test
QUERY_LIMIT = 100    # Set None for full test

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
        res = self.model(img, verbose=False, device=0 if DEVICE=='cuda' else 'cpu')[0]
        if res.keypoints is None or res.keypoints.xy.shape[0] == 0: return None
        kps = res.keypoints.xy[0].cpu().numpy()
        return kps if np.mean(res.keypoints.conf[0].cpu().numpy()) > 0.4 else None

# ==================== 2. EXPERT EXTRACTORS ====================

class OSNetBody:
    def __init__(self):
        print(f"Loading GLOBAL BODY from {PATH_BODY}...")
        self.net = FeatureExtractor('osnet_ain_x1_0', PATH_BODY, device=DEVICE, image_size=(256, 128))
    def extract(self, img):
        return self.net([img])[0].cpu().numpy()

class OSNetUpper:
    def __init__(self):
        print(f"Loading UPPER EXPERT from {PATH_UPPER}...")
        self.net = FeatureExtractor('osnet_ain_x1_0', PATH_UPPER, device=DEVICE, image_size=(128, 128))
        
    def extract(self, img, kps):
        if kps is None: return None
        # Indices: 5,6 (Shoulders) | 11,12 (Hips)
        shoulders = kps[5:7]
        hips = kps[11:13]
        
        # Valid if we see at least one shoulder AND one hip
        if np.sum(shoulders[:,0] > 0) == 0 or np.sum(hips[:,0] > 0) == 0:
            return None # Torso occluded/missing
            
        y1 = np.min(shoulders[shoulders[:,0]>0, 1])
        y2 = np.max(hips[hips[:,0]>0, 1])
        
        valid_pts = np.vstack((shoulders, hips))
        valid_pts = valid_pts[valid_pts[:,0]>0]
        center_x = np.mean(valid_pts[:,0])
        width = (np.max(valid_pts[:,0]) - np.min(valid_pts[:,0])) * 1.4 # Padding
        
        x1 = int(max(0, center_x - width/2))
        x2 = int(min(img.shape[1], center_x + width/2))
        y1 = int(max(0, y1 - (y2-y1)*0.2)) # Include Neck
        y2 = int(min(img.shape[0], y2))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20: return None
        
        crop = cv2.resize(crop, (128, 128))
        return self.net([crop])[0].cpu().numpy()

class OSNetLower:
    def __init__(self):
        print(f"Loading LOWER EXPERT from {PATH_LOWER}...")
        self.net = FeatureExtractor('osnet_ain_x1_0', PATH_LOWER, device=DEVICE, image_size=(128, 128))
        
    def extract(self, img, kps):
        if kps is None: return None
        # Indices: 11,12 (Hips) | 15,16 (Ankles)
        hips = kps[11:13]
        ankles = kps[15:17]
        
        # Valid if we see hips AND ankles
        if np.sum(hips[:,0] > 0) == 0 or np.sum(ankles[:,0] > 0) == 0:
            return None # Legs occluded/missing
            
        y1 = np.min(hips[hips[:,0]>0, 1])
        y2 = np.max(ankles[ankles[:,0]>0, 1])
        
        valid_pts = np.vstack((hips, ankles))
        valid_pts = valid_pts[valid_pts[:,0]>0]
        center_x = np.mean(valid_pts[:,0])
        width = (np.max(valid_pts[:,0]) - np.min(valid_pts[:,0])) * 1.5
        
        x1 = int(max(0, center_x - width/2))
        x2 = int(min(img.shape[1], center_x + width/2))
        y1 = int(max(0, y1))
        y2 = int(min(img.shape[0], y2 + (y2-y1)*0.1)) # Include shoes
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20: return None
        
        crop = cv2.resize(crop, (128, 128))
        return self.net([crop])[0].cpu().numpy()

# ==================== 3. ADAPTIVE SYSTEM ====================

class AdaptivePartSystem:
    def __init__(self, use_parts=False):
        self.use_parts = use_parts
        self.pose_model = PoseEstimator()
        self.body_net = OSNetBody()
        
        # Load experts only if needed
        self.upper_net = OSNetUpper() if use_parts else None
        self.lower_net = OSNetLower() if use_parts else None
        
    def extract(self, img):
        feats = {'body': None, 'upper': None, 'lower': None, 'valid_upper': False, 'valid_lower': False}
        
        # 1. Global Feature (Always extracted)
        feats['body'] = self.body_net.extract(img)
        
        if self.use_parts:
            kps = self.pose_model.extract(img)
            
            # 2. Upper Body
            feat_up = self.upper_net.extract(img, kps)
            if feat_up is not None:
                feats['upper'] = feat_up
                feats['valid_upper'] = True # MARK VALID
                
            # 3. Lower Body
            feat_low = self.lower_net.extract(img, kps)
            if feat_low is not None:
                feats['lower'] = feat_low
                feats['valid_lower'] = True # MARK VALID
                
        return feats

    def compute_similarity(self, f1, f2):
        # 1. Start with the strong Baseline
        w_body = 0.80
        w_upper = 0.10
        w_lower = 0.10
        
        # 2. Check Validity (Standard checks)
        has_upper = f1['valid_upper'] and f2['valid_upper'] and f1['upper'] is not None and f2['upper'] is not None
        has_lower = f1['valid_lower'] and f2['valid_lower'] and f1['lower'] is not None and f2['lower'] is not None
        
        # 3. Handle Occlusion (Dynamic Weight Shifting)
        if not has_upper:
            # If Upper is missing, shift its weight to Body (safest)
            w_upper = 0.0
            w_body += 0.10
            
        if not has_lower:
            # If Lower is missing, shift its weight to Body
            w_lower = 0.0
            w_body += 0.10
            
        # 4. Compute Weighted Score
        score = 0
        total_w = 0
        
        # Body
        if f1['body'] is not None and f2['body'] is not None:
            v1 = f1['body'].flatten()
            v2 = f2['body'].flatten()
            sim_body = max(0, 1 - cosine(v1, v2))
            score += w_body * sim_body
            total_w += w_body
            
        # Upper (Bonus Detail)
        if w_upper > 0:
            v1_up = f1['upper'].flatten()
            v2_up = f2['upper'].flatten()
            sim_upper = max(0, 1 - cosine(v1_up, v2_up))
            score += w_upper * sim_upper
            total_w += w_upper

        # Lower (Bonus Detail)
        if w_lower > 0:
            v1_low = f1['lower'].flatten()
            v2_low = f2['lower'].flatten()
            sim_lower = max(0, 1 - cosine(v1_low, v2_low))
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
        
    # 3. Compute Metrics
    results = []
    for q in tqdm(q_data, desc="Matching"):
        matches = []
        for g in g_data:
            # Skip same-camera pairs (standard Re-ID protocol)
            if q['s'].person_id == g['s'].person_id and q['s'].camera_id == g['s'].camera_id:
                continue
            
            sim = system.compute_similarity(q['f'], g['f'])
            matches.append({'match': q['s'].person_id == g['s'].person_id, 'sim': sim})
            
        matches.sort(key=lambda x: x['sim'], reverse=True)
        results.append(matches)

    # Calculate Rank-1 and mAP
    r1 = np.mean([1 if r[0]['match'] else 0 for r in results if r])
    
    aps = []
    for r in results:
        y_true = [m['match'] for m in r]
        if not any(y_true): continue
        cum_correct = 0
        prec_sum = 0
        for i, match in enumerate(y_true):
            if match:
                cum_correct += 1
                prec_sum += cum_correct / (i + 1)
        aps.append(prec_sum / sum(y_true))
    map_score = np.mean(aps) if aps else 0.0
    
    print(f"\nRESULTS FOR {system_name}:")
    print(f"  Rank-1: {r1*100:.2f}%")
    print(f"  mAP:    {map_score*100:.2f}%")
    return r1, map_score

# ==================== 5. MAIN ====================
if __name__ == "__main__":
    # Load Data
    print("Parsing Dataset...")
    gal_files = sorted(list((DATASET_PATH / "bounding_box_test").glob("*.jpg")))[:GALLERY_LIMIT]
    q_files = sorted(list((DATASET_PATH / "query").glob("*.jpg")))[:QUERY_LIMIT]
    
    gal_samples = [ReIDSample.parse(f.name, str(f)) for f in gal_files]
    q_samples = [ReIDSample.parse(f.name, str(f)) for f in q_files]
    gal_samples = [s for s in gal_samples if s is not None]
    q_samples = [s for s in q_samples if s is not None]
    
    # EXPERIMENT 1: Baseline (Body Only)
    run_evaluation("Baseline (Global Body Only)", False, gal_samples, q_samples)
    
    # EXPERIMENT 2: Adaptive Ensemble
    run_evaluation("Adaptive Ensemble (Body + Upper + Lower)", True, gal_samples, q_samples)