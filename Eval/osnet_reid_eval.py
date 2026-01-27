"""
Multi-Granularity Fusion Evaluation Script
Paper: "Enhancing Person Re-ID with Pose-Guided Local Part Attention"
Dataset: DukeMTMC-reID
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt
import torchreid
from torchreid.utils import FeatureExtractor

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = Path("./duke")   # Path to Duke Dataset
MODEL_PATH = './log/osnet_duke/model/model.pth.tar-50' # Your Duke-Trained Weights
GALLERY_LIMIT = 2000            # Set to None for full dataset (slow but accurate)
QUERY_LIMIT = 500               # Set to None for full dataset
# ---------------------

# ==================== 1. DATA PARSER ====================
@dataclass
class ReIDSample:
    image_path: str
    person_id: int
    camera_id: int
    
    @staticmethod
    def parse(filename: str, full_path: str):
        try:
            if filename.startswith('-1') or filename.startswith('0000'): return None
            parts = filename.replace('.jpg', '').split('_')
            pid = int(parts[0])
            cam_str = parts[1]
            if 's' in cam_str: cid = int(cam_str.split('s')[0].replace('c', '')) # Market
            else: cid = int(cam_str.replace('c', '')) # Duke
            return ReIDSample(full_path, pid, cid)
        except: return None

# ==================== 2. MODELS ====================

class DeepFeatureExtractor:
    def __init__(self): pass

class PoseEstimator:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
    def extract(self, img):
        if img is None: return None
        results = self.model(img, verbose=False, device=0 if DEVICE=='cuda' else 'cpu')
        if not results: return None
        res = results[0]
        if res.keypoints is None or res.keypoints.xy.shape[0] == 0: return None
        kps = res.keypoints.xy[0].cpu().numpy()
        conf = res.keypoints.conf[0].cpu().numpy()
        if np.mean(conf) < 0.4: return None
        return kps
    def compare(self, kps1, kps2):
        if kps1 is None or kps2 is None: return 0.0
        k1_norm = kps1 - np.mean(kps1, axis=0)
        k2_norm = kps2 - np.mean(kps2, axis=0)
        dist = np.linalg.norm(k1_norm - k2_norm)
        return 1 / (1 + dist/100.0)

class OSNetBody(DeepFeatureExtractor):
    def __init__(self):
        print(f"Loading OSNet (Body) from {MODEL_PATH}...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=MODEL_PATH,
            device=DEVICE,
            image_size=(256, 128)
        )
    def extract(self, img):
        if img is None or img.size == 0: return None
        return self.extractor([img])[0].cpu().numpy()

class OSNetFace(DeepFeatureExtractor):
    def __init__(self):
        print(f"Loading OSNet (Face) from {MODEL_PATH}...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=MODEL_PATH, # Ideally use specialized face weights here if available
            device=DEVICE,
            image_size=(128, 128)
        )
    def extract(self, img, kps):
        if kps is None: return None
        face_pts = kps[:5]
        valid = face_pts[face_pts[:,0] > 0]
        if len(valid) < 2: return None
        x1, y1 = np.min(valid, axis=0)
        x2, y2 = np.max(valid, axis=0)
        h, w = img.shape[:2]
        pad = int((x2-x1) * 0.5) 
        crop = img[max(0, int(y1-pad)):min(h, int(y2+pad)), max(0, int(x1-pad)):min(w, int(x2+pad))]
        if crop.size == 0: return None
        return self.extractor([crop])[0].cpu().numpy()

class OSNetHair(DeepFeatureExtractor):
    def __init__(self):
        print(f"Loading OSNet (Hair) from {MODEL_PATH}...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=MODEL_PATH, # Ideally use specialized hair weights here if available
            device=DEVICE,
            image_size=(128, 256)
        )
    def extract(self, img, kps):
        if kps is None: return None
        nose = kps[0]
        if nose[0] == 0: return None
        crop = img[0:int(nose[1]), :]
        if crop.size == 0: return None
        return self.extractor([crop])[0].cpu().numpy()

# ==================== 3. SYSTEM ====================

class MultiGranularitySystem:
    def __init__(self, modalities: List[str]):
        self.modalities = modalities
        self.pose_model = PoseEstimator()
        
        # Load models based on modalities
        # Note: We share instances if possible to save memory, but here we init separate for clarity
        self.body_net = OSNetBody() if 'body' in modalities else None
        self.face_net = OSNetFace() if 'face' in modalities else None
        self.hair_net = OSNetHair() if 'hair' in modalities else None
        
        # Weights for fusion
        # Body is robust (0.7), Hair/Face refine difficult cases (0.15 each)
        self.weights = {'pose': 0.0, 'body': 0.7, 'hair': 0.15, 'face': 0.15}
        
    def extract(self, img):
        feats = {}
        # Always get pose for cropping
        kps = self.pose_model.extract(img)
        feats['pose'] = kps
        
        if self.body_net: feats['body'] = self.body_net.extract(img)
        
        # Only try Local branches if pose exists
        if kps is not None:
            if self.face_net: feats['face'] = self.face_net.extract(img, kps)
            if self.hair_net: feats['hair'] = self.hair_net.extract(img, kps)
            
        return feats

    def compute_similarity(self, f1, f2):
        score, total_w = 0.0, 0.0
        for m in self.modalities:
            if m == 'pose': continue # We don't use pose for similarity score in this advanced version
            
            if m not in f1 or m not in f2: continue
            if f1[m] is None or f2[m] is None: continue
            
            w = self.weights[m]
            sim = max(0, 1 - cosine(f1[m], f2[m]))
            
            score += sim * w
            total_w += w
            
        return score / total_w if total_w > 0 else 0.0

# ==================== 4. METRICS & MAIN ====================

def compute_metrics(system, name, gal_data, q_data):
    results = []
    for q in tqdm(q_data, desc=f"Matching {name}"):
        matches = []
        for g in gal_data:
            if q['s'].person_id == g['s'].person_id and q['s'].camera_id == g['s'].camera_id: continue
            sim = system.compute_similarity(q['f'], g['f'])
            matches.append({'match': q['s'].person_id == g['s'].person_id, 'sim': sim})
        matches.sort(key=lambda x: x['sim'], reverse=True)
        results.append(matches)
        
    if not results: return None
    
    # Ranks
    r1 = np.mean([1 if r[0]['match'] else 0 for r in results if r])
    r5 = np.mean([1 if any(m['match'] for m in r[:5]) else 0 for r in results if r])
    r10 = np.mean([1 if any(m['match'] for m in r[:10]) else 0 for r in results if r])
    
    # mAP
    aps = []
    for r in results:
        y_true = [m['match'] for m in r]
        if not any(y_true): continue
        cumulative_correct = 0
        precision_sum = 0
        for i, match in enumerate(y_true):
            if match:
                cumulative_correct += 1
                precision_sum += cumulative_correct / (i + 1)
        aps.append(precision_sum / sum(y_true))
    map_score = np.mean(aps) if aps else 0.0
    
    return {'name': name, 'r1': r1, 'r5': r5, 'r10': r10, 'map': map_score}

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model weights not found at {MODEL_PATH}")
        print("Please run train_duke.py first!")
        exit()

    print(f"Scanning {DATASET_PATH}...")
    gal_files = sorted(list((DATASET_PATH / "bounding_box_test").glob("*.jpg")))
    query_files = sorted(list((DATASET_PATH / "query").glob("*.jpg")))
    
    # Parse & Limit
    gal_samples = [ReIDSample.parse(f.name, str(f)) for f in gal_files]
    query_samples = [ReIDSample.parse(f.name, str(f)) for f in query_files]
    gal_samples = [s for s in gal_samples if s][:GALLERY_LIMIT]
    query_samples = [s for s in query_samples if s][:QUERY_LIMIT]
    
    print(f"Gallery: {len(gal_samples)} | Query: {len(query_samples)}")
    
    # Define Experiments
    experiments = [
        (['body'], "Baseline (Body Only)"),
        (['body', 'hair'], "Body + Hair"),
        (['body', 'hair', 'face'], "Body + Hair + Face (Full)")
    ]
    
    final_res = []
    
    # Run
    for mods, name in experiments:
        print(f"\n--- Running: {name} ---")
        sys = MultiGranularitySystem(mods)
        
        # Extract
        g_data = []
        for s in tqdm(gal_samples, desc="Extract Gallery"):
            img = cv2.imread(s.image_path)
            f = sys.extract(img)
            g_data.append({'s': s, 'f': f})
            
        q_data = []
        for s in tqdm(query_samples, desc="Extract Query"):
            img = cv2.imread(s.image_path)
            f = sys.extract(img)
            q_data.append({'s': s, 'f': f})
            
        res = compute_metrics(sys, name, g_data, q_data)
        if res: final_res.append(res)
        
    # Print Table
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: Multi-Granularity Fusion (DukeMTMC-reID)")
    print(f"{'='*80}")
    print(f"{'Method':<30} | {'Rank-1':<8} | {'Rank-5':<8} | {'Rank-10':<8} | {'mAP':<8}")
    print("-" * 80)
    for r in final_res:
        print(f"{r['name']:<30} | {r['r1']*100:>6.2f}% | {r['r5']*100:>6.2f}% | {r['r10']*100:>6.2f}% | {r['map']*100:>6.2f}%")
    print("="*80)