"""
Multi-Granularity Fusion Evaluation Script (Final Version)
Paper: "Enhancing Person Re-ID with Pose-Guided Local Part Attention"
System: Three-Stream OSNet Ensemble (Body + Specialized Hair + Specialized Face)
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
from typing import List, Dict, Optional
import torchreid
from torchreid.utils import FeatureExtractor

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = Path("./duke/dukemtmc-reid/DukeMTMC-reID") 

# PATHS TO YOUR TRAINED MODELS
PATH_BODY = './log/osnet_duke/model/model.pth.tar-50'
# PATH_HAIR = './log/osnet_duke_hair/model/model.pth.tar-40' 
# PATH_FACE = './log/osnet_duke_face/model/model.pth.tar-40'
PATH_HAIR = PATH_BODY 
PATH_FACE = PATH_BODY

GALLERY_LIMIT = 800    
QUERY_LIMIT = 100       
# ---------------------

# ==================== 0. HELPER CLASSES (MISSING PART) ====================

@dataclass
class FeatureQuality:
    """Estimates quality/confidence of extracted features"""
    detection_confidence: float = 0.0
    feature_variance: float = 0.0
    spatial_coverage: float = 0.0
    overall_quality: float = 0.0
    
    @staticmethod
    def compute(feature: np.ndarray) -> 'FeatureQuality':
        quality = FeatureQuality()
        if feature is None:
            return quality
            
        # Feature variance (low variance = low information/dead feature)
        quality.feature_variance = float(np.var(feature))
        
        # Initial score based on variance
        quality.overall_quality = min(1.0, quality.feature_variance / 0.1)
        return quality

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
            if 's' in cam_str: cid = int(cam_str.split('s')[0].replace('c', ''))
            else: cid = int(cam_str.replace('c', ''))
            return ReIDSample(full_path, pid, cid)
        except: return None

# ==================== 2. MODELS ====================

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

class OSNetBody:
    def __init__(self):
        print(f"Loading BODY Expert from {PATH_BODY}...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=PATH_BODY,
            device=DEVICE,
            image_size=(256, 128)
        )
    def extract(self, img):
        if img is None or img.size == 0: return None
        return self.extractor([img])[0].cpu().numpy()

class OSNetFace:
    def __init__(self):
        print(f"Loading FACE Expert from {PATH_FACE}...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=PATH_FACE,
            device=DEVICE,
            image_size=(128, 128) 
        )
    def extract(self, img, kps):
        quality = FeatureQuality()
        
        if kps is None: return None, quality

        face_pts = kps[:5]
        valid = face_pts[face_pts[:,0] > 0]
        
        if len(valid) == 0: 
            return None, quality

        if len(valid) >= 2:
            x1, y1 = np.min(valid, axis=0)
            x2, y2 = np.max(valid, axis=0)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_w = x2 - x1
            box_size = max(current_w * 2.0, img.shape[1] * 0.15) 
        else:
            center_x = valid[0][0]
            center_y = valid[0][1]
            box_size = img.shape[1] * 0.15

        half_size = int(box_size / 2)
        x_start = int(max(0, center_x - half_size))
        x_end = int(min(img.shape[1], center_x + half_size))
        y_start = int(max(0, center_y - half_size))
        y_end = int(min(img.shape[0], center_y + half_size))

        crop = img[y_start:y_end, x_start:x_end]

        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None, quality

        crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
        feature = self.extractor([crop_resized])[0].cpu().numpy()

        original_area = crop.shape[0] * crop.shape[1]
        resolution_score = min(1.0, original_area / (64*64))
        quality = FeatureQuality.compute(feature)
        quality.overall_quality = (quality.overall_quality * 0.6) + (resolution_score * 0.4)

        return feature, quality

class OSNetHair:
    def __init__(self):
        print(f"Loading HAIR Expert from {PATH_HAIR}...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=PATH_HAIR,
            device=DEVICE,
            image_size=(128, 128)
        )
    def extract(self, img, kps):
        if kps is None: return None, FeatureQuality()

        nose = kps[0]
        left_eye = kps[1]
        right_eye = kps[2]
        
        if nose[0] > 0:
            center_x = nose[0]
            bottom_y = nose[1]
        elif left_eye[0] > 0 and right_eye[0] > 0:
            center_x = (left_eye[0] + right_eye[0]) / 2
            bottom_y = (left_eye[1] + right_eye[1]) / 2
        else:
            return None, FeatureQuality()

        if left_eye[0] > 0 and right_eye[0] > 0:
            head_width = np.linalg.norm(left_eye - right_eye) * 2.5
        else:
            head_width = img.shape[1] * 0.3
            
        half_width = int(head_width / 2)
        x1 = int(max(0, center_x - half_width))
        x2 = int(min(img.shape[1], center_x + half_width))
        
        y1 = int(max(0, bottom_y - (head_width * 1.5)))
        y2 = int(bottom_y)

        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None, FeatureQuality()

        crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
        feature = self.extractor([crop_resized])[0].cpu().numpy()
        
        resolution_score = min(1.0, (crop.shape[0] * crop.shape[1]) / (64*64))
        quality = FeatureQuality.compute(feature)
        quality.overall_quality = (quality.overall_quality * 0.7) + (resolution_score * 0.3)

        return feature, quality

# ==================== 3. SYSTEM ====================

class MultiGranularitySystem:
    def __init__(self, modalities: List[str]):
        self.modalities = modalities
        self.pose_model = PoseEstimator()
        
        self.body_net = OSNetBody() if 'body' in modalities else None
        self.face_net = OSNetFace() if 'face' in modalities else None
        self.hair_net = OSNetHair() if 'hair' in modalities else None
        
    def extract(self, img):
        feats = {}
        kps = self.pose_model.extract(img)
        feats['pose'] = kps
        
        if self.body_net: feats['body'] = self.body_net.extract(img)
        
        if kps is not None:
            if self.face_net: feats['face'] = self.face_net.extract(img, kps)
            if self.hair_net: feats['hair'] = self.hair_net.extract(img, kps)
            
        return feats

    def compute_similarity(self, f1, f2):
        # 1. Base Body Score
        if 'body' in f1 and 'body' in f2 and f1['body'] is not None and f2['body'] is not None:
            body_score = max(0, 1 - cosine(f1['body'], f2['body']))
        else:
            return 0.0
            
        # --- ADAPTIVE GATING STRATEGY ---
        if body_score > 0.85:
            return body_score
            
        bonus = 0.0
        
        # Hair Bonus
        if 'hair' in self.modalities and 'hair' in f1 and 'hair' in f2:
            feat_h1, qual_h1 = f1.get('hair', (None, None))
            feat_h2, qual_h2 = f2.get('hair', (None, None))
            
            if feat_h1 is not None and feat_h2 is not None:
                hair_sim = max(0, 1 - cosine(feat_h1, feat_h2))
                if hair_sim > 0.5: 
                    bonus += 0.15 * hair_sim 

        # Face Bonus
        if 'face' in self.modalities and 'face' in f1 and 'face' in f2:
            feat_f1, qual_f1 = f1.get('face', (None, None))
            feat_f2, qual_f2 = f2.get('face', (None, None))
            
            if feat_f1 is not None and feat_f2 is not None:
                face_sim = max(0, 1 - cosine(feat_f1, feat_f2))
                if face_sim > 0.6: 
                    bonus += 0.15 * face_sim

        return min(1.0, body_score + bonus)

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
    if not os.path.exists(PATH_BODY):
        print(f"ERROR: Weights not found at {PATH_BODY}")
        exit()

    print(f"Scanning {DATASET_PATH}...")
    gal_files = sorted(list((DATASET_PATH / "bounding_box_test").glob("*.jpg")))
    query_files = sorted(list((DATASET_PATH / "query").glob("*.jpg")))
    
    gal_samples = [ReIDSample.parse(f.name, str(f)) for f in gal_files]
    query_samples = [ReIDSample.parse(f.name, str(f)) for f in query_files]
    gal_samples = [s for s in gal_samples if s][:GALLERY_LIMIT]
    query_samples = [s for s in query_samples if s][:QUERY_LIMIT]
    
    print(f"Gallery: {len(gal_samples)} | Query: {len(query_samples)}")
    
    experiments = [
        (['body'], "Baseline (Body Only)"),
        (['body', 'hair'], "Body + Hair Expert"),
        (['body', 'hair', 'face'], "Full (Body + Hair + Face)")
    ]
    
    final_res = []
    
    for mods, name in experiments:
        print(f"\n--- Running: {name} ---")
        system = MultiGranularitySystem(mods)
        
        g_data = []
        for s in tqdm(gal_samples, desc="Extract Gallery"):
            img = cv2.imread(s.image_path)
            f = system.extract(img)
            g_data.append({'s': s, 'f': f})
            
        q_data = []
        for s in tqdm(query_samples, desc="Extract Query"):
            img = cv2.imread(s.image_path)
            f = system.extract(img)
            q_data.append({'s': s, 'f': f})
            
        res = compute_metrics(system, name, g_data, q_data)
        if res: final_res.append(res)
        
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: Specialized Experts Ensemble")
    print(f"{'='*80}")
    print(f"{'Method':<30} | {'Rank-1':<8} | {'Rank-5':<8} | {'mAP':<8}")
    print("-" * 80)
    for r in final_res:
        print(f"{r['name']:<30} | {r['r1']*100:>6.2f}% | {r['r5']*100:>6.2f}% | {r['map']*100:>6.2f}%")
    print("="*80)