"""
Complete Multi-Modal Re-Identification Evaluation Script
Dataset: Market-1501

Experiments:
1. Pose Only
2. Pose + Hair
3. Pose + Hair + Face
4. Pose + Hair + Face + Color (Full Multi-modal)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Pose extraction will fail.")

# ==================== 1. DATA STRUCTURES ====================

@dataclass
class Market1501Sample:
    image_path: str
    person_id: int
    camera_id: int
    
    @staticmethod
    def parse_filename(filename: str, full_path: str) -> Optional['Market1501Sample']:
        try:
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) < 2: return None
            person_id = int(parts[0])
            camera_id = int(parts[1][1])
            if person_id <= 0: return None # Filter junk (-1 and 0)
            return Market1501Sample(image_path=full_path, person_id=person_id, camera_id=camera_id)
        except:
            return None

@dataclass
class PoseFeatures:
    features: Dict[str, float]
    keypoints: np.ndarray

@dataclass
class HairFeatures:
    color_hist: np.ndarray
    texture_var: float

@dataclass
class FaceFeatures:
    color_hist: np.ndarray
    lbp_hist: np.ndarray

@dataclass
class ColorFeatures:
    top_hist: np.ndarray
    bottom_hist: np.ndarray
    top_dominant: List[Tuple]
    bottom_dominant: List[Tuple]
    top_pct: List[float]
    bottom_pct: List[float]

# ==================== 2. FEATURE EXTRACTORS ====================

class PoseFeatureExtractor:
    """Extract structural features using YOLOv8-Pose"""
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            import torch
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading YOLOv8-Pose on {self.device}...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        self.weights = {'leg_torso': 1.5, 'arm_torso': 1.3, 'shoulder_hip': 1.2}

    def extract_features(self, image: np.ndarray) -> Optional[PoseFeatures]:
        if image is None: return None
        try:
            results = self.pose_model(image, verbose=False, device=self.device)
            if not results or results[0].keypoints is None: return None
            
            kps = results[0].keypoints.xy[0].cpu().numpy()
            conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
            
            if len(kps) < 17 or np.mean(conf) < 0.4: return None

            # Simple geometric features
            feats = {}
            def d(i, j): return np.linalg.norm(kps[i] - kps[j])
            
            torso_h = d(5, 11) + d(6, 12)
            if torso_h > 0:
                leg_len = (d(11, 13) + d(13, 15)) / 2
                feats['leg_torso'] = np.clip(leg_len / torso_h / 2.5, 0, 1) # Normalize
                
                hip_w = d(11, 12)
                sh_w = d(5, 6)
                if hip_w > 0: feats['shoulder_hip'] = np.clip(sh_w / hip_w / 2.0, 0, 1)

            return PoseFeatures(features=feats, keypoints=kps)
        except:
            return None

    def compute_similarity(self, f1: PoseFeatures, f2: PoseFeatures) -> float:
        if not f1 or not f2: return 0.0
        score, total_w = 0.0, 0.0
        for k in f1.features:
            if k in f2.features:
                w = self.weights.get(k, 1.0)
                score += w * (1 - abs(f1.features[k] - f2.features[k]))
                total_w += w
        return score / total_w if total_w > 0 else 0.0


class HairFeatureExtractor:
    """Extract Hair Color and Texture"""
    def __init__(self):
        self.bins = [16, 4, 4] # Compact HSV
    
    def extract_features(self, image: np.ndarray, kps: np.ndarray) -> Optional[HairFeatures]:
        if image is None or kps is None: return None
        # Heuristic: Hair is above nose/eyes
        face_pts = kps[:5] 
        valid = face_pts[face_pts[:,0] > 0]
        if len(valid) == 0: return None
        
        min_y = np.min(valid[:,1])
        h, w = image.shape[:2]
        
        # Crop top region relative to face
        hair_roi = image[max(0, int(min_y-h*0.15)):int(min_y), :]
        if hair_roi.size == 0: return None
        
        # Color Hist
        hsv = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Texture
        gray = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2GRAY)
        return HairFeatures(color_hist=hist, texture_var=np.var(gray)/2500.0)

    def compute_similarity(self, f1: HairFeatures, f2: HairFeatures) -> float:
        if not f1 or not f2: return 0.0
        eps = 1e-10
        # Chi2 distance
        d = 0.5 * np.sum(((f1.color_hist - f2.color_hist)**2) / (f1.color_hist + f2.color_hist + eps))
        return 1 / (1 + d)


class FaceFeatureExtractor:
    """Extract Face Color and LBP Texture"""
    def extract_features(self, image: np.ndarray, kps: np.ndarray) -> Optional[FaceFeatures]:
        if image is None or kps is None: return None
        face_kps = kps[:5]
        valid = face_kps[face_kps[:,0] > 0]
        if len(valid) < 3: return None
        
        x_min, y_min = np.min(valid, axis=0)
        x_max, y_max = np.max(valid, axis=0)
        pad = 5
        face_roi = image[max(0, int(y_min-pad)):int(y_max+pad), max(0, int(x_min-pad)):int(x_max+pad)]
        
        if face_roi.size == 0: return None
        
        # Color
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Gradient Hist (LBP proxy)
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, _ = cv2.cartToPolar(gx, gy)
        lbp_hist, _ = np.histogram(mag, bins=10, range=(0, 255))
        lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-6)
        
        return FaceFeatures(color_hist=hist, lbp_hist=lbp_hist)

    def compute_similarity(self, f1: FaceFeatures, f2: FaceFeatures) -> float:
        if not f1 or not f2: return 0.0
        eps = 1e-10
        d_col = 0.5 * np.sum(((f1.color_hist - f2.color_hist)**2) / (f1.color_hist + f2.color_hist + eps))
        d_tex = 0.5 * np.sum(((f1.lbp_hist - f2.lbp_hist)**2) / (f1.lbp_hist + f2.lbp_hist + eps))
        return 0.6 * (1/(1+d_col)) + 0.4 * (1/(1+d_tex))


class ColorFeatureExtractor:
    """Extract Top/Bottom Clothing Color"""
    def __init__(self):
        self.bins = [30, 8, 8]
    
    def extract_features(self, image: np.ndarray, kps: np.ndarray) -> Optional[ColorFeatures]:
        if image is None: return None
        h, w = image.shape[:2]
        
        # Smart split
        split_y = h // 2
        if kps is not None:
            shoulders_y = (kps[5][1] + kps[6][1]) / 2
            hips_y = (kps[11][1] + kps[12][1]) / 2
            if shoulders_y > 0 and hips_y > 0: split_y = int((shoulders_y + hips_y) / 2)
            
        top = image[max(0, int(h*0.1)):split_y, :]
        bot = image[split_y:min(h, int(h*0.9)), :]
        
        if top.size == 0 or bot.size == 0: return None
        
        def get_color(roi):
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Simple dominant color
            pixels = roi.reshape(-1, 3)
            if len(pixels) > 500: pixels = pixels[::2] # Subsample
            pixels = (pixels // 64) * 64 + 32 # Quantize
            unq, cnt = np.unique(pixels, axis=0, return_counts=True)
            idx = np.argsort(cnt)[::-1][:3]
            return hist, [tuple(c) for c in unq[idx]], (cnt[idx]/cnt.sum()).tolist()
            
        t_hist, t_dom, t_pct = get_color(top)
        b_hist, b_dom, b_pct = get_color(bot)
        
        return ColorFeatures(top_hist=t_hist, bottom_hist=b_hist, 
                             top_dominant=t_dom, bottom_dominant=b_dom,
                             top_pct=t_pct, bottom_pct=b_pct)

    def compute_similarity(self, f1: ColorFeatures, f2: ColorFeatures) -> float:
        if not f1 or not f2: return 0.0
        eps = 1e-10
        # Hist Sim
        s_top = 1 / (1 + 0.5*np.sum(((f1.top_hist - f2.top_hist)**2)/(f1.top_hist + f2.top_hist + eps)))
        s_bot = 1 / (1 + 0.5*np.sum(((f1.bottom_hist - f2.bottom_hist)**2)/(f1.bottom_hist + f2.bottom_hist + eps)))
        
        # Dominant Color Sim
        def d_sim(d1, p1, d2, p2):
            score = 0
            for c1, pc1 in zip(d1, p1):
                best = 0
                for c2, pc2 in zip(d2, p2):
                    dist = np.linalg.norm(np.array(c1)-np.array(c2))
                    best = max(best, 1/(1+dist/50))
                score += best * pc1
            return score
            
        s_dom_t = d_sim(f1.top_dominant, f1.top_pct, f2.top_dominant, f2.top_pct)
        s_dom_b = d_sim(f1.bottom_dominant, f1.bottom_pct, f2.bottom_dominant, f2.bottom_pct)
        
        return 0.35*s_top + 0.35*s_bot + 0.15*s_dom_t + 0.15*s_dom_b


# ==================== 3. FUSION SYSTEM ====================

class MultiModalReID:
    def __init__(self, modalities: List[str]):
        self.modalities = modalities
        self.extractors = {}
        
        # Init necessary extractors
        if any(m in modalities for m in ['pose', 'hair', 'face', 'color']):
            self.extractors['pose'] = PoseFeatureExtractor() # Always needed for KPs
        if 'hair' in modalities: self.extractors['hair'] = HairFeatureExtractor()
        if 'face' in modalities: self.extractors['face'] = FaceFeatureExtractor()
        if 'color' in modalities: self.extractors['color'] = ColorFeatureExtractor()

        # UPDATED WEIGHTS FOR BETTER COLOR PERFORMANCE
        self.raw_weights = {
            'pose': 0.15,
            'hair': 0.10,
            'face': 0.05,
            'color': 0.70  # Color boosted significantly
        }
        
        total = sum(self.raw_weights[m] for m in modalities)
        self.weights = {m: self.raw_weights[m]/total for m in modalities}

    def extract(self, img):
        feats = {}
        # 1. Pose (Base)
        p_feat = self.extractors['pose'].extract_features(img)
        kps = p_feat.keypoints if p_feat else None
        if 'pose' in self.modalities: feats['pose'] = p_feat
        
        # 2. Others
        if 'hair' in self.modalities: feats['hair'] = self.extractors['hair'].extract_features(img, kps)
        if 'face' in self.modalities: feats['face'] = self.extractors['face'].extract_features(img, kps)
        if 'color' in self.modalities: feats['color'] = self.extractors['color'].extract_features(img, kps)
        return feats

    def compute_sim(self, f1, f2):
        score = 0.0
        for m in self.modalities:
            if m in f1 and m in f2:
                s = self.extractors[m].compute_similarity(f1[m], f2[m])
                score += s * self.weights[m]
        return score

# ==================== 4. EVALUATION & MAIN ====================

class MarketEvaluator:
    def __init__(self, root):
        self.root = Path(root)
        self.gallery = self._load(self.root / "bounding_box_test")
        self.query = self._load(self.root / "query")
        
    def _load(self, p):
        s = []
        for f in p.glob("*.jpg"):
            sample = Market1501Sample.parse_filename(f.name, str(f))
            if sample: s.append(sample)
        return s

    def run(self, system, name):
        print(f"\n--- Experiment: {name} ---")
        
        # Extract Gallery (Subset for speed, remove slice [:N] for full)
        gal_feats = []
        print("Extracting Gallery...")
        for s in tqdm(self.gallery[:400]): 
            img = cv2.imread(s.image_path)
            f = system.extract(img)
            if any(v is not None for v in f.values()): gal_feats.append({'s': s, 'f': f})
            
        # Extract Query
        q_feats = []
        print("Extracting Query...")
        for s in tqdm(self.query[:50]): 
            img = cv2.imread(s.image_path)
            f = system.extract(img)
            if any(v is not None for v in f.values()): q_feats.append({'s': s, 'f': f})
            
        # Match
        print("Matching...")
        results = []
        for q in q_feats:
            matches = []
            for g in gal_feats:
                if q['s'].person_id == g['s'].person_id and q['s'].camera_id == g['s'].camera_id: continue
                sim = system.compute_sim(q['f'], g['f'])
                matches.append({'match': q['s'].person_id == g['s'].person_id, 'sim': sim})
            matches.sort(key=lambda x: x['sim'], reverse=True)
            results.append(matches)
            
        # Metrics
        r1 = np.mean([1 if r[0]['match'] else 0 for r in results])
        
        aps = []
        for r in results:
            corr, score = 0, 0.0
            for i, m in enumerate(r):
                if m['match']:
                    corr += 1
                    score += corr / (i+1)
            aps.append(score/corr if corr else 0)
        
        return {'name': name, 'r1': r1, 'map': np.mean(aps)}

if __name__ == "__main__":
    DATASET_PATH = "./Market-1501-v15.09.15" # UPDATE THIS PATH
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
    else:
        evaluator = MarketEvaluator(DATASET_PATH)
        
        # 1. Pose Only
        r1 = evaluator.run(MultiModalReID(['pose']), "Pose Only")
        
        # 2. Pose + Hair
        r2 = evaluator.run(MultiModalReID(['pose', 'hair']), "Pose + Hair")
        
        # 3. Pose + Hair + Face
        r3 = evaluator.run(MultiModalReID(['pose', 'hair', 'face']), "Pose + Hair + Face")
        
        # 4. Full Multi-modal
        r4 = evaluator.run(MultiModalReID(['pose', 'hair', 'face', 'color']), "Full Multi-modal")
        
        # Final Summary
        print("\n" + "="*60)
        print("FINAL ABLATION STUDY RESULTS")
        print("="*60)
        print(f"{'Experiment':<30} | {'Rank-1':<10} | {'mAP':<10}")
        print("-" * 60)
        for res in [r1, r2, r3, r4]:
            print(f"{res['name']:<30} | {res['r1']*100:>6.2f}%   | {res['map']*100:>6.2f}%")
        print("="*60)