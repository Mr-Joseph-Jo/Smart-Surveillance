"""
Complete Multi-Modal Re-Identification Evaluation Script
Dataset: DukeMTMC-reID

Experiments:
1. Pose Only
2. Pose + Hair
3. Pose + Hair + Face
4. Pose + Hair + Face + Color
"""

import cv2
import numpy as np
import os
import re
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

# ==================== 1. DATA STRUCTURES (DukeMTMC-specific) ====================

@dataclass
class DukeMtmcSample:
    image_path: str
    person_id: int
    camera_id: int
    sequence_number: int
    frame_number: int
    
    @staticmethod
    def parse_filename(filename: str, full_path: str) -> Optional['DukeMtmcSample']:
        """
        Parse DukeMTMC filename format: 0002_c2_f0044155.jpg
        Format: personID_cameraID_framenumber.jpg
        Example: 0002_c2_f0044155.jpg -> person_id=2, camera_id=2, frame=44155
        """
        try:
            # Remove extension and split
            basename = filename.replace('.jpg', '')
            parts = basename.split('_')
            
            if len(parts) < 3:
                return None
                
            person_id = int(parts[0])  # e.g., 0002 -> 2
            camera_id = int(parts[1].replace('c', ''))  # e.g., c2 -> 2
            frame_str = parts[2].replace('f', '')  # e.g., f0044155 -> 0044155
            frame_number = int(frame_str)
            
            # Optional: Also track sequence (first part of frame number)
            sequence_number = int(frame_str[:2]) if len(frame_str) >= 2 else 0
            
            if person_id <= 0:  # Filter invalid IDs
                return None
                
            return DukeMtmcSample(
                image_path=full_path,
                person_id=person_id,
                camera_id=camera_id,
                sequence_number=sequence_number,
                frame_number=frame_number
            )
        except (ValueError, IndexError) as e:
            print(f"Error parsing {filename}: {e}")
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
            
            if len(kps) < 17 or np.mean(conf) < 0.3: return None

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
        
        # Smart split using Keypoints
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

        # WEIGHTS optimized for DukeMTMC (adjust based on dataset characteristics)
        # DukeMTMC has more outdoor scenes and varying lighting compared to Market1501
        self.raw_weights = {
            'pose': 0.10,    # Slightly higher weight due to more varied poses
            'hair': 0.10,    # Duke has more visible hair styles
            'face': 0.10,    # Faces can be clearer in some outdoor shots
            'color': 0.7    # Still important but lower due to lighting variations
        }
        
        # Normalize weights based on active modalities
        total = sum(self.raw_weights[m] for m in modalities)
        self.weights = {m: self.raw_weights[m]/total for m in modalities}
        print(f"Active Weights for DukeMTMC: {self.weights}")

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


# ==================== 4. DUKE-MTMC SPECIFIC EVALUATION ====================

class DukeMtmcEvaluator:
    def __init__(self, root):
        """
        DukeMTMC-reID directory structure:
        root/
        ├── query/          # Query images
        ├── gallery/        # Gallery images
        ├── train/          # Training images (not used in testing)
        └── bounding_box_test/ # Alternative naming
        """
        self.root = Path(root)
        
        # First try standard DukeMTMC folder structure
        if (self.root / "query").exists() and (self.root / "gallery").exists():
            print("Using standard DukeMTMC folder structure")
            self.gallery = self._load(self.root / "gallery")
            self.query = self._load(self.root / "query")
        elif (self.root / "bounding_box_test").exists() and (self.root / "query").exists():
            print("Using Market1501-style folder structure")
            self.gallery = self._load(self.root / "bounding_box_test")
            self.query = self._load(self.root / "query")
        else:
            raise FileNotFoundError(f"Could not find query/gallery folders in {root}")
        
        print(f"Loaded {len(self.gallery)} gallery samples")
        print(f"Loaded {len(self.query)} query samples")
        
        # Show sample distribution
        self._analyze_dataset()
        
    def _analyze_dataset(self):
        """Analyze dataset characteristics"""
        gallery_ids = set(s.person_id for s in self.gallery)
        query_ids = set(s.person_id for s in self.query)
        
        print(f"\nDataset Analysis:")
        print(f"Unique person IDs in gallery: {len(gallery_ids)}")
        print(f"Unique person IDs in query: {len(query_ids)}")
        print(f"Common IDs: {len(gallery_ids.intersection(query_ids))}")
        
        # Camera distribution
        gallery_cams = set(s.camera_id for s in self.gallery)
        query_cams = set(s.camera_id for s in self.query)
        print(f"Gallery cameras: {sorted(gallery_cams)}")
        print(f"Query cameras: {sorted(query_cams)}")
        
    def _load(self, p):
        s = []
        image_files = list(p.glob("*.jpg"))
        if not image_files:
            # Try subdirectories if direct folder is empty
            for subdir in p.iterdir():
                if subdir.is_dir():
                    image_files.extend(subdir.glob("*.jpg"))
        
        print(f"Loading from {p} - found {len(image_files)} images")
        
        for f in tqdm(image_files, desc=f"Loading {p.name}"):
            sample = DukeMtmcSample.parse_filename(f.name, str(f))
            if sample: 
                s.append(sample)
            else:
                print(f"Warning: Could not parse {f.name}")
        
        return s

    def run(self, system, name, gallery_limit=None, query_limit=None):
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")
        
        # Extract Gallery features
        gal_feats = []
        gallery_samples = self.gallery[:gallery_limit] if gallery_limit else self.gallery
        print(f"Extracting Gallery features from {len(gallery_samples)} samples...")
        
        for s in tqdm(gallery_samples, desc="Gallery"):
            img = cv2.imread(s.image_path)
            if img is None:
                continue
            f = system.extract(img)
            if any(v is not None for v in f.values()): 
                gal_feats.append({'s': s, 'f': f})
        
        print(f"Successfully extracted features from {len(gal_feats)}/{len(gallery_samples)} gallery images")
        
        # Extract Query features
        q_feats = []
        query_samples = self.query[:query_limit] if query_limit else self.query
        print(f"\nExtracting Query features from {len(query_samples)} samples...")
        
        for s in tqdm(query_samples, desc="Query"):
            img = cv2.imread(s.image_path)
            if img is None:
                continue
            f = system.extract(img)
            if any(v is not None for v in f.values()): 
                q_feats.append({'s': s, 'f': f})
        
        print(f"Successfully extracted features from {len(q_feats)}/{len(query_samples)} query images")
        
        if len(q_feats) == 0 or len(gal_feats) == 0:
            print("ERROR: No valid features extracted!")
            return None
        
        # Match queries against gallery
        print(f"\nMatching {len(q_feats)} queries against {len(gal_feats)} gallery samples...")
        results = []
        
        for q in tqdm(q_feats, desc="Matching"):
            matches = []
            for g in gal_feats:
                # Skip same image (same person, same camera, same sequence)
                if (q['s'].person_id == g['s'].person_id and 
                    q['s'].camera_id == g['s'].camera_id and
                    q['s'].sequence_number == g['s'].sequence_number):
                    continue
                
                sim = system.compute_sim(q['f'], g['f'])
                matches.append({
                    'person_id': g['s'].person_id,
                    'camera_id': g['s'].camera_id,
                    'match': q['s'].person_id == g['s'].person_id, 
                    'sim': sim
                })
            
            # Sort by similarity (descending)
            matches.sort(key=lambda x: x['sim'], reverse=True)
            results.append({
                'query_id': q['s'].person_id,
                'query_camera': q['s'].camera_id,
                'matches': matches,
                'top5_matches': matches[:5] if len(matches) >= 5 else matches,
                'top10_matches': matches[:10] if len(matches) >= 10 else matches
            })
        
        # Calculate metrics
        print("\nCalculating metrics...")
        
        # Rank-1: Correct person is at position 1
        r1 = np.mean([1 if r['matches'][0]['match'] else 0 for r in results if len(r['matches']) > 0])
        
        # Rank-5: Correct person is within top 5 positions
        r5 = np.mean([1 if any(m['match'] for m in r['top5_matches']) else 0 
                     for r in results if len(r['top5_matches']) > 0])
        
        # Rank-10: Correct person is within top 10 positions
        r10 = np.mean([1 if any(m['match'] for m in r['top10_matches']) else 0 
                      for r in results if len(r['top10_matches']) > 0])
        
        # mAP calculation
        aps = []
        for r in results:
            if len(r['matches']) == 0:
                continue
            corr, score = 0, 0.0
            for i, m in enumerate(r['matches']):
                if m['match']:
                    corr += 1
                    score += corr / (i + 1)
            if corr > 0:
                aps.append(score / corr)
        
        map_score = np.mean(aps) if aps else 0.0
        
        # Calculate CMC Curve (Top 20 ranks)
        cmc = np.zeros(20)
        for r in results:
            if len(r['matches']) == 0:
                continue
            for i in range(min(len(r['matches']), 20)):
                if r['matches'][i]['match']:
                    cmc[i:] += 1
                    break
        cmc = cmc / len(results) if len(results) > 0 else cmc

        # Cross-camera analysis
        cross_camera_r1 = 0
        cross_camera_count = 0
        for r in results:
            if len(r['matches']) == 0:
                continue
            # Check if top match is from different camera
            if r['matches'][0]['camera_id'] != r['query_camera']:
                cross_camera_count += 1
                if r['matches'][0]['match']:
                    cross_camera_r1 += 1
        
        cross_camera_r1_rate = cross_camera_r1 / cross_camera_count if cross_camera_count > 0 else 0

        return {
            'name': name, 
            'r1': r1, 
            'r5': r5,
            'r10': r10,
            'map': map_score, 
            'cmc': cmc,
            'cross_camera_r1': cross_camera_r1_rate,
            'num_queries': len(q_feats),
            'num_gallery': len(gal_feats)
        }


# ==================== 5. MAIN EXECUTION ====================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATASET_PATH = "./duke"  # Set your DukeMTMC path here
    # For faster testing, you can limit the number of samples
    GALLERY_LIMIT = 800  # Use 400 gallery samples for faster testing
    QUERY_LIMIT = None    # Set to e.g., 100 for faster testing, None for full dataset
    # ---------------------
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("\nExpected DukeMTMC-reID directory structure:")
        print("DukeMTMC-reID/")
        print("├── query/")
        print("├── gallery/")
        print("├── train/")
        print("└── (optionally) bounding_box_test/")
        
        # Try to create a placeholder structure
        print("\nYou can download DukeMTMC-reID from:")
        print("https://github.com/layumi/DukeMTMC-reID_evaluation")
    else:
        print(f"\n{'='*80}")
        print("DUKE-MTMC REID EVALUATION")
        print(f"{'='*80}")
        print(f"Dataset path: {DATASET_PATH}")
        print(f"Gallery limit: {GALLERY_LIMIT or 'Full dataset'}")
        print(f"Query limit: {QUERY_LIMIT or 'Full dataset'}")
        print(f"{'='*80}")
        
        evaluator = DukeMtmcEvaluator(DATASET_PATH)
        
        # Run experiments
        results_list = []
        
        # 1. Pose Only
        try:
            r1 = evaluator.run(
                MultiModalReID(['pose']), 
                "Pose Only",
                gallery_limit=GALLERY_LIMIT,
                query_limit=QUERY_LIMIT
            )
            if r1: results_list.append(r1)
        except Exception as e:
            print(f"Error in Pose Only: {e}")
        
        # 2. Pose + Hair
        try:
            r2 = evaluator.run(
                MultiModalReID(['pose', 'hair']), 
                "Pose + Hair",
                gallery_limit=GALLERY_LIMIT,
                query_limit=QUERY_LIMIT
            )
            if r2: results_list.append(r2)
        except Exception as e:
            print(f"Error in Pose + Hair: {e}")
        
        # 3. Pose + Hair + Face
        try:
            r3 = evaluator.run(
                MultiModalReID(['pose', 'hair', 'face']), 
                "Pose + Hair + Face",
                gallery_limit=GALLERY_LIMIT,
                query_limit=QUERY_LIMIT
            )
            if r3: results_list.append(r3)
        except Exception as e:
            print(f"Error in Pose + Hair + Face: {e}")
        
        # 4. Pose + Hair + Face + Color
        try:
            r4 = evaluator.run(
                MultiModalReID(['pose', 'hair', 'face', 'color']), 
                "Pose + Hair + Face + Color",
                gallery_limit=GALLERY_LIMIT,
                query_limit=QUERY_LIMIT
            )
            if r4: results_list.append(r4)
        except Exception as e:
            print(f"Error in Full Model: {e}")
        
        # ==========================================
        # VISUALIZATION & OUTPUTS
        # ==========================================
        
        if not results_list:
            print("No results to display!")
        else:
            # 1. Print Text Table with multiple metrics
            print(f"\n{'='*100}")
            print("FINAL ABLATION STUDY RESULTS - DukeMTMC-reID")
            print(f"{'='*100}")
            print(f"{'Experiment':<30} | {'Rank-1':<8} | {'Rank-5':<8} | {'Rank-10':<8} | {'mAP':<8} | {'X-Cam R1':<8} | {'Queries':<8}")
            print("-" * 100)
            for res in results_list:
                print(f"{res['name']:<30} | "
                      f"{res['r1']*100:>6.2f}% | "
                      f"{res['r5']*100:>6.2f}% | "
                      f"{res['r10']*100:>6.2f}% | "
                      f"{res['map']*100:>6.2f}% | "
                      f"{res['cross_camera_r1']*100:>6.2f}% | "
                      f"{res['num_queries']:>8}")
            print("="*100)
            
            print("\nKey Definitions:")
            print("- Rank-1: Highest confidence match is correct (most strict)")
            print("- Rank-5: Correct match appears in top 5 highest confidence predictions")
            print("- Rank-10: Correct match appears in top 10 highest confidence predictions")
            print("- mAP: Mean Average Precision (considers all ranks)")
            print("- X-Cam R1: Rank-1 accuracy for cross-camera matches only")
            
            # 2. Generate CMC Curve Plot 
            plt.figure(figsize=(12, 7))
            markers = ['^', 's', 'o', '*', 'D', 'X']
            colors = ['gray', 'blue', 'orange', 'red', 'green', 'purple']
            
            for idx, res in enumerate(results_list[:len(markers)]):
                plt.plot(range(1, 21), res['cmc'] * 100, 
                         label=res['name'], 
                         marker=markers[idx], 
                         color=colors[idx],
                         linewidth=2,
                         markersize=8)
            
            plt.title('CMC Curve: Ablation Study - DukeMTMC-reID', fontsize=14, fontweight='bold')
            plt.xlabel('Rank', fontsize=12)
            plt.ylabel('Matching Accuracy (%)', fontsize=12)
            plt.legend(fontsize=10, loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(1, 21))
            plt.ylim(0, 105)
            
            # Add Rank-1, Rank-5, and Rank-10 markers
            plt.axvline(x=1, color='black', linestyle=':', alpha=0.3, label='Rank-1')
            plt.axvline(x=5, color='black', linestyle=':', alpha=0.3, label='Rank-5')
            plt.axvline(x=10, color='black', linestyle=':', alpha=0.3, label='Rank-10')
            
            plt.savefig('dukemtmc_cmc_curve.png', dpi=300, bbox_inches='tight')
            print(f"\n[Graph] Saved CMC curve to dukemtmc_cmc_curve.png")
            
           