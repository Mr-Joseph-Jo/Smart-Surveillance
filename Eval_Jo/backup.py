"""
Complete Multi-Modal Re-Identification Evaluation Script
Dataset: Market-1501

Experiments:
1. Segmented Body Only
2. Body + Pose
3. Body + Pose + Face (Gated)
4. Full Gated Hybrid
"""

import cv2
import numpy as np
import os
import random
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
    confidences: np.ndarray

@dataclass
class HairFeatures:
    color_hist: np.ndarray
    texture_var: float

@dataclass
class FaceFeatures:
    color_hist: np.ndarray
    lbp_hist: np.ndarray

@dataclass
class BodyRegionFeatures:
    region_hists: Dict[str, np.ndarray]

# ==================== 2. FEATURE EXTRACTORS ====================

class PoseFeatureExtractor:
    """Extract structural features using YOLOv8-Pose"""
    def __init__(self, device: str = 'auto'):
        import torch
        if device == 'auto':
            self.device = 0 if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading YOLOv8-Pose on {'cuda:0' if self.device == 0 else 'cpu'}...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        if torch.cuda.is_available() and self.device == 0:
            self.pose_model.to('cuda')
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

            return PoseFeatures(features=feats, keypoints=kps, confidences=conf)
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
    def extract_features(self, image: np.ndarray, kps: np.ndarray, conf: Optional[np.ndarray] = None) -> Optional[FaceFeatures]:
        if image is None or kps is None: return None
        if conf is None or len(conf) < 5: return None
        face_conf = conf[:5]
        if np.mean(face_conf) <= 0.6: return None

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


class SegmentedBodyExtractor:
    """Use YOLO segmentation to isolate body regions and extract HSV histograms."""

    def __init__(self, weights: str = 'yolov8n-seg.pt', device: str = 'auto', mask_threshold: float = 0.5):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is required for SegmentedBodyExtractor. Please install ultralytics before running the evaluator.")

        self.bins = [30, 8, 8]
        self.mask_threshold = mask_threshold
        import torch
        if device == 'auto':
            self.device = 0 if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading YOLO segmentation model on {'cuda:0' if self.device == 0 else 'cpu'}...")
        self.seg_model = YOLO(weights)
        if torch.cuda.is_available() and self.device == 0:
            self.seg_model.to('cuda')

    def _region_hist(self, hsv_img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        if mask is None or cv2.countNonZero(mask) == 0:
            return None
        hist = cv2.calcHist([hsv_img], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_features(self, image: np.ndarray) -> Optional[BodyRegionFeatures]:
        if image is None: return None

        results = self.seg_model(image, verbose=False, device=self.device)
        if not results or results[0].masks is None:
            return None

        res = results[0]
        masks = res.masks.data.cpu().numpy()
        if len(masks) == 0:
            return None

        target_idx = 0
        if res.boxes is not None and len(res.boxes.cls) == len(masks):
            classes = res.boxes.cls.tolist()
            confidences = res.boxes.conf.tolist()
            person_indices = [i for i, c in enumerate(classes) if int(c) == 0]
            if person_indices:
                target_idx = max(person_indices, key=lambda i: confidences[i])

        mask = masks[target_idx]
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        binary_mask = (mask > self.mask_threshold).astype(np.uint8)
        if cv2.countNonZero(binary_mask) == 0:
            return None

        ys = np.where(binary_mask > 0)[0]
        if len(ys) < 3:
            return None

        head_cut, torso_cut = np.percentile(ys, [33.3, 66.6])
        head_cut, torso_cut = int(head_cut), int(torso_cut)

        rows = np.arange(binary_mask.shape[0])[:, None]
        mask_bool = binary_mask.astype(bool)
        head_mask = np.where(np.logical_and(mask_bool, rows <= head_cut), 255, 0).astype(np.uint8)
        torso_mask = np.where(np.logical_and(mask_bool, np.logical_and(rows > head_cut, rows <= torso_cut)), 255, 0).astype(np.uint8)
        leg_mask = np.where(np.logical_and(mask_bool, rows > torso_cut), 255, 0).astype(np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        region_hists = {}
        for name, region_mask in [('head', head_mask), ('torso', torso_mask), ('legs', leg_mask)]:
            hist = self._region_hist(hsv, region_mask)
            if hist is not None:
                region_hists[name] = hist

        return BodyRegionFeatures(region_hists=region_hists) if region_hists else None

    def compute_similarity(self, f1: BodyRegionFeatures, f2: BodyRegionFeatures) -> float:
        if not f1 or not f2:
            return 0.0

        shared = set(f1.region_hists.keys()) & set(f2.region_hists.keys())
        if not shared:
            return 0.0

        eps = 1e-10
        sims = []
        for region in shared:
            h1 = f1.region_hists[region]
            h2 = f2.region_hists[region]
            d = 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))
            sims.append(1 / (1 + d))

        return float(np.mean(sims)) if sims else 0.0


# ==================== 3. FUSION SYSTEM ====================

class MultiModalReID:
    def __init__(self, modalities: List[str]):
        self.modalities = modalities
        self.extractors = {}
        
        # Body extractor is now the always-on base modality
        self.extractors['body'] = SegmentedBodyExtractor()

        # Pose is required whenever any keypoint-dependent modality is requested
        pose_needed = any(m in modalities for m in ['pose', 'hair', 'face'])
        if pose_needed:
            self.extractors['pose'] = PoseFeatureExtractor()

        if 'hair' in modalities:
            self.extractors['hair'] = HairFeatureExtractor()
        if 'face' in modalities:
            self.extractors['face'] = FaceFeatureExtractor()

    def extract(self, img):
        feats = {}

        p_feat = None
        if 'pose' in self.extractors:
            p_feat = self.extractors['pose'].extract_features(img)
            if 'pose' in self.modalities:
                feats['pose'] = p_feat

        kps = p_feat.keypoints if p_feat else None
        conf = p_feat.confidences if p_feat else None

        if 'hair' in self.modalities and 'hair' in self.extractors:
            feats['hair'] = self.extractors['hair'].extract_features(img, kps)
        if 'face' in self.extractors:
            feats['face'] = self.extractors['face'].extract_features(img, kps, conf)

        # Body features are always present for downstream gating
        feats['body'] = self.extractors['body'].extract_features(img)
        return feats

    def compute_sim(self, f1, f2):
        body_f1, body_f2 = f1.get('body'), f2.get('body')
        if body_f1 is None or body_f2 is None:
            return 0.0

        body_score = self.extractors['body'].compute_similarity(body_f1, body_f2)

        face_ready = (
            'face' in self.extractors and
            f1.get('face') is not None and
            f2.get('face') is not None
        )

        if face_ready:
            face_score = self.extractors['face'].compute_similarity(f1['face'], f2['face'])
            return 0.8 * body_score + 0.2 * face_score

        # Gate closed → body similarity carries 100% weight
        return body_score

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
        
        # Extract Gallery
        gal_feats = []
        print("Extracting Gallery...")
        for s in tqdm(self.gallery): 
            img = cv2.imread(s.image_path)
            f = system.extract(img)
            if any(v is not None for v in f.values()): gal_feats.append({'s': s, 'f': f})
            
        # Extract Query
        q_feats = []
        print("Extracting Query...")
        for s in tqdm(self.query): 
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

    def visualize_retrieval(self, system, num_queries: int = 3, top_k: int = 5, out_dir: str = "debug_results"):
        if not self.gallery or not self.query:
            print("Visualization skipped: missing gallery or query samples.")
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        gallery_subset = self.gallery
        gallery_feats = []
        print("Preparing gallery features for visualization...")
        for sample in tqdm(gallery_subset, desc="Viz Gallery", leave=False):
            img = cv2.imread(sample.image_path)
            if img is None:
                continue
            feats = system.extract(img)
            if not any(v is not None for v in feats.values()):
                continue
            gallery_feats.append({'s': sample, 'f': feats, 'img': cv2.cvtColor(img, cv2.COLOR_BGR2RGB)})

        if not gallery_feats:
            print("Visualization skipped: no gallery features extracted.")
            return

        chosen = random.sample(self.query, min(num_queries, len(self.query)))
        saved = 0
        for sample in chosen:
            img = cv2.imread(sample.image_path)
            if img is None:
                continue
            feats = system.extract(img)
            if not any(v is not None for v in feats.values()):
                continue

            matches = []
            for gal in gallery_feats:
                if sample.person_id == gal['s'].person_id and sample.camera_id == gal['s'].camera_id:
                    continue
                sim = system.compute_sim(feats, gal['f'])
                matches.append({
                    'sim': sim,
                    'correct': sample.person_id == gal['s'].person_id,
                    'img': gal['img'],
                    'pid': gal['s'].person_id
                })

            if not matches:
                continue

            matches.sort(key=lambda m: m['sim'], reverse=True)
            top_matches = matches[:min(top_k, len(matches))]

            fig, axes = plt.subplots(1, len(top_matches) + 1, figsize=(3 * (len(top_matches) + 1), 4))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            axes = axes.flatten()
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Query ID {sample.person_id}")
            axes[0].axis('off')

            for idx, match in enumerate(top_matches, start=1):
                ax = axes[idx]
                ax.imshow(match['img'])
                ax.axis('off')
                ax.set_title(f"ID {match['pid']}\nSim {match['sim']:.2f}")
                border_color = 'green' if match['correct'] else 'red'
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(4)

            out_file = out_path / f"result_query_{sample.person_id}.png"
            fig.tight_layout()
            fig.savefig(out_file, bbox_inches='tight')
            plt.close(fig)
            saved += 1

        print(f"Saved {saved} retrieval visualization(s) to {out_path}.")

if __name__ == "__main__":
    DATASET_PATH = "./Market-1501-v15.09.15" # UPDATE THIS PATH
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
    else:
        evaluator = MarketEvaluator(DATASET_PATH)
        
        # Prepare systems for each experiment
        body_only_system = MultiModalReID(['body'])
        system = MultiModalReID(['body', 'face'])
        body_pose_face_system = MultiModalReID(['body', 'pose', 'face'])
        full_system = MultiModalReID(['body', 'pose', 'face', 'hair'])

        # 1. Segmented Body Only
        r1 = evaluator.run(body_only_system, "Segmented Body Only")
        
        # 2. Body + Face (Gated)
        r2 = evaluator.run(system, "Body + Face (Gated)")
        
        # 3. Body + Pose + Face
        r3 = evaluator.run(body_pose_face_system, "Body + Pose + Face")
        
        # 4. Full Gated Hybrid
        r4 = evaluator.run(full_system, "Full Gated Hybrid")
        
        # Final Summary
        print("\n" + "="*60)
        print("FINAL ABLATION STUDY RESULTS")
        print("="*60)
        print(f"{'Experiment':<30} | {'Rank-1':<10} | {'mAP':<10}")
        print("-" * 60)
        for res in [r1, r2, r3, r4]:
            print(f"{res['name']:<30} | {res['r1']*100:>6.2f}%   | {res['map']*100:>6.2f}%")
        print("="*60)

        Path("debug_results").mkdir(parents=True, exist_ok=True)
        print("\nGenerating retrieval visualizations...")
        evaluator.visualize_retrieval(system)