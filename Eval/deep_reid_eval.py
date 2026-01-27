"""
Deep Multi-Modal Re-Identification Evaluation Script
Dataset: DukeMTMC-reID
Paper: "Ablation Study of Part-Based Deep Embeddings for Person Re-ID"
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from PIL import Image
from scipy.spatial.distance import cosine
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt
import torchreid
from torchreid.utils import FeatureExtractor

# Check for GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 1. DATA STRUCTURES ====================

@dataclass
class ReIDSample:
    """Generic Parser for both Market-1501 and DukeMTMC"""
    image_path: str
    person_id: int
    camera_id: int
    
    @staticmethod
    def parse(filename: str, full_path: str):
        # Market-1501: 0001_c1s1_001051_00.jpg
        # DukeMTMC:    0002_c2_f0044155.jpg
        try:
            # -1 is a garbage label in Market/Duke (junk images)
            if filename.startswith('-1') or filename.startswith('0000'):
                return None

            parts = filename.replace('.jpg', '').split('_')
            pid = int(parts[0])
            cam_str = parts[1]  # e.g., "c1s1" or "c2"
            
            # Logic to handle both "c1s1" and "c2" formats
            if 's' in cam_str:
                # Market style: extract number between 'c' and 's'
                cid = int(cam_str.split('s')[0].replace('c', ''))
            else:
                # Duke style: just remove 'c'
                cid = int(cam_str.replace('c', ''))
                
            return ReIDSample(full_path, pid, cid)
        except: 
            return None

# ==================== 2. DEEP LEARNING MODELS ====================

class DeepFeatureExtractor:
    def __init__(self, model_name):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, img_crop):
        if img_crop is None or img_crop.size == 0: return None
        img_pil = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        return self.transform(img_pil).unsqueeze(0).to(DEVICE)

# class FaceModel(DeepFeatureExtractor):
#     def __init__(self):
#         super().__init__("FaceNet")
#         # Suppress the pickle warning by loading explicitly if needed, but standard load is fine here
#         self.net = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    
#     def extract(self, img, kps):
#         if kps is None: return None
#         face_pts = kps[:5]
#         valid = face_pts[face_pts[:,0] > 0]
#         if len(valid) < 2: return None
        
#         x1, y1 = np.min(valid, axis=0)
#         x2, y2 = np.max(valid, axis=0)
#         h, w = img.shape[:2]
#         pad = int((x2-x1) * 0.5) 
#         crop = img[max(0, int(y1-pad)):min(h, int(y2+pad)), 
#                    max(0, int(x1-pad)):min(w, int(x2+pad))]
        
#         tensor = self.preprocess(crop)
#         if tensor is None: return None
#         with torch.no_grad():
#             emb = self.net(tensor)
#         return emb.cpu().numpy().flatten()

# class HairModel(DeepFeatureExtractor):
#     def __init__(self):
#         super().__init__("ResNet18-Hair")
#         base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         self.net = nn.Sequential(*list(base.children())[:-1]).to(DEVICE).eval()
        
#     def extract(self, img, kps):
#         if kps is None: return None
#         nose = kps[0]
#         if nose[0] == 0: return None
#         crop = img[0:int(nose[1]), :]
#         tensor = self.preprocess(crop)
#         if tensor is None: return None
#         with torch.no_grad():
#             emb = self.net(tensor)
#         return emb.cpu().numpy().flatten()
# ==================== UPDATED MODELS (ALL OSNET) ====================

class FaceModel(DeepFeatureExtractor):
    """
    NOW USING: OSNet for Face
    Hypothesis: In low-res surveillance, a ReID model (texture matcher) 
    might perform better than a specific Face Recognition model.
    """
    def __init__(self):
        # We use a separate OSNet instance just for faces
        print("Loading OSNet (Face Branch)...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='./log/osnet_duke/model/model.pth.tar-50', 
            device=DEVICE,
            image_size=(128, 128) # Faces are square-ish, so we adjust input size
        )
    
    def extract(self, img, kps):
        if kps is None: return None
        face_pts = kps[:5]
        valid = face_pts[face_pts[:,0] > 0]
        if len(valid) < 2: return None
        
        # Crop logic
        x1, y1 = np.min(valid, axis=0)
        x2, y2 = np.max(valid, axis=0)
        h, w = img.shape[:2]
        pad = int((x2-x1) * 0.5) 
        crop = img[max(0, int(y1-pad)):min(h, int(y2+pad)), 
                   max(0, int(x1-pad)):min(w, int(x2+pad))]
        
        if crop.size == 0: return None
        
        # OSNet Extract
        features = self.extractor([crop])
        return features[0].cpu().numpy()

class HairModel(DeepFeatureExtractor):
    """
    NOW USING: OSNet for Hair
    Hypothesis: OSNet captures fine-grained hair texture better than ResNet18.
    """
    def __init__(self):
        print("Loading OSNet (Hair Branch)...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='./log/osnet_duke/model/model.pth.tar-50',
            device=DEVICE,
            image_size=(128, 256) # Hair is wide and short
        )
        
    def extract(self, img, kps):
        if kps is None: return None
        nose = kps[0]
        if nose[0] == 0: return None
        
        # Crop Top (Hair)
        crop = img[0:int(nose[1]), :]
        if crop.size == 0: return None
        
        # OSNet Extract
        features = self.extractor([crop])
        return features[0].cpu().numpy()

class BodyColorModel(DeepFeatureExtractor):
    def __init__(self):
        super().__init__("ResNet50-Body")
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*list(base.children())[:-1]).to(DEVICE).eval()
        
    def extract(self, img):
        tensor = self.preprocess(img)
        if tensor is None: return None
        with torch.no_grad():
            emb = self.net(tensor)
        return emb.cpu().numpy().flatten()
class OSNetModel(DeepFeatureExtractor):
    """
    State-of-the-Art ReID Model (OSNet)
    Pre-trained on Market-1501 for high accuracy.
    """
    def __init__(self):
        # We don't call super() here because torchreid handles its own transforms
        print("Loading OSNet (Ain_x1_0) pre-trained on Market-1501...")
        
        # This utility wrapper handles resizing, normalization, and inference automatically
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',  # 'ain' is a newer, faster version of OSNet
            model_path='./log/osnet_duke/model/model.pth.tar-50',                # Empty string = download generic ImageNet/Market weights
            device=DEVICE,
            image_size=(256, 128)         # Standard ReID input size
        )
        
    def extract(self, img):
        if img is None or img.size == 0: return None
        
        # torchreid expects a list of images (even if just one)
        # It handles BGR -> RGB conversion internally usually, but safe to pass straight CV2
        # It returns a tensor, so we flatten it to numpy
        features = self.extractor([img])
        return features[0].cpu().numpy()
    
class MaskedBodyModel(DeepFeatureExtractor):
    """
    Experimental: Uses YOLOv8-Seg to mask out background before Deep Feature Extraction.
    Hypothesis: Removing background noise will improve ReID on DukeMTMC.
    """
    def __init__(self):
        super().__init__("ResNet50-Masked")
        print("Loading YOLOv8-Seg for Background Removal...")
        self.seg_model = YOLO('yolov8n-seg.pt') # Segmentation model
        
        print("Loading ResNet50 for Body Features...")
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*list(base.children())[:-1]).to(DEVICE).eval()
        
    def extract(self, img):
        if img is None: return None
        
        # 1. Run Segmentation
        # We use a low confidence threshold to ensure we catch the person
        results = self.seg_model(img, verbose=False, conf=0.3, device=0 if DEVICE=='cuda' else 'cpu')
        
        masked_img = img.copy() # Default to original if segmentation fails
        
        if results and results[0].masks is not None:
            # Get the mask of the person with highest confidence
            # (Assumes the largest/most confident mask is the person of interest)
            masks = results[0].masks.data.cpu().numpy() # Shape: (N, H, W)
            
            # Resize mask to match original image size (YOLO masks are smaller)
            mask = masks[0] # Get first detection
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            # Binary mask (0 or 1)
            mask = (mask > 0.5).astype(np.uint8)
            
            # Apply Mask: Multiply image by mask (Background becomes Black)
            # Expand mask to 3 channels for BGR multiplication
            mask_3ch = np.stack([mask]*3, axis=-1)
            masked_img = img * mask_3ch

        # 2. Extract Deep Features from the Masked Image
        tensor = self.preprocess(masked_img)
        if tensor is None: return None
        
        with torch.no_grad():
            emb = self.net(tensor)
        return emb.cpu().numpy().flatten()

class PoseEstimator:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
        
    def extract(self, img):
        if img is None: return None
        # Safe inference
        results = self.model(img, verbose=False, device=0 if DEVICE=='cuda' else 'cpu')
        
        if not results: return None
        res = results[0]
        
        # FIX: Check if keypoints exist and are not empty
        if res.keypoints is None or res.keypoints.xy.numel() == 0 or res.keypoints.xy.shape[0] == 0:
            return None
            
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

# ==================== 3. SYSTEM ORCHESTRATOR ====================

class DeepReIDSystem:
    def __init__(self, modalities: List[str]):
        self.modalities = modalities
        # Shared Models (loaded once if possible, but simpler to reload for ablation script)
        self.pose_model = PoseEstimator()
        
        # Load only what is needed for this experiment
        self.face_net = FaceModel() if 'face' in modalities else None
        self.hair_net = HairModel() if 'hair' in modalities else None
        if 'color' in modalities:
            #self.body_net = BodyColorModel() # OLD
            #self.body_net = MaskedBodyModel()  # NEW (Segmentation + ResNet)
            self.body_net = OSNetModel()      # SOTA ReID Model
        else:
            self.body_net = None
        
        # Weights
        self.weights = {'pose': 0.05, 'hair': 0.1, 'face': 0.15, 'color': 0.7}
        
    def extract(self, img):
        feats = {}
        # 1. Pose (Always needed for crops)
        kps = self.pose_model.extract(img)
        feats['pose'] = kps
        
        # If we need geometric crops but pose failed, we can't get hair/face
        if kps is None and ('face' in self.modalities or 'hair' in self.modalities):
            pass 
        else:
            if self.face_net: feats['face'] = self.face_net.extract(img, kps)
            if self.hair_net: feats['hair'] = self.hair_net.extract(img, kps)
            
        if self.body_net: feats['color'] = self.body_net.extract(img)
        return feats

    def compute_similarity(self, f1, f2):
        score, total_w = 0.0, 0.0
        for m in self.modalities:
            if m not in f1 or m not in f2: continue
            if f1[m] is None or f2[m] is None: continue
            
            w = self.weights[m]
            if m == 'pose':
                sim = self.pose_model.compare(f1[m], f2[m])
            else:
                sim = max(0, 1 - cosine(f1[m], f2[m]))
            
            score += sim * w
            total_w += w
        return score / total_w if total_w > 0 else 0.0

# ==================== 4. EVALUATION & METRICS ====================

class MetricsCalculator:
    def compute(self, system, name, gallery_data, query_data):
        # Match queries against gallery
        results = []
        
        for q in tqdm(query_data, desc=f"Matching ({name})"):
            matches = []
            for g in gallery_data:
                # Skip junk (same ID, same camera)
                if q['s'].person_id == g['s'].person_id and q['s'].camera_id == g['s'].camera_id:
                    continue
                
                sim = system.compute_similarity(q['f'], g['f'])
                matches.append({
                    'pid': g['s'].person_id,
                    'cam': g['s'].camera_id,
                    'match': q['s'].person_id == g['s'].person_id,
                    'sim': sim
                })
            
            matches.sort(key=lambda x: x['sim'], reverse=True)
            results.append({'qid': q['s'].person_id, 'qcam': q['s'].camera_id, 'matches': matches})

        # Calculate Statistics
        if not results: return None
        
        # Ranks
        r1 = np.mean([1 if r['matches'] and r['matches'][0]['match'] else 0 for r in results])
        r5 = np.mean([1 if any(m['match'] for m in r['matches'][:5]) else 0 for r in results])
        r10 = np.mean([1 if any(m['match'] for m in r['matches'][:10]) else 0 for r in results])
        
        # mAP
        aps = []
        for r in results:
            y_true = [m['match'] for m in r['matches']]
            if not any(y_true): continue
            
            cumulative_correct = 0
            precision_sum = 0
            for i, match in enumerate(y_true):
                if match:
                    cumulative_correct += 1
                    precision_sum += cumulative_correct / (i + 1)
            aps.append(precision_sum / sum(y_true))
        map_score = np.mean(aps) if aps else 0.0
        
        # Cross-Camera R1
        xcam_correct = 0
        xcam_total = 0
        for r in results:
            # Filter matches to different cameras
            xcam_matches = [m for m in r['matches'] if m['cam'] != r['qcam']]
            if xcam_matches:
                xcam_total += 1
                if xcam_matches[0]['match']:
                    xcam_correct += 1
        xcam_r1 = xcam_correct / xcam_total if xcam_total > 0 else 0.0
        
        # CMC Curve
        cmc = np.zeros(20)
        for r in results:
            for i in range(min(len(r['matches']), 20)):
                if r['matches'][i]['match']:
                    cmc[i:] += 1
                    break
        cmc /= len(results)

        return {
            'name': name, 'r1': r1, 'r5': r5, 'r10': r10, 
            'map': map_score, 'xcam_r1': xcam_r1, 'cmc': cmc,
            'queries': len(results)
        }

# ==================== 5. MAIN EXECUTION ====================

if __name__ == "__main__":
    print(f"Running on Device: {DEVICE}")
    
    # --- CONFIGURATION ---
    # Update to your actual Market-1501 path
    DATASET_PATH = Path("C:\\Users\\abela\\Downloads\\archive (3)\\Market-1501-v15.09.15") 
    #DATASET_PATH = Path("./duke/dukemtmc-reid/DukeMTMC-reID")  # Set your DukeMTMC path here
    
    # Market is large (19k+ images). 
    # We parse ALL first, then select a subset for speed.
    GALLERY_LIMIT = 800  
    QUERY_LIMIT = 100     
    # ---------------------

    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        print("Ensure structure is: /bounding_box_test and /query")
        exit()

    print(f"Scanning dataset: {DATASET_PATH.name}...")
    
    gal_path = DATASET_PATH / "bounding_box_test"
    query_path = DATASET_PATH / "query"
    
    # 1. Get ALL file paths first (Do not limit yet)
    print("Reading file lists...")
    gallery_files = sorted(list(gal_path.glob("*.jpg")))
    query_files = sorted(list(query_path.glob("*.jpg")))
    
    # 2. Parse everything
    print(f"Parsing {len(gallery_files)} gallery files (this may take a moment)...")
    gallery_samples = [ReIDSample.parse(f.name, str(f)) for f in gallery_files]
    query_samples = [ReIDSample.parse(f.name, str(f)) for f in query_files]
    
    # 3. Filter out None (Junk images starting with -1 or 0000)
    gallery_samples = [s for s in gallery_samples if s is not None]
    query_samples = [s for s in query_samples if s is not None]

    # 4. NOW apply the limit to valid samples only
    if GALLERY_LIMIT:
        gallery_samples = gallery_samples[:GALLERY_LIMIT]
    if QUERY_LIMIT:
        query_samples = query_samples[:QUERY_LIMIT]

    print(f"Active Gallery Samples: {len(gallery_samples)}")
    print(f"Active Query Samples: {len(query_samples)}")
    
    if len(gallery_samples) == 0:
        print("ERROR: No valid gallery samples found! Check your path or parser logic.")
        exit()

    # Define Ablations
    experiments = [
        (['pose'], "Pose Only"),
        (['pose', 'hair'], "Pose + Hair"),
        (['pose', 'hair', 'face'], "Pose + Hair + Face"),
        (['pose', 'hair', 'face', 'color'], "Pose + Hair + Face + Color")
    ]
    
    final_results = []
    calculator = MetricsCalculator()

    # Run Loop
    for mods, name in experiments:
        print(f"\n--- Experiment: {name} ---")
        system = DeepReIDSystem(mods)
        
        # Extract Features
        gal_data = []
        for s in tqdm(gallery_samples, desc="Gallery"):
            img = cv2.imread(s.image_path)
            f = system.extract(img)
            if f: gal_data.append({'s': s, 'f': f})
            
        q_data = []
        for s in tqdm(query_samples, desc="Query"):
            img = cv2.imread(s.image_path)
            f = system.extract(img)
            if f: q_data.append({'s': s, 'f': f})
            
        # Calculate Metrics
        res = calculator.compute(system, name, gal_data, q_data)
        if res: final_results.append(res)

    # --- OUTPUT GENERATION ---
    print(f"\n{'='*110}")
    print("FINAL ABLATION STUDY RESULTS - market (Deep Learning)")
    print(f"{'='*110}")
    print(f"{'Experiment':<30} | {'Rank-1':<8} | {'Rank-5':<8} | {'Rank-10':<8} | {'mAP':<8} | {'X-Cam R1':<8} | {'Queries':<8}")
    print("-" * 110)
    
    for r in final_results:
        print(f"{r['name']:<30} | "
              f"{r['r1']*100:>6.2f}% | "
              f"{r['r5']*100:>6.2f}% | "
              f"{r['r10']*100:>6.2f}% | "
              f"{r['map']*100:>6.2f}% | "
              f"{r['xcam_r1']*100:>6.2f}% | "
              f"{r['queries']:>8}")
    print("="*110)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    markers = ['^', 's', 'o', '*']
    for i, r in enumerate(final_results):
        plt.plot(range(1, 21), r['cmc']*100, label=r['name'], marker=markers[i], linewidth=2)
    
    plt.title('CMC Curve: Deep Learning Ablation Study (dukeMTMC-reID)')
    plt.xlabel('Rank')
    plt.ylabel('Matching Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(range(1, 21))
    plt.ylim(0, 105)
    
    out_file = 'market1501_results.png'
    plt.savefig(out_file)
    print(f"\n[Graph] CMC Curve saved to {out_file}")