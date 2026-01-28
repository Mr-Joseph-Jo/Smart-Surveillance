"""
Multi-Granularity Fusion System for Person Re-ID (Advanced Version)
Addresses: Feature quality estimation, smart fusion, normalization, debugging
Author: Research Implementation
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import rankdata
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torchreid
from torchreid.utils import FeatureExtractor
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = Path("C:/Users/abela/Documents/GitHub/Smart-Surveillance/Eval/duke/dukemtmc-reid/DukeMTMC-reID")

# YOUR TRAINED MODEL PATHS - UPDATE THESE!
PATH_BODY = 'C:/Users/abela/Documents/GitHub/Smart-Surveillance/Eval/log/osnet_duke/model/model.pth.tar-50'
PATH_HAIR = 'C:/Users/abela/Documents/GitHub/Smart-Surveillance/Eval/log/osnet_duke_hair/model/model.pth.tar-40' 
PATH_FACE = 'C:/Users/abela/Documents/GitHub/Smart-Surveillance/Eval/log/osnet_duke_face/model/model.pth.tar-40'

# Evaluation settings
GALLERY_LIMIT = None  # None for full evaluation
QUERY_LIMIT = None
BATCH_SIZE = 32  # For faster feature extraction

# ==================== DATA STRUCTURES ====================

@dataclass
class ReIDSample:
    image_path: str
    person_id: int
    camera_id: int
    
    @staticmethod
    def parse(filename: str, full_path: str):
        try:
            if filename.startswith('-1') or filename.startswith('0000'):
                return None
            parts = filename.replace('.jpg', '').split('_')
            pid = int(parts[0])
            cam_str = parts[1]
            if 's' in cam_str:
                cid = int(cam_str.split('s')[0].replace('c', ''))
            else:
                cid = int(cam_str.replace('c', ''))
            return ReIDSample(full_path, pid, cid)
        except:
            return None

@dataclass
class FeatureQuality:
    """Estimates quality/confidence of extracted features"""
    detection_confidence: float = 0.0
    feature_variance: float = 0.0
    spatial_coverage: float = 0.0
    overall_quality: float = 0.0
    
    @staticmethod
    def compute(feature: np.ndarray, keypoints: Optional[np.ndarray] = None, 
                keypoint_indices: Optional[List[int]] = None) -> 'FeatureQuality':
        quality = FeatureQuality()
        
        if feature is None:
            return quality
            
        # Feature variance (low variance = low information)
        quality.feature_variance = float(np.var(feature))
        
        # Keypoint confidence
        if keypoints is not None and keypoint_indices is not None:
            relevant_kps = keypoints[keypoint_indices]
            # Check visibility (non-zero coordinates)
            visible = np.sum(relevant_kps[:, 0] > 0)
            quality.detection_confidence = visible / len(keypoint_indices)
            
            # Spatial coverage (spread of keypoints)
            if visible > 1:
                coords = relevant_kps[relevant_kps[:, 0] > 0]
                bbox_area = (coords[:, 0].max() - coords[:, 0].min()) * \
                           (coords[:, 1].max() - coords[:, 1].min())
                quality.spatial_coverage = min(1.0, bbox_area / (256 * 128))
        
        # Overall quality score
        quality.overall_quality = (
            0.4 * quality.detection_confidence +
            0.3 * min(1.0, quality.feature_variance / 0.1) +
            0.3 * quality.spatial_coverage
        )
        
        return quality

# ==================== FEATURE EXTRACTORS ====================

class PoseEstimator:
    def __init__(self):
        print("Loading YOLOv8-Pose...")
        self.model = YOLO('yolov8n-pose.pt')
        
    def extract(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img is None or img.size == 0:
            return None
            
        results = self.model(img, verbose=False, device=DEVICE)
        if not results or len(results) == 0:
            return None
            
        res = results[0]
        if res.keypoints is None or res.keypoints.xy.shape[0] == 0:
            return None
            
        # Get the most confident detection
        kps = res.keypoints.xy[0].cpu().numpy()
        conf = res.keypoints.conf[0].cpu().numpy()
        
        # Filter out low confidence detections
        if np.mean(conf) < 0.3:
            return None
            
        return kps

class OSNetExtractor:
    """Generic OSNet extractor with quality estimation"""
    
    def __init__(self, model_path: str, model_name: str, image_size: Tuple[int, int]):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        print(f"Loading {model_name} from {model_path}...")
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=model_path,
            device=DEVICE,
            image_size=image_size
        )
        self.model_name = model_name
        
    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of images"""
        valid_images = [img for img in images if img is not None and img.size > 0]
        if not valid_images:
            return None
            
        features = self.extractor(valid_images)
        return features.cpu().numpy()
    
    def extract_single(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract feature from single image"""
        if image is None or image.size == 0:
            return None
        features = self.extract_features([image])
        return features[0] if features is not None else None

class BodyExtractor(OSNetExtractor):
    def __init__(self, model_path: str):
        super().__init__(model_path, "Body-Global", (256, 128))
        
    def extract(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], FeatureQuality]:
        feature = self.extract_single(img)
        quality = FeatureQuality.compute(feature)
        quality.detection_confidence = 1.0  # Full body always available
        quality.spatial_coverage = 1.0
        quality.overall_quality = 0.9  # High confidence for global features
        return feature, quality

class FaceExtractor(OSNetExtractor):
    def __init__(self, model_path: str):
        super().__init__(model_path, "Face-Local", (128, 128))
        
    def extract(self, img: np.ndarray, keypoints: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], FeatureQuality]:
        if keypoints is None:
            return None, FeatureQuality()
            
        # Face keypoints: nose, left_eye, right_eye, left_ear, right_ear (0-4)
        face_kps = keypoints[:5]
        valid_kps = face_kps[face_kps[:, 0] > 0]
        
        if len(valid_kps) < 2:
            return None, FeatureQuality()
            
        # Compute bounding box with padding
        x1, y1 = np.min(valid_kps, axis=0)
        x2, y2 = np.max(valid_kps, axis=0)
        
        # Adaptive padding based on face size
        face_width = x2 - x1
        face_height = y2 - y1
        pad_x = int(face_width * 0.3)
        pad_y = int(face_height * 0.5)
        
        h, w = img.shape[:2]
        y1_crop = max(0, int(y1 - pad_y))
        y2_crop = min(h, int(y2 + pad_y))
        x1_crop = max(0, int(x1 - pad_x))
        x2_crop = min(w, int(x2 + pad_x))
        
        crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None, FeatureQuality()
            
        feature = self.extract_single(crop)
        quality = FeatureQuality.compute(feature, keypoints, list(range(5)))
        
        return feature, quality

class HairExtractor(OSNetExtractor):
    def __init__(self, model_path: str):
        super().__init__(model_path, "Hair-Local", (128, 128))
        
    def extract(self, img: np.ndarray, keypoints: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], FeatureQuality]:
        if keypoints is None:
            return None, FeatureQuality()
            
        # Use nose and eyes to determine hair region
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        
        if nose[0] == 0 or (left_eye[0] == 0 and right_eye[0] == 0):
            return None, FeatureQuality()
            
        # Calculate top of head
        if left_eye[0] > 0 and right_eye[0] > 0:
            eye_y = (left_eye[1] + right_eye[1]) / 2
            eye_x = (left_eye[0] + right_eye[0]) / 2
        else:
            eye_y = nose[1]
            eye_x = nose[0]
            
        # Hair region: from top of image to just above eyes
        h, w = img.shape[:2]
        y_top = 0
        y_bottom = int(eye_y - 10)  # Slightly above eyes
        
        if y_bottom <= y_top or y_bottom - y_top < 20:
            return None, FeatureQuality()
            
        # Add horizontal margin
        x_margin = int(w * 0.1)
        crop = img[y_top:y_bottom, x_margin:w-x_margin]
        
        if crop.size == 0 or crop.shape[0] < 20:
            return None, FeatureQuality()
            
        feature = self.extract_single(crop)
        quality = FeatureQuality.compute(feature, keypoints, [0, 1, 2])
        
        return feature, quality

# ==================== FUSION STRATEGIES ====================

class FusionStrategy:
    """Base class for fusion strategies"""
    
    def __init__(self, name: str):
        self.name = name
        
    def compute_similarity(self, features_q: Dict, features_g: Dict) -> float:
        raise NotImplementedError

class WeightedSumFusion(FusionStrategy):
    """Traditional weighted sum with L2 normalization"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__("Weighted Sum")
        self.weights = weights
        
    def compute_similarity(self, fq: Dict, fg: Dict) -> float:
        if 'body' not in fq or 'body' not in fg:
            return 0.0
            
        score = 0.0
        total_weight = 0.0
        
        for modality in ['body', 'hair', 'face']:
            if modality in self.weights and modality in fq and modality in fg:
                feat_q = fq[modality]
                feat_g = fg[modality]
                
                if feat_q is not None and feat_g is not None:
                    # L2 normalize
                    feat_q = feat_q / (np.linalg.norm(feat_q) + 1e-8)
                    feat_g = feat_g / (np.linalg.norm(feat_g) + 1e-8)
                    
                    # Cosine similarity
                    sim = 1 - cosine(feat_q, feat_g)
                    score += self.weights[modality] * max(0, sim)
                    total_weight += self.weights[modality]
        
        return score / total_weight if total_weight > 0 else 0.0

class QualityAwareFusion(FusionStrategy):
    """Fusion that adapts weights based on feature quality"""
    
    def __init__(self, base_weights: Dict[str, float], quality_threshold: float = 0.5):
        super().__init__("Quality-Aware")
        self.base_weights = base_weights
        self.quality_threshold = quality_threshold
        
    def compute_similarity(self, fq: Dict, fg: Dict) -> float:
        if 'body' not in fq or 'body' not in fg:
            return 0.0
            
        score = 0.0
        total_weight = 0.0
        
        for modality in ['body', 'hair', 'face']:
            if modality not in self.base_weights:
                continue
                
            feat_q = fq.get(f'{modality}_feat')
            feat_g = fg.get(f'{modality}_feat')
            qual_q = fq.get(f'{modality}_quality')
            qual_g = fg.get(f'{modality}_quality')
            
            if feat_q is None or feat_g is None:
                continue
                
            # Compute similarity
            feat_q = feat_q / (np.linalg.norm(feat_q) + 1e-8)
            feat_g = feat_g / (np.linalg.norm(feat_g) + 1e-8)
            sim = 1 - cosine(feat_q, feat_g)
            
            # Quality-based weight adjustment
            if qual_q and qual_g:
                min_quality = min(qual_q.overall_quality, qual_g.overall_quality)
                
                # Only use feature if quality is above threshold
                if min_quality > self.quality_threshold:
                    adaptive_weight = self.base_weights[modality] * min_quality
                    score += adaptive_weight * max(0, sim)
                    total_weight += adaptive_weight
            else:
                # Fallback to base weight
                score += self.base_weights[modality] * max(0, sim)
                total_weight += self.base_weights[modality]
        
        return score / total_weight if total_weight > 0 else 0.0

class RankFusion(FusionStrategy):
    """Rank-level fusion (Borda count)"""
    
    def __init__(self):
        super().__init__("Rank Fusion")
        self.gallery_features = None
        self.query_features = None
        
    def set_data(self, gallery_features: List[Dict], query_features: List[Dict]):
        self.gallery_features = gallery_features
        self.query_features = query_features
        
    def compute_all_similarities(self) -> np.ndarray:
        """Compute similarity matrix for rank-based fusion"""
        n_queries = len(self.query_features)
        n_gallery = len(self.gallery_features)
        
        # Compute similarity matrices for each modality
        modalities = ['body', 'hair', 'face']
        sim_matrices = {}
        
        for mod in modalities:
            sim_mat = np.zeros((n_queries, n_gallery))
            
            for i, fq in enumerate(self.query_features):
                feat_q = fq.get(f'{mod}_feat')
                if feat_q is None:
                    continue
                feat_q = feat_q / (np.linalg.norm(feat_q) + 1e-8)
                
                for j, fg in enumerate(self.gallery_features):
                    feat_g = fg.get(f'{mod}_feat')
                    if feat_g is None:
                        continue
                    feat_g = feat_g / (np.linalg.norm(feat_g) + 1e-8)
                    
                    sim_mat[i, j] = 1 - cosine(feat_q, feat_g)
            
            sim_matrices[mod] = sim_mat
        
        # Convert to ranks (higher rank = better match)
        rank_matrices = {}
        for mod, sim_mat in sim_matrices.items():
            rank_mat = np.zeros_like(sim_mat)
            for i in range(n_queries):
                # Rank gallery items for this query
                ranks = rankdata(sim_mat[i], method='average')
                rank_mat[i] = ranks
            rank_matrices[mod] = rank_mat
        
        # Combine ranks with weights
        weights = {'body': 0.5, 'hair': 0.25, 'face': 0.25}
        final_ranks = np.zeros((n_queries, n_gallery))
        
        for mod, rank_mat in rank_matrices.items():
            final_ranks += weights[mod] * rank_mat
        
        # Convert back to similarities (higher rank = higher similarity)
        final_sims = final_ranks / n_gallery
        
        return final_sims
        
    def compute_similarity(self, fq: Dict, fg: Dict) -> float:
        # This method not used for rank fusion
        return 0.0

class AdaptiveGatingFusion(FusionStrategy):
    """Smart gating: use local features only when global is uncertain"""
    
    def __init__(self, body_confidence_threshold: float = 0.70,
                 local_min_similarity: float = 0.55):
        super().__init__("Adaptive Gating")
        self.body_threshold = body_confidence_threshold
        self.local_threshold = local_min_similarity
        
    def compute_similarity(self, fq: Dict, fg: Dict) -> float:
        feat_body_q = fq.get('body_feat')
        feat_body_g = fg.get('body_feat')
        
        if feat_body_q is None or feat_body_g is None:
            return 0.0
            
        # Normalize and compute body similarity
        feat_body_q = feat_body_q / (np.linalg.norm(feat_body_q) + 1e-8)
        feat_body_g = feat_body_g / (np.linalg.norm(feat_body_g) + 1e-8)
        body_sim = max(0, 1 - cosine(feat_body_q, feat_body_g))
        
        # HIGH CONFIDENCE: Trust body features alone
        if body_sim > self.body_threshold:
            return body_sim
            
        # LOW CONFIDENCE: Add local features with strict quality control
        bonus = 0.0
        
        # Hair contribution
        feat_hair_q = fq.get('hair_feat')
        feat_hair_g = fg.get('hair_feat')
        qual_hair_q = fq.get('hair_quality')
        qual_hair_g = fg.get('hair_quality')
        
        if (feat_hair_q is not None and feat_hair_g is not None and
            qual_hair_q and qual_hair_g):
            if min(qual_hair_q.overall_quality, qual_hair_g.overall_quality) > 0.5:
                feat_hair_q = feat_hair_q / (np.linalg.norm(feat_hair_q) + 1e-8)
                feat_hair_g = feat_hair_g / (np.linalg.norm(feat_hair_g) + 1e-8)
                hair_sim = max(0, 1 - cosine(feat_hair_q, feat_hair_g))
                
                if hair_sim > self.local_threshold:
                    bonus += 0.10 * hair_sim
        
        # Face contribution (stricter)
        feat_face_q = fq.get('face_feat')
        feat_face_g = fg.get('face_feat')
        qual_face_q = fq.get('face_quality')
        qual_face_g = fg.get('face_quality')
        
        if (feat_face_q is not None and feat_face_g is not None and
            qual_face_q and qual_face_g):
            if min(qual_face_q.overall_quality, qual_face_g.overall_quality) > 0.6:
                feat_face_q = feat_face_q / (np.linalg.norm(feat_face_q) + 1e-8)
                feat_face_g = feat_face_g / (np.linalg.norm(feat_face_g) + 1e-8)
                face_sim = max(0, 1 - cosine(feat_face_q, feat_face_g))
                
                if face_sim > 0.65:  # Very strict
                    bonus += 0.10 * face_sim
        
        return min(1.0, body_sim + bonus)

# ==================== MAIN SYSTEM ====================

class MultiGranularitySystem:
    def __init__(self, use_hair: bool = True, use_face: bool = True):
        print("\n" + "="*80)
        print("Initializing Multi-Granularity Person Re-ID System")
        print("="*80)
        
        # Initialize extractors
        self.pose_model = PoseEstimator()
        self.body_extractor = BodyExtractor(PATH_BODY)
        
        self.use_hair = use_hair
        self.use_face = use_face
        
        if use_hair:
            self.hair_extractor = HairExtractor(PATH_HAIR)
        if use_face:
            self.face_extractor = FaceExtractor(PATH_FACE)
            
        print("All extractors loaded successfully!\n")
        
    def extract_features(self, img: np.ndarray) -> Dict:
        """Extract all features from an image"""
        features = {}
        
        # Body features (always)
        body_feat, body_quality = self.body_extractor.extract(img)
        features['body_feat'] = body_feat
        features['body_quality'] = body_quality
        
        # Pose keypoints
        keypoints = self.pose_model.extract(img)
        features['keypoints'] = keypoints
        
        # Local features (if keypoints detected)
        if keypoints is not None:
            if self.use_hair:
                hair_feat, hair_quality = self.hair_extractor.extract(img, keypoints)
                features['hair_feat'] = hair_feat
                features['hair_quality'] = hair_quality
                
            if self.use_face:
                face_feat, face_quality = self.face_extractor.extract(img, keypoints)
                features['face_feat'] = face_feat
                features['face_quality'] = face_quality
        
        return features

# ==================== EVALUATION ====================

def compute_metrics(gallery_data: List[Dict], query_data: List[Dict], 
                   fusion_strategy: FusionStrategy, name: str) -> Dict:
    """Compute Rank-1, Rank-5, Rank-10, and mAP"""
    
    print(f"\nEvaluating: {name}")
    print("-" * 80)
    
    # Special handling for rank fusion
    if isinstance(fusion_strategy, RankFusion):
        fusion_strategy.set_data(
            [g['features'] for g in gallery_data],
            [q['features'] for q in query_data]
        )
        sim_matrix = fusion_strategy.compute_all_similarities()
    else:
        sim_matrix = None
    
    results = []
    
    for q_idx, q in enumerate(tqdm(query_data, desc="Matching")):
        matches = []
        
        for g_idx, g in enumerate(gallery_data):
            # Skip same camera same person
            if (q['sample'].person_id == g['sample'].person_id and
                q['sample'].camera_id == g['sample'].camera_id):
                continue
                
            # Compute similarity
            if sim_matrix is not None:
                sim = sim_matrix[q_idx, g_idx]
            else:
                sim = fusion_strategy.compute_similarity(q['features'], g['features'])
                
            matches.append({
                'match': q['sample'].person_id == g['sample'].person_id,
                'sim': sim,
                'pid': g['sample'].person_id
            })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['sim'], reverse=True)
        results.append(matches)
    
    if not results:
        return None
    
    # Compute metrics
    r1 = np.mean([1 if r[0]['match'] else 0 for r in results if r])
    r5 = np.mean([1 if any(m['match'] for m in r[:5]) else 0 for r in results if r])
    r10 = np.mean([1 if any(m['match'] for m in r[:10]) else 0 for r in results if r])
    
    # mAP
    aps = []
    for r in results:
        y_true = [m['match'] for m in r]
        if not any(y_true):
            continue
            
        cumulative_correct = 0
        precision_sum = 0
        for i, match in enumerate(y_true):
            if match:
                cumulative_correct += 1
                precision_sum += cumulative_correct / (i + 1)
        
        ap = precision_sum / sum(y_true)
        aps.append(ap)
    
    map_score = np.mean(aps) if aps else 0.0
    
    return {
        'name': name,
        'r1': r1,
        'r5': r5,
        'r10': r10,
        'map': map_score
    }

def load_dataset(split: str, limit: Optional[int] = None) -> List[ReIDSample]:
    """Load gallery or query samples"""
    if split == 'gallery':
        folder = DATASET_PATH / "bounding_box_test"
    else:
        folder = DATASET_PATH / "query"
        
    files = sorted(list(folder.glob("*.jpg")))
    samples = []
    
    for f in files:
        sample = ReIDSample.parse(f.name, str(f))
        if sample:
            samples.append(sample)
            
    if limit:
        samples = samples[:limit]
        
    return samples

def extract_all_features(samples: List[ReIDSample], system: MultiGranularitySystem) -> List[Dict]:
    """Extract features for all samples"""
    data = []
    
    for sample in tqdm(samples, desc="Extracting features"):
        img = cv2.imread(sample.image_path)
        if img is None:
            continue
            
        features = system.extract_features(img)
        data.append({
            'sample': sample,
            'features': features
        })
    
    return data

# ==================== MAIN ====================

def main():
    print("\n" + "="*80)
    print("MULTI-GRANULARITY PERSON RE-ID EVALUATION")
    print("="*80)
    
    # Verify model paths
    for name, path in [("Body", PATH_BODY), ("Hair", PATH_HAIR), ("Face", PATH_FACE)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} model not found at {path}")
            print("Please update the model paths in the configuration section.")
            return
    
    # Load dataset
    print("\nLoading dataset...")
    gallery_samples = load_dataset('gallery', GALLERY_LIMIT)
    query_samples = load_dataset('query', QUERY_LIMIT)
    
    print(f"Gallery: {len(gallery_samples)} samples")
    print(f"Query: {len(query_samples)} samples")
    
    # Define experiments
    experiments = [
        {
            'name': 'Baseline (Body Only)',
            'use_hair': False,
            'use_face': False,
            'fusion': None
        },
        {
            'name': 'Body + Hair (Weighted Sum)',
            'use_hair': True,
            'use_face': False,
            'fusion': WeightedSumFusion({'body': 0.7, 'hair': 0.3})
        },
        {
            'name': 'Body + Face (Weighted Sum)',
            'use_hair': False,
            'use_face': True,
            'fusion': WeightedSumFusion({'body': 0.7, 'face': 0.3})
        },
        {
            'name': 'Full (Weighted Sum 0.6/0.2/0.2)',
            'use_hair': True,
            'use_face': True,
            'fusion': WeightedSumFusion({'body': 0.6, 'hair': 0.2, 'face': 0.2})
        },
        {
            'name': 'Full (Quality-Aware Fusion)',
            'use_hair': True,
            'use_face': True,
            'fusion': QualityAwareFusion({'body': 0.6, 'hair': 0.2, 'face': 0.2}, 0.5)
        },
        {
            'name': 'Full (Adaptive Gating)',
            'use_hair': True,
            'use_face': True,
            'fusion': AdaptiveGatingFusion(0.70, 0.55)
        },
    ]
    
    all_results = []
    
    for exp in experiments:
        print("\n" + "="*80)
        print(f"Experiment: {exp['name']}")
        print("="*80)
        
        # Initialize system
        system = MultiGranularitySystem(
            use_hair=exp['use_hair'],
            use_face=exp['use_face']
        )
        
        # Extract features
        gallery_data = extract_all_features(gallery_samples, system)
        query_data = extract_all_features(query_samples, system)
        
        # Evaluate
        if exp['fusion'] is None:
            # Baseline: body only
            fusion = WeightedSumFusion({'body': 1.0})
        else:
            fusion = exp['fusion']
            
        metrics = compute_metrics(gallery_data, query_data, fusion, exp['name'])
        
        if metrics:
            all_results.append(metrics)
            print(f"\nResults:")
            print(f"  Rank-1: {metrics['r1']*100:.2f}%")
            print(f"  Rank-5: {metrics['r5']*100:.2f}%")
            print(f"  Rank-10: {metrics['r10']*100:.2f}%")
            print(f"  mAP: {metrics['map']*100:.2f}%")
    
    # Print final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<40} | {'Rank-1':<8} | {'Rank-5':<8} | {'Rank-10':<8} | {'mAP':<8}")
    print("-" * 80)
    
    for r in all_results:
        print(f"{r['name']:<40} | {r['r1']*100:>6.2f}% | {r['r5']*100:>6.2f}% | "
              f"{r['r10']*100:>6.2f}% | {r['map']*100:>6.2f}%")
    
    print("="*80)
    
    # Save results
    import json
    results_file = 'multi_granularity_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
