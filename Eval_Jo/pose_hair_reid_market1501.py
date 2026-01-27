"""
Multi-Modal Person Re-Identification for Market-1501
Modalities: Pose + Hair + Face + Color
Evaluation: CMC curve, mAP, Rank-1, Rank-5, Rank-10

Research Paper Experiments:
1. Pose only
2. Pose + Hair
3. Pose + Hair + Face
4. Pose + Hair + Face + Color (Full Multi-modal)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean
from collections import defaultdict
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics required! Install: pip install ultralytics")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, using fallback clustering")


# ==================== POSE-BASED REID ====================

@dataclass
class PoseFeatures:
    """Pose-based features for ReID"""
    features: Dict[str, float]
    keypoints: np.ndarray
    quality_score: float


class PoseFeatureExtractor:
    """Extract VIEW-INVARIANT pose features from cropped images"""
    
    def __init__(self, device: str = 'auto'):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8-Pose required for pose extraction")
        
        # Auto-detect GPU
        if device == 'auto':
            import torch
            device = '0' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLOv8-Pose model
        print(f"Loading YOLOv8-Pose model on {device}...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        self.device = device
        
        if device != 'cpu':
            print(f"✓ Using GPU: {device}")
        else:
            print("⚠ Using CPU (this will be slow)")
        
        print("✓ Pose model loaded")
        
        # Feature weights (learned from experiments)
        self.feature_weights = {
            'leg_torso_ratio': 1.5,
            'arm_torso_ratio': 1.3,
            'shoulder_hip_ratio': 1.2,
            'upper_lower_leg_ratio': 1.4,
            'upper_lower_arm_ratio': 1.1,
            'torso_aspect_ratio': 1.0,
            'arm_leg_ratio': 1.2,
            'body_compactness': 0.8,
            'limb_symmetry': 0.7,
        }
    
    def extract_features(self, image: np.ndarray) -> Optional[PoseFeatures]:
        """Extract pose features from cropped person image"""
        if image is None or image.size == 0:
            return None
        
        # Run pose estimation on GPU
        try:
            results = self.pose_model(image, verbose=False, device=self.device)
            
            if len(results) == 0 or results[0].keypoints is None:
                return None
            
            # Check if keypoints were detected
            kp_data = results[0].keypoints.xy
            if kp_data is None or len(kp_data) == 0:
                return None
            
            # Get keypoints (17 COCO keypoints)
            keypoints = kp_data[0].cpu().numpy()
            
            # Get confidences
            if results[0].keypoints.conf is not None and len(results[0].keypoints.conf) > 0:
                confidences = results[0].keypoints.conf[0].cpu().numpy()
            else:
                confidences = np.ones(17)
        except Exception as e:
            # Silently skip images where pose detection fails
            return None
        
        # Check quality
        avg_conf = np.mean(confidences)
        if avg_conf < 0.3:
            return None
        
        # Extract structural features
        features = self._extract_structural_features(keypoints, confidences)
        
        if not features:
            return None
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Compute quality score
        quality = self._compute_quality(keypoints, confidences)
        
        return PoseFeatures(
            features=features,
            keypoints=keypoints,
            quality_score=quality
        )
    
    def _extract_structural_features(self, kp: np.ndarray, conf: np.ndarray) -> Dict[str, float]:
        """Extract view-invariant structural features"""
        features = {}
        
        if len(kp) < 17:
            return features
        
        try:
            # Body measurements
            shoulder_center = (kp[5] + kp[6]) / 2
            hip_center = (kp[11] + kp[12]) / 2
            torso_height = self._distance(shoulder_center, hip_center)
            
            if torso_height < 1e-6:
                return features
            
            shoulder_width = self._distance(kp[5], kp[6])
            hip_width = self._distance(kp[11], kp[12])
            
            # Leg measurements
            left_upper_leg = self._distance(kp[11], kp[13])
            left_lower_leg = self._distance(kp[13], kp[15])
            left_leg = left_upper_leg + left_lower_leg
            
            right_upper_leg = self._distance(kp[12], kp[14])
            right_lower_leg = self._distance(kp[14], kp[16])
            right_leg = right_upper_leg + right_lower_leg
            
            avg_leg = (left_leg + right_leg) / 2
            
            # Arm measurements
            left_upper_arm = self._distance(kp[5], kp[7])
            left_lower_arm = self._distance(kp[7], kp[9])
            left_arm = left_upper_arm + left_lower_arm
            
            right_upper_arm = self._distance(kp[6], kp[8])
            right_lower_arm = self._distance(kp[8], kp[10])
            right_arm = right_upper_arm + right_lower_arm
            
            avg_arm = (left_arm + right_arm) / 2
            
            # Structural ratios (scale-invariant)
            if conf[11] > 0.5 and conf[12] > 0.5 and conf[13] > 0.5 and conf[14] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
                features['leg_torso_ratio'] = avg_leg / torso_height
            
            if conf[5] > 0.5 and conf[6] > 0.5 and conf[7] > 0.5 and conf[8] > 0.5 and conf[9] > 0.5 and conf[10] > 0.5:
                features['arm_torso_ratio'] = avg_arm / torso_height
            
            if conf[5] > 0.5 and conf[6] > 0.5 and conf[11] > 0.5 and conf[12] > 0.5:
                if hip_width > 1e-6:
                    features['shoulder_hip_ratio'] = shoulder_width / hip_width
                features['torso_aspect_ratio'] = (shoulder_width + hip_width) / 2 / torso_height
            
            if left_lower_leg + right_lower_leg > 1e-6:
                features['upper_lower_leg_ratio'] = (left_upper_leg + right_upper_leg) / (left_lower_leg + right_lower_leg)
            
            if left_lower_arm + right_lower_arm > 1e-6:
                features['upper_lower_arm_ratio'] = (left_upper_arm + right_upper_arm) / (left_lower_arm + right_lower_arm)
            
            if avg_leg > 1e-6:
                features['arm_leg_ratio'] = avg_arm / avg_leg
            
            total_limb = left_arm + right_arm + left_leg + right_leg
            features['body_compactness'] = total_limb / torso_height
            
            if max(left_leg, right_leg) > 1e-6:
                features['limb_symmetry'] = 1 - abs(left_leg - right_leg) / max(left_leg, right_leg)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return features
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to [0, 1]"""
        feature_ranges = {
            'leg_torso_ratio': (1.0, 3.0),
            'arm_torso_ratio': (0.8, 2.5),
            'shoulder_hip_ratio': (0.8, 1.5),
            'upper_lower_leg_ratio': (0.7, 1.3),
            'upper_lower_arm_ratio': (0.8, 1.4),
            'torso_aspect_ratio': (0.3, 1.0),
            'arm_leg_ratio': (0.4, 0.8),
            'body_compactness': (2.0, 6.0),
            'limb_symmetry': (0.7, 1.0),
        }
        
        normalized = {}
        for key, value in features.items():
            if key in feature_ranges:
                min_val, max_val = feature_ranges[key]
                clipped = np.clip(value, min_val, max_val)
                normalized[key] = (clipped - min_val) / (max_val - min_val)
            else:
                normalized[key] = np.clip(value, 0, 1)
        
        return normalized
    
    def _compute_quality(self, kp: np.ndarray, conf: np.ndarray) -> float:
        """Compute pose quality score"""
        # Visibility score
        visible_ratio = np.sum(conf > 0.5) / len(conf)
        
        # Keypoint spread (avoid collapsed poses)
        if len(kp) > 0:
            spread = np.std(kp, axis=0).mean()
            spread_score = min(spread / 50, 1.0)
        else:
            spread_score = 0.0
        
        quality = 0.6 * visible_ratio + 0.4 * spread_score
        return quality
    
    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return np.linalg.norm(p1 - p2)
    
    def compute_similarity(self, feat1: PoseFeatures, feat2: PoseFeatures) -> float:
        """Compute weighted similarity between pose features"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Get common features
        common_keys = set(feat1.features.keys()) & set(feat2.features.keys())
        
        if not common_keys:
            return 0.0
        
        # Weighted similarity
        weighted_diffs = []
        total_weight = 0.0
        
        for key in common_keys:
            weight = self.feature_weights.get(key, 1.0)
            diff = abs(feat1.features[key] - feat2.features[key])
            weighted_diffs.append(weight * (1 - diff))
            total_weight += weight
        
        similarity = sum(weighted_diffs) / total_weight if total_weight > 0 else 0.0
        return max(0.0, min(1.0, similarity))


# ==================== HAIR-BASED REID ====================

@dataclass
class HairFeatures:
    """Hair-based features for ReID"""
    color_hist: np.ndarray  # HSV histogram of hair region
    dominant_colors: List[Tuple[int, int, int]]  # RGB
    color_percentages: List[float]
    texture_features: Dict[str, float]
    hair_region_ratio: float  # Hair region size relative to head
    quality_score: float


class HairFeatureExtractor:
    """Extract hair features from person crops"""
    
    def __init__(self):
        # HSV histogram bins for hair color
        self.hsv_bins = [30, 8, 8]
        self.hsv_ranges = [0, 180, 0, 256, 0, 256]
        self.n_dominant_colors = 3
    
    def extract_features(self, image: np.ndarray, 
                        keypoints: Optional[np.ndarray] = None) -> Optional[HairFeatures]:
        """Extract hair features from cropped image"""
        if image is None or image.size == 0:
            return None
        
        h, w = image.shape[:2]
        
        # Locate hair region
        hair_region = self._locate_hair_region(image, keypoints)
        
        if hair_region is None or hair_region.size == 0:
            return None
        
        # Extract color histogram
        color_hist = self._extract_color_histogram(hair_region)
        
        # Extract dominant colors
        dominant_colors, percentages = self._extract_dominant_colors(hair_region)
        
        # Extract texture features
        texture_features = self._extract_texture_features(hair_region)
        
        # Compute quality
        hair_ratio = hair_region.size / image.size
        quality = self._compute_quality(hair_region, dominant_colors)
        
        return HairFeatures(
            color_hist=color_hist,
            dominant_colors=dominant_colors,
            color_percentages=percentages,
            texture_features=texture_features,
            hair_region_ratio=hair_ratio,
            quality_score=quality
        )
    
    def _locate_hair_region(self, image: np.ndarray,
                           keypoints: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Locate hair region in image"""
        h, w = image.shape[:2]
        
        if keypoints is not None and len(keypoints) >= 17:
            # Use face keypoints if available
            # Keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
            face_kps = keypoints[:5]
            valid_kps = face_kps[face_kps[:, 0] > 0]
            
            if len(valid_kps) > 0:
                # Hair is above face
                min_y = int(np.min(valid_kps[:, 1]))
                max_x = int(np.max(valid_kps[:, 0]))
                min_x = int(np.min(valid_kps[:, 0]))
                
                # Expand region for hair
                hair_top = max(0, min_y - int(h * 0.2))
                hair_bottom = min(h, min_y + int(h * 0.05))
                hair_left = max(0, min_x - int(w * 0.1))
                hair_right = min(w, max_x + int(w * 0.1))
                
                hair_region = image[hair_top:hair_bottom, hair_left:hair_right]
                
                if hair_region.size > 0:
                    return hair_region
        
        # Fallback: top 25% of image
        top_region_height = int(h * 0.25)
        hair_region = image[0:top_region_height, :]
        
        return hair_region
    
    def _extract_color_histogram(self, region: np.ndarray) -> np.ndarray:
        """Extract HSV color histogram"""
        if region is None or region.size == 0:
            return np.array([])
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Mask to remove very dark/bright pixels
        mask = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([180, 255, 240]))
        
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, self.hsv_bins, self.hsv_ranges)
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def _extract_dominant_colors(self, region: np.ndarray) -> Tuple[List[Tuple], List[float]]:
        """Extract dominant hair colors using K-means"""
        if region is None or region.size == 0:
            return [], []
        
        pixels = region.reshape(-1, 3)
        
        # Sample pixels
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        if len(pixels) < 10:
            return [], []
        
        try:
            if SKLEARN_AVAILABLE:
                kmeans = KMeans(n_clusters=min(self.n_dominant_colors, len(pixels)),
                               random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                colors = kmeans.cluster_centers_.astype(int)
                labels = kmeans.labels_
                
                percentages = []
                for i in range(len(colors)):
                    count = np.sum(labels == i)
                    percentages.append(count / len(labels))
                
                sorted_indices = np.argsort(percentages)[::-1]
                colors = [tuple(colors[i][::-1]) for i in sorted_indices]  # BGR to RGB
                percentages = [percentages[i] for i in sorted_indices]
                
                return colors, percentages
            else:
                # Simple fallback
                return self._simple_dominant_colors(pixels)
        except:
            return [], []
    
    def _simple_dominant_colors(self, pixels: np.ndarray) -> Tuple[List[Tuple], List[float]]:
        """Fallback dominant color extraction"""
        quantized = (pixels // 32) * 32
        unique, counts = np.unique(quantized, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        top_colors = unique[sorted_indices[:self.n_dominant_colors]]
        top_counts = counts[sorted_indices[:self.n_dominant_colors]]
        
        colors = [tuple(c[::-1]) for c in top_colors]
        percentages = (top_counts / top_counts.sum()).tolist()
        
        return colors, percentages
    
    def _extract_texture_features(self, region: np.ndarray) -> Dict[str, float]:
        """Extract texture features from hair"""
        features = {}
        
        if region is None or region.size == 0:
            return features
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Variance (texture complexity)
        features['variance'] = np.var(gray) / 10000.0
        
        # Edge density (hair detail)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Gradient magnitude (hair flow)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        features['gradient_mean'] = np.mean(gradient) / 255.0
        
        return features
    
    def _compute_quality(self, region: np.ndarray, dominant_colors: List) -> float:
        """Compute hair feature quality"""
        if region is None or region.size == 0:
            return 0.0
        
        # Size score
        size_score = min(region.size / (50 * 50), 1.0)
        
        # Color diversity score
        color_score = len(dominant_colors) / self.n_dominant_colors
        
        # Sharpness
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)
        
        quality = 0.4 * size_score + 0.3 * color_score + 0.3 * sharpness_score
        return quality
    
    def compute_similarity(self, feat1: HairFeatures, feat2: HairFeatures) -> float:
        """Compute hair similarity"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        similarities = []
        weights = []
        
        # Color histogram similarity
        hist_sim = 1 / (1 + self._chi2_distance(feat1.color_hist, feat2.color_hist))
        similarities.append(hist_sim)
        weights.append(0.5)
        
        # Dominant color similarity
        dom_sim = self._dominant_color_similarity(
            feat1.dominant_colors, feat1.color_percentages,
            feat2.dominant_colors, feat2.color_percentages
        )
        similarities.append(dom_sim)
        weights.append(0.3)
        
        # Texture similarity
        texture_sim = self._texture_similarity(feat1.texture_features, feat2.texture_features)
        similarities.append(texture_sim)
        weights.append(0.2)
        
        overall = sum(s * w for s, w in zip(similarities, weights))
        return overall
    
    @staticmethod
    def _chi2_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
        if len(hist1) == 0 or len(hist2) == 0:
            return 1.0
        eps = 1e-10
        distance = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
        return distance
    
    @staticmethod
    def _dominant_color_similarity(colors1, pcts1, colors2, pcts2) -> float:
        if not colors1 or not colors2:
            return 0.0
        
        sims = []
        for c1, p1 in zip(colors1, pcts1):
            best_sim = 0.0
            for c2, p2 in zip(colors2, pcts2):
                color_dist = np.linalg.norm(np.array(c1) - np.array(c2))
                color_sim = 1 / (1 + color_dist / 100)
                weighted_sim = color_sim * min(p1, p2)
                best_sim = max(best_sim, weighted_sim)
            sims.append(best_sim)
        
        return np.mean(sims) if sims else 0.0
    
    @staticmethod
    def _texture_similarity(tex1: Dict, tex2: Dict) -> float:
        common_keys = set(tex1.keys()) & set(tex2.keys())
        if not common_keys:
            return 0.0
        
        diffs = [abs(tex1[k] - tex2[k]) for k in common_keys]
        similarity = 1 - np.mean(diffs)
        return max(0.0, similarity)


# ==================== MULTI-MODAL FUSION ====================

class MultiModalReID:
    """Fuse multiple modalities for ReID"""
    
    def __init__(self, modalities: List[str] = ['pose', 'hair'], device: str = 'auto'):
        """
        Args:
            modalities: List of modalities to use
                       Options: 'pose', 'hair', 'face', 'color'
            device: 'auto', 'cpu', '0', '1', etc. (GPU device)
        """
        self.modalities = modalities
        self.extractors = {}
        
        # Initialize extractors
        if 'pose' in modalities:
            print("Initializing Pose extractor...")
            self.extractors['pose'] = PoseFeatureExtractor(device=device)
        
        if 'hair' in modalities:
            print("Initializing Hair extractor...")
            self.extractors['hair'] = HairFeatureExtractor()
        
        # Fusion weights (can be learned/tuned)
        self.fusion_weights = {
            'pose': 0.5,
            'hair': 0.5,
            'face': 0.0,  # TODO: implement
            'color': 0.0  # TODO: implement
        }
        
        # Normalize weights
        total_weight = sum(self.fusion_weights[m] for m in modalities)
        for m in modalities:
            self.fusion_weights[m] /= total_weight
        
        print(f"✓ Multi-modal ReID initialized: {modalities}")
        print(f"  Fusion weights: {self.fusion_weights}")
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """Extract features from all modalities"""
        features = {}
        
        # Extract pose features (also provides keypoints for hair)
        keypoints = None
        if 'pose' in self.modalities:
            pose_feat = self.extractors['pose'].extract_features(image)
            features['pose'] = pose_feat
            if pose_feat is not None:
                keypoints = pose_feat.keypoints
        
        # Extract hair features
        if 'hair' in self.modalities:
            hair_feat = self.extractors['hair'].extract_features(image, keypoints)
            features['hair'] = hair_feat
        
        return features
    
    def compute_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """Compute fused similarity across all modalities"""
        similarities = []
        weights = []
        
        for modality in self.modalities:
            if modality in feat1 and modality in feat2:
                if feat1[modality] is not None and feat2[modality] is not None:
                    sim = self.extractors[modality].compute_similarity(
                        feat1[modality], feat2[modality]
                    )
                    similarities.append(sim)
                    weights.append(self.fusion_weights[modality])
        
        if not similarities:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        fused_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        
        return fused_sim


# ==================== EXAMPLE USAGE ====================

def main():
    """Example: Extract pose and hair features from an image"""
    
    print("="*70)
    print("Multi-Modal ReID Feature Extraction Demo")
    print("="*70)
    
    # Test with a sample image
    test_image_path = "path/to/test/image.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"\nPlease provide a test image at: {test_image_path}")
        print("Or modify the path in the script")
        return
    
    image = cv2.imread(test_image_path)
    
    # Initialize multi-modal system
    reid_system = MultiModalReID(modalities=['pose', 'hair'])
    
    # Extract features
    print("\nExtracting features...")
    features = reid_system.extract_features(image)
    
    # Display results
    print("\n📊 Extracted Features:")
    for modality, feat in features.items():
        if feat is not None:
            print(f"\n{modality.upper()}:")
            print(f"  Quality: {feat.quality_score:.3f}")
            if hasattr(feat, 'features'):
                print(f"  Features: {len(feat.features)}")
        else:
            print(f"\n{modality.upper()}: Failed to extract")
    
    print("\n✓ Feature extraction complete!")


if __name__ == "__main__":
    import os
    main()