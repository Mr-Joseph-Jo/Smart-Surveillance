"""
Color-Based Person Re-Identification System
Features:
- Separate top/bottom color extraction
- Multi-color pattern analysis
- HSV color histograms (illumination-invariant)
- Dominant color detection
- Color texture patterns
- Temporal color averaging
- Occlusion-aware color extraction
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean
from collections import deque, defaultdict, Counter
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed, using basic clustering")


@dataclass
class ColorFeatures:
    """Store color appearance features"""
    person_id: int
    
    # Top clothing colors
    top_hsv_hist: np.ndarray          # HSV histogram
    top_dominant_colors: List[Tuple[int, int, int]]  # RGB dominant colors
    top_color_percentages: List[float]  # Percentage of each dominant color
    
    # Bottom clothing colors
    bottom_hsv_hist: np.ndarray
    bottom_dominant_colors: List[Tuple[int, int, int]]
    bottom_color_percentages: List[float]
    
    # Full body (for backup)
    full_hsv_hist: np.ndarray
    full_dominant_colors: List[Tuple[int, int, int]]
    
    # Metadata
    timestamp: float
    camera_id: str
    bbox: Tuple[int, int, int, int]
    quality_score: float
    track_id: int = -1
    
    # Additional features
    brightness: float = 0.0
    saturation: float = 0.0
    color_complexity: float = 0.0  # Number of distinct colors


class ColorFeatureExtractor:
    """Extract color appearance features from person crops"""
    
    def __init__(self):
        # HSV histogram bins (Hue: 30, Saturation: 8, Value: 8)
        self.hsv_bins = [30, 8, 8]
        self.hsv_ranges = [0, 180, 0, 256, 0, 256]
        
        # Number of dominant colors to extract
        self.n_dominant_colors = 5
        
        # Color names for semantic understanding (optional)
        self.color_map = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'red2': ([170, 100, 100], [180, 255, 255]),  # Red wraps around
            'orange': ([10, 100, 100], [25, 255, 255]),
            'yellow': ([25, 100, 100], [35, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255]),
            'cyan': ([85, 100, 100], [95, 255, 255]),
            'blue': ([95, 100, 100], [125, 255, 255]),
            'purple': ([125, 100, 100], [155, 255, 255]),
            'pink': ([155, 100, 100], [170, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 50]),
            'gray': ([0, 0, 50], [180, 30, 200]),
        }
        
    def extract_features(self, person_crop: np.ndarray,
                        keypoints: Optional[np.ndarray] = None) -> Tuple[Optional[ColorFeatures], float]:
        """
        Extract color features from person crop
        
        Args:
            person_crop: BGR image of person
            keypoints: Optional pose keypoints for better segmentation
            
        Returns:
            (ColorFeatures, quality_score) or (None, 0.0)
        """
        if person_crop is None or person_crop.size == 0:
            return None, 0.0
        
        h, w = person_crop.shape[:2]
        
        if h < 30 or w < 20:  # Too small
            return None, 0.0
        
        # Segment top and bottom regions
        top_region, bottom_region = self._segment_top_bottom(
            person_crop, keypoints
        )
        
        if top_region is None or bottom_region is None:
            return None, 0.0
        
        # Extract color features for each region
        top_hist, top_dominant, top_percentages = self._extract_color_histogram(top_region)
        bottom_hist, bottom_dominant, bottom_percentages = self._extract_color_histogram(bottom_region)
        full_hist, full_dominant, _ = self._extract_color_histogram(person_crop)
        
        # Compute quality metrics
        quality_score = self._compute_quality_score(
            person_crop, top_region, bottom_region
        )
        
        # Additional features
        brightness = self._compute_brightness(person_crop)
        saturation = self._compute_saturation(person_crop)
        color_complexity = len(full_dominant)
        
        color_features = ColorFeatures(
            person_id=-1,
            top_hsv_hist=top_hist,
            top_dominant_colors=top_dominant,
            top_color_percentages=top_percentages,
            bottom_hsv_hist=bottom_hist,
            bottom_dominant_colors=bottom_dominant,
            bottom_color_percentages=bottom_percentages,
            full_hsv_hist=full_hist,
            full_dominant_colors=full_dominant,
            timestamp=0.0,
            camera_id="",
            bbox=(0, 0, 0, 0),
            quality_score=quality_score,
            brightness=brightness,
            saturation=saturation,
            color_complexity=color_complexity
        )
        
        return color_features, quality_score
    
    def _segment_top_bottom(self, person_crop: np.ndarray,
                           keypoints: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segment person into top and bottom clothing regions
        
        Strategy:
        1. If keypoints available: use shoulder-hip boundary
        2. Otherwise: use 60-40 split (top 60%, bottom 40%)
        """
        h, w = person_crop.shape[:2]
        
        if keypoints is not None and len(keypoints) >= 17:
            # Use pose keypoints for intelligent segmentation
            # YOLO keypoints: 5,6=shoulders, 11,12=hips
            
            # Normalize keypoints to crop coordinates
            # Assuming keypoints are in original image coords, need to transform
            # For simplicity, use ratio-based approach
            
            try:
                # Get shoulder and hip y-coordinates
                shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
                hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
                
                # Convert to crop coordinates (approximate)
                # This assumes keypoints are already in crop space
                if 0 < shoulder_y < h and 0 < hip_y < h and shoulder_y < hip_y:
                    split_y = int((shoulder_y + hip_y) / 2)
                else:
                    # Fallback to fixed ratio
                    split_y = int(h * 0.5)
            except:
                split_y = int(h * 0.5)
        else:
            # Simple split: top 50%, bottom 50%
            split_y = int(h * 0.5)
        
        # Add margins to avoid head and feet
        top_margin = int(h * 0.1)  # Skip top 10% (head/neck)
        bottom_margin = int(h * 0.1)  # Skip bottom 10% (feet)
        
        # Extract regions
        top_region = person_crop[top_margin:split_y, :]
        bottom_region = person_crop[split_y:h-bottom_margin, :]
        
        # Validate regions
        if top_region.size == 0 or bottom_region.size == 0:
            return None, None
        
        return top_region, bottom_region
    
    def _extract_color_histogram(self, region: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[float]]:
        """
        Extract HSV histogram and dominant colors
        
        Returns:
            (hsv_histogram, dominant_colors, percentages)
        """
        if region is None or region.size == 0:
            return np.array([]), [], []
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Create mask to remove background (very dark or very bright pixels)
        mask = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([180, 255, 240]))
        
        # Compute HSV histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, self.hsv_bins, self.hsv_ranges)
        hist = cv2.normalize(hist, hist).flatten()
        
        # Extract dominant colors using K-means
        dominant_colors, percentages = self._extract_dominant_colors(
            region, mask, n_colors=self.n_dominant_colors
        )
        
        return hist, dominant_colors, percentages
    
    def _extract_dominant_colors(self, region: np.ndarray, mask: np.ndarray,
                                 n_colors: int = 5) -> Tuple[List[Tuple], List[float]]:
        """Extract dominant colors using K-means clustering"""
        
        # Reshape image to list of pixels
        pixels = region.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        
        # Filter using mask
        valid_pixels = pixels[mask_flat > 0]
        
        if len(valid_pixels) < n_colors:
            return [], []
        
        # Sample pixels if too many (for speed)
        if len(valid_pixels) > 10000:
            indices = np.random.choice(len(valid_pixels), 10000, replace=False)
            valid_pixels = valid_pixels[indices]
        
        try:
            if SKLEARN_AVAILABLE:
                # Use sklearn K-means
                kmeans = KMeans(n_clusters=min(n_colors, len(valid_pixels)), 
                               random_state=42, n_init=10)
                kmeans.fit(valid_pixels)
                
                colors = kmeans.cluster_centers_.astype(int)
                labels = kmeans.labels_
                
                # Calculate percentages
                percentages = []
                for i in range(len(colors)):
                    count = np.sum(labels == i)
                    percentages.append(count / len(labels))
                
                # Sort by percentage
                sorted_indices = np.argsort(percentages)[::-1]
                colors = [tuple(colors[i][::-1]) for i in sorted_indices]  # BGR to RGB
                percentages = [percentages[i] for i in sorted_indices]
                
            else:
                # Simple histogram-based approach
                colors, percentages = self._simple_dominant_colors(valid_pixels, n_colors)
            
            return colors, percentages
            
        except Exception as e:
            print(f"Dominant color extraction error: {e}")
            return [], []
    
    def _simple_dominant_colors(self, pixels: np.ndarray, n_colors: int) -> Tuple[List[Tuple], List[float]]:
        """Simple dominant color extraction without sklearn"""
        # Quantize colors to reduce space
        quantized = (pixels // 32) * 32
        
        # Count unique colors
        unique, counts = np.unique(quantized, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        top_colors = unique[sorted_indices[:n_colors]]
        top_counts = counts[sorted_indices[:n_colors]]
        
        # Convert to RGB and percentages
        colors = [tuple(c[::-1]) for c in top_colors]  # BGR to RGB
        percentages = (top_counts / top_counts.sum()).tolist()
        
        return colors, percentages
    
    def _compute_quality_score(self, full_crop: np.ndarray,
                               top_region: np.ndarray,
                               bottom_region: np.ndarray) -> float:
        """Compute quality score for color features"""
        
        scores = []
        
        # 1. Image sharpness (Laplacian variance)
        gray = cv2.cvtColor(full_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000, 1.0)
        scores.append(sharpness_score)
        
        # 2. Brightness (not too dark, not too bright)
        hsv = cv2.cvtColor(full_crop, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2]) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal at 0.5
        scores.append(brightness_score)
        
        # 3. Color saturation (colorful is better)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        saturation_score = saturation
        scores.append(saturation_score)
        
        # 4. Region size (bigger is better)
        h, w = full_crop.shape[:2]
        size_score = min((h * w) / (200 * 100), 1.0)  # Normalize by typical size
        scores.append(size_score)
        
        # Overall quality
        quality = np.mean(scores)
        return quality
    
    def _compute_brightness(self, image: np.ndarray) -> float:
        """Compute average brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2]) / 255.0
    
    def _compute_saturation(self, image: np.ndarray) -> float:
        """Compute average saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 1]) / 255.0
    
    def get_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """Get semantic color name (optional, for visualization)"""
        # Convert RGB to HSV
        rgb_array = np.uint8([[list(rgb_color)]])
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]
        
        # Check against color map
        for color_name, (lower, upper) in self.color_map.items():
            if (lower[0] <= hsv[0] <= upper[0] and
                lower[1] <= hsv[1] <= upper[1] and
                lower[2] <= hsv[2] <= upper[2]):
                return color_name
        
        return "unknown"


class GalleryManager:
    """Manage color gallery with temporal averaging"""
    
    def __init__(self, max_frames_per_person: int = 15):
        self.max_frames = max_frames_per_person
        self.gallery: Dict[int, List[ColorFeatures]] = {}
    
    def add(self, color_features: ColorFeatures):
        """Add color features to gallery"""
        if color_features.person_id not in self.gallery:
            self.gallery[color_features.person_id] = []
        
        self.gallery[color_features.person_id].append(color_features)
        
        # Keep best frames
        if len(self.gallery[color_features.person_id]) > self.max_frames:
            self.gallery[color_features.person_id].sort(
                key=lambda x: x.quality_score, reverse=True
            )
            self.gallery[color_features.person_id] = \
                self.gallery[color_features.person_id][:self.max_frames]
    
    def get(self, person_id: int) -> List[ColorFeatures]:
        return self.gallery.get(person_id, [])
    
    def get_all_ids(self) -> List[int]:
        return list(self.gallery.keys())
    
    def get_averaged_features(self, person_id: int) -> Optional[ColorFeatures]:
        """Get temporally averaged color features"""
        frames = self.get(person_id)
        if not frames:
            return None
        
        # Average histograms
        top_hists = [f.top_hsv_hist for f in frames]
        bottom_hists = [f.bottom_hsv_hist for f in frames]
        full_hists = [f.full_hsv_hist for f in frames]
        
        avg_top_hist = np.mean(top_hists, axis=0)
        avg_bottom_hist = np.mean(bottom_hists, axis=0)
        avg_full_hist = np.mean(full_hists, axis=0)
        
        # Use dominant colors from best frame
        best_frame = max(frames, key=lambda x: x.quality_score)
        
        # Create averaged feature
        avg_features = ColorFeatures(
            person_id=person_id,
            top_hsv_hist=avg_top_hist,
            top_dominant_colors=best_frame.top_dominant_colors,
            top_color_percentages=best_frame.top_color_percentages,
            bottom_hsv_hist=avg_bottom_hist,
            bottom_dominant_colors=best_frame.bottom_dominant_colors,
            bottom_color_percentages=best_frame.bottom_color_percentages,
            full_hsv_hist=avg_full_hist,
            full_dominant_colors=best_frame.full_dominant_colors,
            timestamp=best_frame.timestamp,
            camera_id=best_frame.camera_id,
            bbox=best_frame.bbox,
            quality_score=np.mean([f.quality_score for f in frames]),
            brightness=np.mean([f.brightness for f in frames]),
            saturation=np.mean([f.saturation for f in frames]),
            color_complexity=best_frame.color_complexity
        )
        
        return avg_features


class ColorReIDMatcher:
    """Match persons based on color appearance"""
    
    def __init__(self, similarity_threshold: float = 0.70):
        self.similarity_threshold = similarity_threshold
        self.gallery_manager = GalleryManager(max_frames_per_person=15)
        
        # Weights for different similarity components
        self.weights = {
            'top_hist': 0.35,
            'bottom_hist': 0.35,
            'top_dominant': 0.15,
            'bottom_dominant': 0.15,
        }
    
    def add_to_gallery(self, color_features: ColorFeatures):
        """Add color features to gallery"""
        self.gallery_manager.add(color_features)
    
    def match(self, query_features: ColorFeatures) -> Tuple[Optional[int], float, Dict]:
        """
        Match query against gallery
        
        Returns:
            (matched_person_id, similarity, debug_info)
        """
        gallery_ids = self.gallery_manager.get_all_ids()
        
        if not gallery_ids:
            return None, 0.0, {}
        
        best_match_id = None
        best_similarity = 0.0
        debug_info = {}
        
        for person_id in gallery_ids:
            # Get averaged features for this person
            gallery_features = self.gallery_manager.get_averaged_features(person_id)
            
            if gallery_features is None:
                continue
            
            # Compute similarity
            similarity = self._compute_similarity(query_features, gallery_features)
            
            debug_info[person_id] = {
                'similarity': similarity,
                'num_frames': len(self.gallery_manager.get(person_id)),
                'avg_quality': gallery_features.quality_score
            }
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        # Adaptive threshold
        adaptive_threshold = self._get_adaptive_threshold(len(gallery_ids))
        
        if best_similarity >= adaptive_threshold:
            return best_match_id, best_similarity, debug_info
        else:
            return None, best_similarity, debug_info
    
    def _compute_similarity(self, query: ColorFeatures, gallery: ColorFeatures) -> float:
        """Compute color similarity between query and gallery"""
        
        similarities = {}
        
        # 1. Top histogram similarity (chi-square distance)
        top_hist_sim = 1 / (1 + self._chi2_distance(query.top_hsv_hist, gallery.top_hsv_hist))
        similarities['top_hist'] = top_hist_sim
        
        # 2. Bottom histogram similarity
        bottom_hist_sim = 1 / (1 + self._chi2_distance(query.bottom_hsv_hist, gallery.bottom_hsv_hist))
        similarities['bottom_hist'] = bottom_hist_sim
        
        # 3. Top dominant colors similarity
        top_dom_sim = self._dominant_color_similarity(
            query.top_dominant_colors, query.top_color_percentages,
            gallery.top_dominant_colors, gallery.top_color_percentages
        )
        similarities['top_dominant'] = top_dom_sim
        
        # 4. Bottom dominant colors similarity
        bottom_dom_sim = self._dominant_color_similarity(
            query.bottom_dominant_colors, query.bottom_color_percentages,
            gallery.bottom_dominant_colors, gallery.bottom_color_percentages
        )
        similarities['bottom_dominant'] = bottom_dom_sim
        
        # Weighted combination
        overall_similarity = sum(
            self.weights[key] * similarities[key]
            for key in self.weights.keys()
        )
        
        return overall_similarity
    
    @staticmethod
    def _chi2_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute chi-square distance between histograms"""
        if len(hist1) == 0 or len(hist2) == 0:
            return 1.0
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        distance = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
        return distance
    
    @staticmethod
    def _dominant_color_similarity(colors1: List[Tuple], percentages1: List[float],
                                   colors2: List[Tuple], percentages2: List[float]) -> float:
        """
        Compute similarity between dominant color sets
        Uses Hungarian-style matching of colors weighted by percentages
        """
        if not colors1 or not colors2:
            return 0.0
        
        # Compute pairwise color distances
        similarities = []
        
        for i, (c1, p1) in enumerate(zip(colors1, percentages1)):
            best_match_sim = 0.0
            
            for j, (c2, p2) in enumerate(zip(colors2, percentages2)):
                # Euclidean distance in RGB space
                color_dist = np.linalg.norm(np.array(c1) - np.array(c2))
                color_sim = 1 / (1 + color_dist / 100)  # Normalize
                
                # Weight by percentages
                weighted_sim = color_sim * min(p1, p2)
                
                if weighted_sim > best_match_sim:
                    best_match_sim = weighted_sim
            
            similarities.append(best_match_sim)
        
        # Average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def _get_adaptive_threshold(self, gallery_size: int) -> float:
        """Adaptive threshold based on gallery size"""
        if gallery_size <= 2:
            return self.similarity_threshold
        elif gallery_size <= 5:
            return self.similarity_threshold + 0.05
        else:
            return self.similarity_threshold + 0.10


class TemporalSmoother:
    """Smooth ReID over time"""
    
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_and_vote(self, track_id: int, reid_id: Optional[int]) -> Optional[int]:
        """Add and vote"""
        self.history[track_id].append(reid_id)
        
        valid_votes = [x for x in self.history[track_id] if x is not None]
        
        if not valid_votes or len(valid_votes) < 3:
            return None
        
        counter = Counter(valid_votes)
        most_common_id, count = counter.most_common(1)[0]
        
        if count >= len(valid_votes) * 0.5:
            return most_common_id
        
        return None


class ColorReIDSystem:
    """Complete Color-based Re-Identification System"""
    
    def __init__(self, model_path: str = 'yolov8n-pose.pt'):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        print(f"Loading YOLO model: {model_path}")
        self.yolo_model = YOLO(model_path)
        self.color_extractor = ColorFeatureExtractor()
        self.matcher = ColorReIDMatcher(similarity_threshold=0.70)
        self.temporal_smoother = TemporalSmoother(window_size=7)
        
        print("✓ Color ReID System initialized")
        print("  - Top/Bottom color segmentation")
        print("  - Multi-color pattern analysis")
        print("  - HSV histogram matching")
        print("  - Dominant color extraction")
        print("  - Temporal averaging")
        
        self.track_to_reid_map: Dict[int, int] = {}
    
    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List:
        """Detect and track persons ONLY (filter class 0)"""
        # Track only persons (class 0)
        results = self.yolo_model.track(frame, persist=persist, verbose=False, 
                                       tracker="bytetrack.yaml", classes=[0])
        persons = []
        
        if len(results) == 0:
            return persons
        
        result = results[0]
        
        if result.boxes is None or result.boxes.id is None:
            return persons
        
        # Get class IDs to double-check
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
        
        num_persons = len(result.boxes)
        
        for i in range(num_persons):
            # Double-check: only process if class is 0 (person)
            if class_ids is not None and int(class_ids[i]) != 0:
                continue
            
            track_id = int(result.boxes.id[i].item())
            box = result.boxes[i].xyxy[0].cpu().numpy()
            bbox = tuple(map(int, box))
            conf = float(result.boxes.conf[i].item())
            
            # Get keypoints if available
            kp = None
            if result.keypoints is not None:
                kp = result.keypoints.xy[i].cpu().numpy()
            
            persons.append((track_id, bbox, conf, kp))
        
        return persons
    
    def process_frame(self, frame: np.ndarray, person_id: int,
                     camera_id: str, bbox: Tuple[int, int, int, int],
                     timestamp: float, keypoints: Optional[np.ndarray] = None,
                     track_id: int = -1) -> Optional[ColorFeatures]:
        """Extract color features from frame"""
        x1, y1, x2, y2 = bbox
        
        # Add padding
        h, w = frame.shape[:2]
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
        
        # Extract color features
        color_features, quality_score = self.color_extractor.extract_features(
            person_crop, keypoints
        )
        
        if color_features is None or quality_score < 0.3:
            return None
        
        # Update metadata
        color_features.person_id = person_id
        color_features.timestamp = timestamp
        color_features.camera_id = camera_id
        color_features.bbox = bbox
        color_features.track_id = track_id
        
        return color_features
    
    def register_person(self, color_features: ColorFeatures):
        """Register person"""
        self.matcher.add_to_gallery(color_features)
    
    def identify_person_smoothed(self, track_id: int, color_features: ColorFeatures) -> Tuple[Optional[int], float, Dict]:
        """Identify with smoothing"""
        matched_id, similarity, debug_info = self.matcher.match(color_features)
        smoothed_id = self.temporal_smoother.add_and_vote(track_id, matched_id)
        return smoothed_id, similarity, debug_info
    
    def update_track_mapping(self, track_id: int, reid_id: int):
        """Update mapping"""
        self.track_to_reid_map[track_id] = reid_id


class MouseSelector:
    """Mouse selection"""
    
    def __init__(self):
        self.selected_point = None
        self.selecting = False
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)
            self.selecting = True
    
    def get_selected_person(self, persons: List[Tuple]) -> Optional[int]:
        if not self.selecting or self.selected_point is None:
            return None
        
        x, y = self.selected_point
        
        for idx, person_data in enumerate(persons):
            bbox = person_data[1]
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selecting = False
                self.selected_point = None
                return idx
        
        self.selecting = False
        self.selected_point = None
        return None


def visualize_colors(frame: np.ndarray, color_features: ColorFeatures,
                    bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Visualize extracted colors on frame"""
    x1, y1, x2, y2 = bbox
    vis_frame = frame.copy()
    
    # Draw bbox
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Check if we have space to draw colors (avoid drawing outside frame)
    frame_height, frame_width = vis_frame.shape[:2]
    color_box_size = 30
    x_offset = x2 + 10
    
    # If not enough space on right, draw on left
    if x_offset + color_box_size + 50 > frame_width:
        x_offset = max(10, x1 - color_box_size - 60)
    
    y_offset = y1
    
    cv2.putText(vis_frame, "Top:", (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    
    # Draw top dominant colors
    for i, (color, pct) in enumerate(zip(color_features.top_dominant_colors[:3],
                                          color_features.top_color_percentages[:3])):
        # Ensure color is in BGR format and integer tuple
        # Colors from dominant_colors are in RGB, need to convert to BGR
        bgr_color = (int(color[2]), int(color[1]), int(color[0]))  # RGB to BGR
        
        # Make sure we don't draw outside frame
        if y_offset + color_box_size > frame_height:
            break
        
        cv2.rectangle(vis_frame, 
                     (x_offset, y_offset), 
                     (x_offset + color_box_size, y_offset + color_box_size),
                     bgr_color, -1)
        cv2.putText(vis_frame, f"{pct*100:.0f}%", 
                   (x_offset + color_box_size + 5, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += color_box_size + 5
    
    # Draw bottom dominant colors
    y_offset += 10
    
    if y_offset < frame_height - 20:
        cv2.putText(vis_frame, "Bottom:", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for i, (color, pct) in enumerate(zip(color_features.bottom_dominant_colors[:3],
                                              color_features.bottom_color_percentages[:3])):
            # Convert RGB to BGR
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))
            
            # Make sure we don't draw outside frame
            if y_offset + color_box_size > frame_height:
                break
            
            cv2.rectangle(vis_frame,
                         (x_offset, y_offset),
                         (x_offset + color_box_size, y_offset + color_box_size),
                         bgr_color, -1)
            cv2.putText(vis_frame, f"{pct*100:.0f}%",
                       (x_offset + color_box_size + 5, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += color_box_size + 5
    
    return vis_frame


def process_video_1_registration(reid_system: ColorReIDSystem, video_path: str,
                                  num_frames_to_collect: int = 30) -> Tuple[bool, int]:
    """Registration phase"""
    print("\n" + "=" * 70)
    print("VIDEO 1: Color Registration Phase")
    print("=" * 70)
    print(f"Click on person to register")
    print(f"Collecting {num_frames_to_collect} frames")
    print("Press 'q' to quit")
    print()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return False, -1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}")
    
    mouse_selector = MouseSelector()
    window_name = "Video 1 - Color Registration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_selector.mouse_callback)
    
    registered = False
    registered_person_id = -1
    frame_count = 0
    
    collecting_mode = False
    selected_track_id = None
    frames_collected = 0
    quality_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        persons = reid_system.detect_and_track(frame, persist=True)
        display_frame = frame.copy()
        
        if not collecting_mode:
            # Show all persons
            for track_id, bbox, conf, kp in persons:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Track:{track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Click on person to register",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Detected: {len(persons)} | Frame: {frame_count}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            # Collection mode
            person_found = False
            for track_id, bbox, conf, kp in persons:
                if track_id == selected_track_id:
                    person_found = True
                    
                    color_features = reid_system.process_frame(
                        frame, person_id=1, camera_id="video1",
                        bbox=bbox, timestamp=frame_count/fps,
                        keypoints=kp, track_id=track_id
                    )
                    
                    if color_features and color_features.quality_score > 0.3:
                        reid_system.register_person(color_features)
                        frames_collected += 1
                        quality_scores.append(color_features.quality_score)
                        
                        print(f"  ✓ Frame {frames_collected}/{num_frames_to_collect} | "
                              f"Quality: {color_features.quality_score:.3f} | "
                              f"Top colors: {len(color_features.top_dominant_colors)} | "
                              f"Bottom colors: {len(color_features.bottom_dominant_colors)}")
                        
                        # Visualize colors
                        display_frame = visualize_colors(display_frame, color_features, bbox)
                    else:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    
                    break
            
            if not person_found:
                cv2.putText(display_frame, f"Waiting for Track {selected_track_id}...",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Progress
            progress = int((frames_collected / num_frames_to_collect) * 100)
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            cv2.putText(display_frame,
                       f"Collecting: {frames_collected}/{num_frames_to_collect} ({progress}%) | Avg Q: {avg_quality:.2f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Progress bar
            bar_w, bar_h = 400, 30
            bar_x, bar_y = 10, 90
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
            filled_w = int((frames_collected / num_frames_to_collect) * bar_w)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)
            
            if frames_collected >= num_frames_to_collect:
                print(f"\n✓✓✓ Color Registration complete!")
                print(f"  Frames collected: {frames_collected}")
                print(f"  Average quality: {avg_quality:.3f}")
                
                cv2.putText(display_frame, "REGISTRATION COMPLETE!",
                           (display_frame.shape[1]//4, display_frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(2000)
                
                registered = True
                registered_person_id = 1
                break
        
        cv2.imshow(window_name, display_frame)
        
        if not collecting_mode:
            selected_idx = mouse_selector.get_selected_person(persons)
            if selected_idx is not None:
                selected_track_id = persons[selected_idx][0]
                print(f"\n✓ Selected Track {selected_track_id}")
                collecting_mode = True
                frames_collected = 0
                quality_scores = []
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return registered, registered_person_id


def process_video_2_identification(reid_system: ColorReIDSystem, video_path: str,
                                    registered_person_id: int):
    """Identification phase"""
    print("\n" + "=" * 70)
    print("VIDEO 2: Color Identification Phase")
    print("=" * 70)
    print("Matching based on clothing colors")
    print("Press 'q' to quit")
    print()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}")
    
    window_name = "Video 2 - Color Identification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    raw_matches = 0
    smoothed_matches = 0
    track_stats = defaultdict(lambda: {'raw': 0, 'smoothed': 0, 'best_sim': 0.0})
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        persons = reid_system.detect_and_track(frame, persist=True)
        display_frame = frame.copy()
        
        for track_id, bbox, conf, kp in persons:
            color_features = reid_system.process_frame(
                frame, person_id=999, camera_id="video2",
                bbox=bbox, timestamp=frame_count/fps,
                keypoints=kp, track_id=track_id
            )
            
            if color_features:
                smoothed_id, similarity, debug_info = reid_system.identify_person_smoothed(
                    track_id, color_features
                )
                
                # Raw match check
                raw_matched = any(info['similarity'] >= reid_system.matcher.similarity_threshold 
                                 for info in debug_info.values())
                
                if raw_matched:
                    raw_matches += 1
                    track_stats[track_id]['raw'] += 1
                
                if smoothed_id is not None:
                    smoothed_matches += 1
                    track_stats[track_id]['smoothed'] += 1
                    track_stats[track_id]['best_sim'] = max(track_stats[track_id]['best_sim'], similarity)
                
                x1, y1, x2, y2 = bbox
                
                if track_id in reid_system.track_to_reid_map:
                    mapped_id = reid_system.track_to_reid_map[track_id]
                    color = (0, 255, 0)
                    label = f"ReID:{mapped_id} Track:{track_id} ({similarity:.2f})"
                    thickness = 3
                    
                    # Show color visualization
                    display_frame = visualize_colors(display_frame, color_features, bbox)
                    
                elif smoothed_id is not None:
                    reid_system.update_track_mapping(track_id, smoothed_id)
                    color = (0, 255, 0)
                    label = f"MATCH! ReID:{smoothed_id} Track:{track_id} ({similarity:.2f})"
                    thickness = 3
                    
                    print(f"[Frame {frame_count}] ✓ COLOR MATCH! Track:{track_id} → ReID:{smoothed_id} | "
                          f"Sim:{similarity:.3f} | Quality:{color_features.quality_score:.2f}")
                    
                    display_frame = visualize_colors(display_frame, color_features, bbox)
                    
                elif raw_matched:
                    color = (0, 255, 255)
                    label = f"Candidate Track:{track_id} ({similarity:.2f})"
                    thickness = 2
                else:
                    color = (0, 0, 255)
                    label = f"Track:{track_id} ({similarity:.2f})"
                    thickness = 1
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(display_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display info
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Raw: {raw_matches} | Smoothed: {smoothed_matches}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Confirmed: {len(reid_system.track_to_reid_map)}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print(f"\n{'='*70}")
    print("FINAL COLOR-BASED REID STATISTICS")
    print(f"{'='*70}")
    print(f"Total frames: {frame_count}")
    print(f"Raw matches: {raw_matches} ({raw_matches/frame_count*100:.1f}%)")
    print(f"Smoothed matches: {smoothed_matches} ({smoothed_matches/frame_count*100:.1f}%)")
    print(f"\n📊 Per-Track Statistics:")
    for track_id, stats in sorted(track_stats.items()):
        reid_id = reid_system.track_to_reid_map.get(track_id, "None")
        print(f"  Track {track_id} → ReID {reid_id}:")
        print(f"    Raw: {stats['raw']} | Smoothed: {stats['smoothed']} | Best sim: {stats['best_sim']:.3f}")


def main():
    """Color-based ReID demo"""
    
    print("=" * 70)
    print("Color-Based Person Re-Identification System")
    print("=" * 70)
    print("Features:")
    print("  ✓ Top/Bottom clothing segmentation")
    print("  ✓ HSV color histograms (illumination-invariant)")
    print("  ✓ Dominant color extraction (K-means)")
    print("  ✓ Multi-color pattern analysis")
    print("  ✓ Temporal color averaging")
    print("  ✓ Adaptive threshold matching")
    print("=" * 70)
    
    if not YOLO_AVAILABLE:
        print("\n✗ ultralytics not installed!")
        return
    
    video1_path = "../videos/4PTF1_short.mp4"
    video2_path = "../videos/4PTF2_short.mp4"
    
    try:
        reid_system = ColorReIDSystem(model_path='yolov8n.pt')
        
        registered, registered_person_id = process_video_1_registration(
            reid_system, video1_path, num_frames_to_collect=30
        )
        
        if registered:
            process_video_2_identification(reid_system, video2_path, registered_person_id)
        
        print("\n✓ Color ReID Demo complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()