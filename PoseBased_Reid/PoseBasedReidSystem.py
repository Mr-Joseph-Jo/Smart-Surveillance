"""
Pose-Based Person Re-Identification System with YOLOv8-Pose + ByteTrack
Two-video demo: Register from video 1, identify in video 2
ENHANCED VERSION: Multi-frame registration + ByteTrack tracking + Better matching
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean
from collections import deque, defaultdict
import json

# Import YOLO with tracking
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

@dataclass
class PoseFeatures:
    """Store extracted pose features for a person"""
    person_id: int
    features: Dict[str, float]
    keypoints: np.ndarray
    timestamp: float
    camera_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float = 0.0
    track_id: int = -1  # ByteTrack ID
    
class PoseFeatureExtractor:
    """Extract discriminative features from pose keypoints"""
    
    def __init__(self):
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def extract_features(self, keypoints: np.ndarray, 
                        confidences: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract pose features from YOLO keypoints"""
        features = {}
        
        if confidences is not None:
            valid_mask = confidences > 0.3
            if not np.any(valid_mask):
                return features
        
        features.update(self._extract_body_features(keypoints))
        return features
    
    def _extract_body_features(self, kp: np.ndarray) -> Dict[str, float]:
        """Extract features from YOLO 17-point COCO pose"""
        features = {}
        
        if len(kp) < 17:
            return features
            
        try:
            # Body Proportions
            shoulder_width = self._distance(kp[5], kp[6])
            hip_width = self._distance(kp[11], kp[12])
            
            shoulder_center = (kp[5] + kp[6]) / 2
            hip_center = (kp[11] + kp[12]) / 2
            torso_height = self._distance(shoulder_center, hip_center)
            
            # Leg lengths
            left_upper_leg = self._distance(kp[11], kp[13])
            left_lower_leg = self._distance(kp[13], kp[15])
            left_leg = left_upper_leg + left_lower_leg
            
            right_upper_leg = self._distance(kp[12], kp[14])
            right_lower_leg = self._distance(kp[14], kp[16])
            right_leg = right_upper_leg + right_lower_leg
            
            avg_leg_length = (left_leg + right_leg) / 2
            
            # Arm lengths
            left_upper_arm = self._distance(kp[5], kp[7])
            left_lower_arm = self._distance(kp[7], kp[9])
            left_arm = left_upper_arm + left_lower_arm
            
            right_upper_arm = self._distance(kp[6], kp[8])
            right_lower_arm = self._distance(kp[8], kp[10])
            right_arm = right_upper_arm + right_lower_arm
            
            avg_arm_length = (left_arm + right_arm) / 2
            
            # Normalized Ratios (scale-invariant)
            if torso_height > 1e-6:
                features['shoulder_hip_ratio'] = shoulder_width / hip_width if hip_width > 1e-6 else 0
                features['leg_torso_ratio'] = avg_leg_length / torso_height
                features['arm_torso_ratio'] = avg_arm_length / torso_height
                features['shoulder_width_norm'] = shoulder_width / torso_height
                features['hip_width_norm'] = hip_width / torso_height
                features['upper_lower_leg_ratio'] = (left_upper_leg + right_upper_leg) / (left_lower_leg + right_lower_leg) if (left_lower_leg + right_lower_leg) > 1e-6 else 0
                features['upper_lower_arm_ratio'] = (left_upper_arm + right_upper_arm) / (left_lower_arm + right_lower_arm) if (left_lower_arm + right_lower_arm) > 1e-6 else 0
                
            # Joint Angles
            features['left_elbow_angle'] = self._angle(kp[5], kp[7], kp[9])
            features['right_elbow_angle'] = self._angle(kp[6], kp[8], kp[10])
            features['left_knee_angle'] = self._angle(kp[11], kp[13], kp[15])
            features['right_knee_angle'] = self._angle(kp[12], kp[14], kp[16])
            features['left_shoulder_angle'] = self._angle(kp[11], kp[5], kp[7])
            features['right_shoulder_angle'] = self._angle(kp[12], kp[6], kp[8])
            features['left_hip_angle'] = self._angle(kp[5], kp[11], kp[13])
            features['right_hip_angle'] = self._angle(kp[6], kp[12], kp[14])
            
            # Body Alignment
            features['body_tilt'] = self._body_tilt(shoulder_center, hip_center)
            features['shoulder_tilt'] = self._body_tilt(kp[5], kp[6])
            features['hip_tilt'] = self._body_tilt(kp[11], kp[12])
            
            # Symmetry Features
            if max(left_arm, right_arm) > 1e-6:
                features['arm_length_symmetry'] = 1 - abs(left_arm - right_arm) / max(left_arm, right_arm)
            if max(left_leg, right_leg) > 1e-6:
                features['leg_length_symmetry'] = 1 - abs(left_leg - right_leg) / max(left_leg, right_leg)
            
            features['elbow_angle_symmetry'] = 1 - abs(features['left_elbow_angle'] - features['right_elbow_angle']) / 180.0
            features['knee_angle_symmetry'] = 1 - abs(features['left_knee_angle'] - features['right_knee_angle']) / 180.0
            
            # Posture Features
            if len(kp) > 0:
                head_pos = kp[0]
                features['head_shoulder_distance'] = self._distance(head_pos, shoulder_center) / torso_height if torso_height > 1e-6 else 0
                
        except Exception as e:
            print(f"Warning: Feature extraction error: {e}")
            
        return features
    
    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return np.linalg.norm(p1 - p2)
    
    @staticmethod
    def _angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product < 1e-6:
            return 0.0
        cos_angle = np.dot(v1, v2) / norm_product
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    @staticmethod
    def _body_tilt(p1: np.ndarray, p2: np.ndarray) -> float:
        vector = p2 - p1
        angle = np.arctan2(vector[1], vector[0])
        return np.degrees(angle)


class PoseReIDMatcher:
    """Match persons using aggregated pose features from multiple frames"""
    
    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold
        self.gallery: Dict[int, List[PoseFeatures]] = {}  # person_id -> List of features
        
    def add_to_gallery(self, pose_features: PoseFeatures):
        """Add a person's pose features to gallery"""
        if pose_features.person_id not in self.gallery:
            self.gallery[pose_features.person_id] = []
        self.gallery[pose_features.person_id].append(pose_features)
        
    def match(self, query_features: PoseFeatures) -> Tuple[Optional[int], float, Dict]:
        """
        Match query against gallery using MULTIPLE frames
        
        Returns:
            (matched_person_id, best_similarity, debug_info)
        """
        if not self.gallery:
            return None, 0.0, {}
            
        best_match_id = None
        best_similarity = 0.0
        debug_info = {}
        
        query_vector = self._features_to_vector(query_features.features)
        
        for person_id, gallery_features_list in self.gallery.items():
            # Compare query with EACH registered frame and take the BEST match
            similarities = []
            
            for gallery_features in gallery_features_list:
                gallery_vector = self._features_to_vector(gallery_features.features)
                sim = self._compute_similarity(query_vector, gallery_vector)
                similarities.append(sim)
            
            # Take MAXIMUM similarity across all frames (best match)
            max_similarity = max(similarities) if similarities else 0.0
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            debug_info[person_id] = {
                'max_sim': max_similarity,
                'avg_sim': avg_similarity,
                'num_frames': len(gallery_features_list)
            }
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match_id = person_id
                
        if best_similarity >= self.similarity_threshold:
            return best_match_id, best_similarity, debug_info
        else:
            return None, best_similarity, debug_info
    
    @staticmethod
    def _features_to_vector(features: Dict[str, float]) -> np.ndarray:
        if not features:
            return np.array([])
        keys = sorted(features.keys())
        return np.array([features[k] for k in keys])
    
    @staticmethod
    def _compute_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        if len(v1) == 0 or len(v2) == 0 or len(v1) != len(v2):
            return 0.0
        try:
            cosine_sim = 1 - cosine(v1, v2)
            euclidean_sim = 1 / (1 + euclidean(v1, v2))
            similarity = 0.6 * cosine_sim + 0.4 * euclidean_sim
            return max(0.0, min(1.0, similarity))
        except:
            return 0.0


class PoseReIDSystem:
    """Complete Pose-based Re-Identification System using YOLOv8-Pose + ByteTrack"""
    
    def __init__(self, model_path: str = 'yolov8n-pose.pt'):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed. Install it with:\npip install ultralytics")
        
        print(f"Loading YOLO pose model: {model_path}")
        self.yolo_model = YOLO(model_path)
        self.feature_extractor = PoseFeatureExtractor()
        self.matcher = PoseReIDMatcher(similarity_threshold=0.65)
        print("✓ System initialized successfully")
        
        # Track ID mapping: track_id -> reid_person_id
        self.track_to_reid_map: Dict[int, int] = {}
        
    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List[Tuple[int, np.ndarray, np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detect and track all persons using ByteTrack
        
        Returns:
            List of (track_id, keypoints, confidences, bbox, avg_confidence)
        """
        # Use YOLO tracking (includes ByteTrack)
        results = self.yolo_model.track(frame, persist=persist, verbose=False, tracker="bytetrack.yaml")
        persons = []
        
        if len(results) == 0:
            return persons
        
        result = results[0]
        
        if result.boxes is None or result.keypoints is None:
            return persons
        
        # Check if tracking IDs are available
        if result.boxes.id is None:
            return persons
        
        num_persons = len(result.boxes)
        
        for i in range(num_persons):
            # Get tracking ID from ByteTrack
            track_id = int(result.boxes.id[i].item())
            
            # Get bbox
            box = result.boxes[i].xyxy[0].cpu().numpy()
            bbox = tuple(map(int, box))
            
            # Get keypoints
            kp = result.keypoints.xy[i].cpu().numpy()
            conf = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else np.ones(17)
            avg_conf = float(np.mean(conf))
            
            if avg_conf > 0.3:
                persons.append((track_id, kp, conf, bbox, avg_conf))
        
        return persons
    
    def process_frame(self, frame: np.ndarray, person_id: int, 
                     camera_id: str, bbox: Tuple[int, int, int, int],
                     timestamp: float, keypoints: np.ndarray,
                     confidences: np.ndarray, avg_conf: float, 
                     track_id: int = -1) -> Optional[PoseFeatures]:
        """Process a single frame to extract pose features"""
        features = self.feature_extractor.extract_features(keypoints, confidences)
        
        if not features:
            return None
        
        pose_features = PoseFeatures(
            person_id=person_id,
            features=features,
            keypoints=keypoints,
            timestamp=timestamp,
            camera_id=camera_id,
            bbox=bbox,
            confidence=avg_conf,
            track_id=track_id
        )
        
        return pose_features
    
    def register_person(self, pose_features: PoseFeatures):
        """Register a person in the gallery"""
        self.matcher.add_to_gallery(pose_features)
        
    def identify_person(self, pose_features: PoseFeatures) -> Tuple[Optional[int], float, Dict]:
        """Identify a person across cameras"""
        return self.matcher.match(pose_features)
    
    def update_track_mapping(self, track_id: int, reid_id: int):
        """Update the mapping between ByteTrack ID and ReID person ID"""
        self.track_to_reid_map[track_id] = reid_id


class MouseSelector:
    """Handle mouse selection of persons in video"""
    
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
            bbox = person_data[3]  # bbox is at index 3
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selecting = False
                self.selected_point = None
                return idx
        
        self.selecting = False
        self.selected_point = None
        return None


def process_video_1_registration(reid_system: PoseReIDSystem, video_path: str, 
                                  num_frames_to_collect: int = 30) -> Tuple[bool, int]:
    """
    Process first video - collect MULTIPLE frames of selected person with ByteTrack
    
    Returns:
        (registered_success, registered_person_id)
    """
    print("\n" + "=" * 70)
    print("VIDEO 1: Registration Phase (with ByteTrack)")
    print("=" * 70)
    print(f"Instructions: Click on a person to register them")
    print(f"System will collect {num_frames_to_collect} frames for robust matching")
    print("Press 'q' to quit without registering")
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
    window_name = "Video 1 - Click on person to register"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_selector.mouse_callback)
    
    registered = False
    registered_person_id = -1
    frame_count = 0
    paused = False
    
    # Collection mode variables
    collecting_mode = False
    selected_track_id = None
    frames_collected = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_count += 1
        
        # Detect and track persons using ByteTrack
        persons = reid_system.detect_and_track(frame, persist=True)
        display_frame = frame.copy()
        
        if not collecting_mode:
            # NORMAL MODE: Show all tracked persons
            for track_id, kp, conf, bbox, avg_conf in persons:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Track ID: {track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                for point in kp:
                    if point[0] > 0 and point[1] > 0:
                        cv2.circle(display_frame, tuple(map(int, point)), 3, (0, 255, 255), -1)
            
            cv2.putText(display_frame, "Click on a person to register", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Detected: {len(persons)} persons | Frame: {frame_count}/{total_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            # COLLECTION MODE: Track selected person and collect frames
            person_found = False
            for track_id, kp, conf, bbox, avg_conf in persons:
                if track_id == selected_track_id:
                    person_found = True
                    x1, y1, x2, y2 = bbox
                    
                    # Extract and register this frame
                    pose_features = reid_system.process_frame(
                        frame, person_id=1, camera_id="video1",
                        bbox=bbox, timestamp=frame_count/fps,
                        keypoints=kp, confidences=conf, avg_conf=avg_conf,
                        track_id=track_id
                    )
                    
                    if pose_features:
                        reid_system.register_person(pose_features)
                        frames_collected += 1
                        print(f"  Collected frame {frames_collected}/{num_frames_to_collect} (Track ID: {track_id}, conf: {avg_conf:.2f})")
                    
                    # Draw collection visualization
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(display_frame, f"COLLECTING Track ID: {track_id}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    for point in kp:
                        if point[0] > 0 and point[1] > 0:
                            cv2.circle(display_frame, tuple(map(int, point)), 3, (0, 0, 255), -1)
                    break
            
            if not person_found:
                cv2.putText(display_frame, f"Waiting for Track ID {selected_track_id}...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Progress bar
            progress = int((frames_collected / num_frames_to_collect) * 100)
            cv2.putText(display_frame, f"Collecting: {frames_collected}/{num_frames_to_collect} ({progress}%)", 
                       (10, 60 if not person_found else 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            bar_width = 400
            bar_height = 30
            bar_x, bar_y = 10, 90 if not person_found else 120
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            filled_width = int((frames_collected / num_frames_to_collect) * bar_width)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # Check if collection complete
            if frames_collected >= num_frames_to_collect:
                print(f"\n✓ Registration complete!")
                print(f"  Track ID: {selected_track_id}")
                print(f"  Total frames collected: {frames_collected}")
                print(f"  Gallery now has: {len(reid_system.matcher.gallery[1])} frames for person 1")
                
                cv2.putText(display_frame, "REGISTRATION COMPLETE!", 
                           (display_frame.shape[1]//4, display_frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(2000)
                
                registered = True
                registered_person_id = 1
                break
        
        cv2.imshow(window_name, display_frame)
        
        # Check for person selection
        if not collecting_mode:
            selected_idx = mouse_selector.get_selected_person(persons)
            if selected_idx is not None and selected_idx < len(persons):
                selected_track_id = persons[selected_idx][0]  # track_id is first element
                print(f"\n✓ Person with Track ID {selected_track_id} selected")
                print(f"  Starting collection of {num_frames_to_collect} frames...")
                collecting_mode = True
                frames_collected = 0
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not collecting_mode:
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    return registered, registered_person_id


def process_video_2_identification(reid_system: PoseReIDSystem, video_path: str, 
                                    registered_person_id: int):
    """Process second video - identify registered person with ByteTrack IDs"""
    print("\n" + "=" * 70)
    print("VIDEO 2: Identification Phase (with ByteTrack)")
    print("=" * 70)
    print("Searching for the registered person...")
    print(f"Gallery has {len(reid_system.matcher.gallery[registered_person_id])} reference frames")
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
    
    window_name = "Video 2 - Identifying person"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    matches_found = 0
    track_match_counts = defaultdict(int)  # Count matches per track_id
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and track persons
        persons = reid_system.detect_and_track(frame, persist=True)
        display_frame = frame.copy()
        
        for track_id, kp, conf, bbox, avg_conf in persons:
            x1, y1, x2, y2 = bbox
            
            # Extract features
            pose_features = reid_system.process_frame(
                frame, person_id=999, camera_id="video2",
                bbox=bbox, timestamp=frame_count/fps,
                keypoints=kp, confidences=conf, avg_conf=avg_conf,
                track_id=track_id
            )
            
            if pose_features:
                matched_id, similarity, debug_info = reid_system.identify_person(pose_features)
                
                # Check if this track_id already has a ReID mapping
                if track_id in reid_system.track_to_reid_map:
                    reid_id = reid_system.track_to_reid_map[track_id]
                    color = (0, 255, 0)
                    label = f"ReID:{reid_id} | Track:{track_id} ({similarity:.2f})"
                    thickness = 3
                elif matched_id is not None:
                    # NEW MATCH FOUND!
                    matches_found += 1
                    track_match_counts[track_id] += 1
                    
                    # Map this track_id to the matched reid_id
                    reid_system.update_track_mapping(track_id, matched_id)
                    
                    color = (0, 255, 0)
                    label = f"MATCH! ReID:{matched_id} | Track:{track_id} ({similarity:.2f})"
                    thickness = 3
                    
                    print(f"[Frame {frame_count}] ✓ MATCH! Track ID: {track_id} → ReID: {matched_id} | Sim: {similarity:.3f}")
                    print(f"  Debug: {debug_info[matched_id]}")
                else:
                    # No match
                    color = (0, 0, 255)
                    label = f"Track:{track_id} ({similarity:.2f})"
                    thickness = 2
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(display_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                for point in kp:
                    if point[0] > 0 and point[1] > 0:
                        cv2.circle(display_frame, tuple(map(int, point)), 3, color, -1)
        
        cv2.putText(display_frame, f"Searching for registered person...", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames} | Matches: {matches_found}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Active mappings: {len(reid_system.track_to_reid_map)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        
    
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Video 2 processing complete")
    print(f"  Total matches found: {matches_found}")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Match rate: {matches_found/frame_count*100:.1f}%")
    print(f"\n📊 Per-Track Match Statistics:")
    for track_id, count in sorted(track_match_counts.items()):
        reid_id = reid_system.track_to_reid_map.get(track_id, "Unknown")
        print(f"  Track ID {track_id} → ReID {reid_id}: {count} matches")


def main():
    """Two-video ReID demo with multi-frame registration"""
    
    print("=" * 70)
    print("Pose-Based Re-Identification System - Enhanced Version")
    print("=" * 70)
    
    if not YOLO_AVAILABLE:
        print("\n✗ ERROR: ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return
    
    video1_path = "../videos//4PTF1_short.mp4"
    video2_path = "../videos//4PTF2_short.mp4"
    
    if not video1_path or not video2_path:
        print("\n✗ Both video paths are required!")
        return
    
    try:
        reid_system = PoseReIDSystem(model_path='yolov8n-pose.pt')
        
        # Collect 30 frames for robust registration
        registered, registered_person_id = process_video_1_registration(
            reid_system, video1_path, num_frames_to_collect=60
        )
        process_video_2_identification(reid_system, video2_path, registered_person_id)
        
        print("\n✓ Demo complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()