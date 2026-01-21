import os
import random
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from ultralytics import YOLO
import supervision as sv
import mediapipe as mp
from collections import deque
from sklearn.preprocessing import normalize
import torchreid


# --- FEATURE 6: Cross-Camera Domain Normalization ---
class ImagePreProcessor:
    """Standardizes image appearance across different cameras."""
    def __init__(self):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, img):
        if img is None or img.size == 0: return img
        
        # 1. Convert to LAB color space to isolate luminance
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Apply CLAHE to L-channel (Lighting Normalization)
        l = self.clahe.apply(l)
        
        # 3. Merge back
        lab = cv2.merge((l, a, b))
        img_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return img_norm


def set_deterministic(seed: int = 42):
    """Set seeds for reproducibility without altering model numerics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- FEATURE 1: True ReID Model Architecture ---
class ReIDBackbone(nn.Module):
    """
    OSNet backbone pretrained for person ReID.
    Outputs L2-normalized embeddings suitable for cosine similarity.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=0,      # dummy, not used
            pretrained=True
        )

        self.model.eval()
        self.model.to(device)

    def forward(self, x):
        # torchreid models already output pooled features
        feat = self.model(x)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = ReIDBackbone(device=device)
        self.preprocessor = ImagePreProcessor()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        if img is None: return None
        
        # Apply Domain Normalization before Deep Feature Extraction
        img = self.preprocessor.preprocess(img)
        
        # RGB Conversion
        if img.shape[-1] == 3:
            img = img[:, :, ::-1]
            
        t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model(t)
            # Normalize for Cosine Similarity
            
        return feat.cpu().numpy().flatten()

# --- FEATURE 5 & 7: Advanced Multi-Modal Extractor ---
class MultiModalFeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        print(f"[MultiModal] Initializing modules on {device}...")
        
        self.reid_model = FeatureExtractor(device)
        # Cache face/pose per track to avoid redundant work while preserving correctness
        self.face_cache = {}
        self.pose_cache = {}
        self.face_pose_interval = 2  # frames between re-processing
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

    def get_color_descriptor(self, image):
        """FEATURE 5: HSV Histogram for color matching"""
        if image is None or image.size == 0: return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Extract histogram for Hue (color) and Saturation
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def get_pose_embedding(self, landmarks):
        """FEATURE 7: Full Pose Embedding (Normalized Keypoints)"""
        if not landmarks: return None
        
        # Convert to numpy
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Normalize: Center around mid-hip
        left_hip = lm_array[23]
        right_hip = lm_array[24]
        mid_hip = (left_hip + right_hip) / 2
        
        # Scale: Torso height
        left_shoulder = lm_array[11]
        torso_size = np.linalg.norm(left_shoulder - left_hip) + 1e-6
        
        norm_pose = (lm_array - mid_hip) / torso_size
        return norm_pose.flatten() # 33 points * 3 dims = 99 dim vector

    def extract(self, image, bbox, track_id=None, frame_idx=None, detection_conf=None):
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = image.shape[:2]
        
        # Quality Check: Skip very small boxes
        if (x2-x1) < 20 or (y2-y1) < 50: return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return None

        # Quality Check: Blur detection (blurry crops are discarded for ReID correctness)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_score < 20.0 # Threshold
        if is_blurry:
            return None

        features = {
            'reid': None, 'face': None, 'pose': None, 'hair': None, 'color': None,
            'confidences': {'reid': 1.0, 'face': 0.0, 'pose': 0.0, 'color': 1.0}
        }
        
        # 1. Deep ReID (required core embedding)
        reid_feat = self.reid_model(crop)
        if reid_feat is None or not np.isfinite(reid_feat).all():
            return None
        features['reid'] = reid_feat

        # 2. Color (Works even if blurry)
        features['color'] = self.get_color_descriptor(crop)

        # 3. MediaPipe Processing (Face & Pose)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cache_key = track_id if track_id is not None else id(crop)
        use_cache = frame_idx is not None and track_id is not None

        # Face (reuse within interval to preserve speed without changing semantics)
        if use_cache and cache_key in self.face_cache:
            last_frame, cached_face = self.face_cache[cache_key]
            if frame_idx - last_frame < self.face_pose_interval:
                features['face'] = cached_face
                features['confidences']['face'] = 1.0 if cached_face is not None else 0.0
        if features['face'] is None:
            face_res = self.mp_face_mesh.process(crop_rgb)
            if face_res.multi_face_landmarks:
                face_vec = np.array([[l.x, l.y, l.z] for l in face_res.multi_face_landmarks[0].landmark]).flatten()
                features['face'] = face_vec
                features['confidences']['face'] = 1.0
                if use_cache:
                    self.face_cache[cache_key] = (frame_idx, face_vec)
            elif use_cache:
                self.face_cache[cache_key] = (frame_idx, None)
        
        # Pose (reuse within interval)
        if use_cache and cache_key in self.pose_cache:
            last_frame, cached_pose = self.pose_cache[cache_key]
            if frame_idx - last_frame < self.face_pose_interval:
                features['pose'] = cached_pose
                features['confidences']['pose'] = 1.0 if cached_pose is not None else 0.0
        if features['pose'] is None:
            pose_res = self.mp_pose.process(crop_rgb)
            if pose_res.pose_landmarks:
                pose_vec = self.get_pose_embedding(pose_res.pose_landmarks.landmark)
                features['pose'] = pose_vec
                features['confidences']['pose'] = pose_res.pose_landmarks.landmark[0].visibility # Nose visibility proxy
                if use_cache:
                    self.pose_cache[cache_key] = (frame_idx, pose_vec)
            elif use_cache:
                self.pose_cache[cache_key] = (frame_idx, None)

        # 4. Hair (Upper 20% crop)
        head_h = int((y2 - y1) * 0.2)
        head_crop = crop[0:head_h, :]
        if head_crop.size > 0:
            hair_feat = self.reid_model(head_crop)
            if hair_feat is not None and np.isfinite(hair_feat).all():
                features['hair'] = hair_feat
            
        features['bbox_ar'] = (x2-x1) / (y2-y1) # Aspect Ratio for gating
        features['det_conf'] = detection_conf if detection_conf is not None else 1.0
        return features

    def clear_track_caches(self):
        self.face_cache.clear()
        self.pose_cache.clear()
        

# --- FEATURE 3: Temporal Voting History ---
class TrackHistory:
    def __init__(self, maxlen=10):
        self.history = {}
        self.maxlen = maxlen
        self.last_seen = {}
        
    def update(self, tid, score, frame_idx=None):
        if tid not in self.history:
            self.history[tid] = deque(maxlen=self.maxlen)
        self.history[tid].append(score)
        if frame_idx is not None:
            self.last_seen[tid] = frame_idx
        
    def get_average(self, tid):
        if tid not in self.history or not self.history[tid]:
            return 0.0
        return np.mean(self.history[tid])

    def expire(self, current_frame, max_age=120):
        """Drop track histories that have been inactive for too long."""
        stale_ids = [tid for tid, last in self.last_seen.items() if current_frame - last > max_age]
        for tid in stale_ids:
            self.history.pop(tid, None)
            self.last_seen.pop(tid, None)

# --- MAIN SYSTEM ---
class EnhancedPersonReIDSystem:
    def __init__(self):
        set_deterministic(42)
        self.locked_target_tid = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # safe speedup for fixed input sizes
        print(f"[System] Running on: {self.device}")

        # Models
        self.detection_model = YOLO('yolov8n.pt')
        self.detection_model.to(self.device)
        self.tracker = sv.ByteTrack()
        self.extractor = MultiModalFeatureExtractor(device=self.device)
        self.track_history = TrackHistory(maxlen=15) # For Temporal Voting

        # Config
        # Use local sample clips in the videos folder
        self.cam1_source = "videos/4PTF2_short.mp4" 
        self.cam2_source = "videos/4PTF4_short.mp4"
        
        # --- FEATURE 2: Dynamic Gallery ---
        self.target_gallery = [] 
        self.target_tracker_id = None
        self.max_gallery_size = 25

        # Feature Weights (Base)
        self.base_weights = {'reid': 0.4, 'face': 0.15, 'pose': 0.15, 'color': 0.2, 'hair': 0.1}
        self.match_threshold = 0.75

        # UI State
        self.current_detections = None
        self.colors = {'match': (0, 255, 0), 'no_match': (0, 0, 255), 'target': (255, 255, 0)}

    # --- FEATURE 4: Dynamic Confidence-Aware Weighting ---
    def compute_pair_similarity(self, target, candidate):
        if target is None or candidate is None: return 0.0
        if target.get('reid') is None or candidate.get('reid') is None:
            return 0.0

        # Aspect Ratio Gating (reject early on shape mismatch)
        if 'bbox_ar' in target and 'bbox_ar' in candidate:
            ar_diff = abs(target['bbox_ar'] - candidate['bbox_ar'])
            if ar_diff > 0.45:
                return 0.0
        
        def get_sim(v1, v2):
            return 1.0 - cosine(v1, v2) if (v1 is not None and v2 is not None) else 0.0

        def get_hist_sim(h1, h2):
            return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) if (h1 is not None and h2 is not None) else 0.0

        sims_weights = []

        reid_conf = candidate['confidences'].get('reid', 1.0)
        if target['reid'] is not None and candidate['reid'] is not None:
            w_reid = self.base_weights['reid'] * reid_conf
            sims_weights.append((get_sim(target['reid'], candidate['reid']), w_reid))

        face_conf = candidate['confidences'].get('face', 0.0)
        # Gate only: reject if grossly inconsistent
        if face_conf > 0 and target.get('face') is not None and candidate.get('face') is not None:
            if cosine(target['face'], candidate['face']) > 0.6:
                return 0.0


        pose_conf = candidate['confidences'].get('pose', 0.0)
        if pose_conf > 0 and target.get('pose') is not None and candidate.get('pose') is not None:
            if cosine(target['pose'], candidate['pose']) > 0.8:
                return 0.0

        # Color and hair are attenuated by ReID confidence to prevent dominance
        if candidate.get('color') is not None and target.get('color') is not None and reid_conf >= 0.5:
            w_color = self.base_weights['color'] * min(reid_conf, 0.5)
            s_color = max(0.0, get_hist_sim(target['color'], candidate['color']))
            sims_weights.append((s_color, w_color))

        if candidate.get('hair') is not None and target.get('hair') is not None and reid_conf >= 0.5:
            w_hair = self.base_weights['hair'] * min(reid_conf, 0.5)
            sims_weights.append((get_sim(target['hair'], candidate['hair']), w_hair))

        total_weight = sum(w for _, w in sims_weights if w > 0)
        if total_weight == 0:
            return 0.0

        final_sim = sum(score * w for score, w in sims_weights) / total_weight
        return final_sim

    def get_gallery_similarity(self, candidate_feats, topk=3):
        if not self.target_gallery:
            return 0.0

        scores = [
            self.compute_pair_similarity(t, candidate_feats)
            for t in self.target_gallery
        ]

        # Hard ReID consistency gate
        scores = [s for s in scores if s > 0.65]

        if len(scores) < 3:
            return 0.0
        scores.sort(reverse=True)
        return float(np.mean(scores[:min(topk, len(scores))]))



    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_detections:
            for (x1, y1, x2, y2, _, _, tid) in self.current_detections:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"[Click] Target ID {tid} selected. Building Gallery...")
                    self.target_tracker_id = tid
                    self.target_gallery = [] # Reset
                    return

    def run(self):
        # Basic source validation
        if not os.path.exists(self.cam1_source):
            print(f"[Error] Camera 1 source not found: {self.cam1_source}")
            return
        if not os.path.exists(self.cam2_source):
            print(f"[Error] Camera 2 source not found: {self.cam2_source}")
            return

        # --- PHASE 1: TARGET SELECTION (With Gallery Building) ---
        cap = cv2.VideoCapture(self.cam1_source)
        if not cap.isOpened():
            print(f"[Error] Unable to open source: {self.cam1_source}")
            return
        try:
            cv2.namedWindow("Select Target")
        except cv2.error:
            print("[Error] Unable to create display window (headless environment?).")
            cap.release()
            return
        cv2.setMouseCallback("Select Target", self.mouse_callback)

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[Video] End reached")
                break


            # Detect & Track
            results = self.detection_model(frame, classes=[0], verbose=False, device=self.device)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            self.current_detections = []
            
            # Gallery Collection Logic
            target_present = False
            for xyxy, conf, cls, tid in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
                self.current_detections.append((*xyxy, conf, cls, tid))
                
                if tid == self.target_tracker_id:
                    target_present = True
                    if frame_idx % 5 != 0:
                        continue
                    # Continuous Extraction with quality gating
                    feats = self.extractor.extract(frame, xyxy, track_id=tid, frame_idx=frame_idx, detection_conf=conf)
                    if feats and feats.get('reid') is not None and conf is not None and conf > 0.25:
                        # Diversity filter to keep gallery clean
                        is_diverse = True
                        for existing in self.target_gallery:
                            if existing.get('reid') is None:
                                continue
                            sim = 1.0 - cosine(existing['reid'], feats['reid'])
                            if sim > 0.98:
                                is_diverse = False
                                break
                        if is_diverse:
                            self.target_gallery.append(feats)

                            if len(self.target_gallery) > self.max_gallery_size:
                                self.target_gallery.pop(0)
                            
                            print(f"[Gallery] Size = {len(self.target_gallery)} | conf={conf:.2f}")

            # Draw
            annotated = frame.copy()
            for *xyxy, _, _, tid in self.current_detections:
                color = self.colors['target'] if tid == self.target_tracker_id else (0,0,255)
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"ID {tid}", (x1 + 2, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            status = f"Collecting: {len(self.target_gallery)} frames" if self.target_tracker_id else "CLICK TO SELECT"
            cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Select Target", annotated)

            k = cv2.waitKey(20)
            if k == ord('q'): 
                cap.release()
                cv2.destroyWindow("Select Target")
                return
            if k == ord('c'):
                if len(self.target_gallery) >= 10:
                    print(f"[Gallery] Finalized with {len(self.target_gallery)} samples")
                    break
                else:
                    print("[Gallery] Need at least 10 clean samples before continuing")

        cap.release()
        cv2.destroyWindow("Select Target")

        # Reset tracker and caches between streams to avoid cross-contamination
        self.tracker = sv.ByteTrack()
        self.extractor.clear_track_caches()
        self.track_history = TrackHistory(maxlen=15)

        # --- PHASE 2: SEARCH (With Temporal Voting) ---
        print("\n--- STARTING SEARCH ---")
        cap = cv2.VideoCapture(self.cam2_source)
        if not cap.isOpened():
            print(f"[Error] Unable to open source: {self.cam2_source}")
            return
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            results = self.detection_model(frame, classes=[0], verbose=False, device=self.device)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            for xyxy, conf, tid in zip(detections.xyxy, detections.confidence, detections.tracker_id):
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Extract Candidate
                c_feats = self.extractor.extract(frame, xyxy, track_id=tid, frame_idx=frame_idx, detection_conf=conf)
                
                if c_feats and c_feats.get('reid') is not None:
                    # 1. Instant Similarity
                    inst_sim = self.get_gallery_similarity(c_feats)

                    #Update temporal history
                    self.track_history.update(tid, inst_sim, frame_idx=frame_idx)
                    smooth_sim = self.track_history.get_average(tid)

                    # ---- IDENTITY LOCK LOGIC ----
                    if self.locked_target_tid is None:
                        if smooth_sim > self.match_threshold and inst_sim > 0.7:
                            self.locked_target_tid = tid
                            is_match = True

                        else:
                            is_match = False
                    else:
                        if tid == self.locked_target_tid:
                            is_match = smooth_sim > 0.55  # relaxed for same identity
                        else:
                            is_match = False
                    # --------------------------------

                    # Draw bounding box
                    color = self.colors['match'] if is_match else self.colors['no_match']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label strictly based on locked identity
                    if tid == self.locked_target_tid:
                        label = f"TARGET ({smooth_sim:.2f})"
                    else:
                        label = f"ID {tid} ({smooth_sim:.2f})"

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 2, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            self.track_history.expire(frame_idx, max_age=180)
            cv2.imshow("Search", frame)
            if cv2.waitKey(1) == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EnhancedPersonReIDSystem()
    app.run()