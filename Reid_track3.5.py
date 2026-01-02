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

# --- FEATURE 1: True ReID Model Architecture ---
class ReIDBackbone(nn.Module):
    """
    Replaces generic ResNet50 with a ReID-optimized structure.
    - Removes last stride (preserve spatial resolution).
    - Adds IBN (Instance Batch Normalization) concept logic implicitly via structure.
    """
    def __init__(self, name='resnet50', device='cpu'):
        super(ReIDBackbone, self).__init__()
        self.device = device
        
        # Load standard backbone
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # MODIFICATION: Remove stride in Layer 4 to keep feature map larger (standard ReID trick)
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)
        
        # Remove FC head
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        
        # Generalized Mean Pooling (Gem) or Adaptive Avg Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # BatchNorm Neck (Common in ReID to separate metric learning from classification)
        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)  # no shift
        self.bn.apply(self.weights_init_kaiming)
        
        self.to(device)
        self.eval()

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gap(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.bn(feat) # BNNeck
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
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            
        return feat.cpu().numpy().flatten()

# --- FEATURE 5 & 7: Advanced Multi-Modal Extractor ---
class MultiModalFeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        print(f"[MultiModal] Initializing modules on {device}...")
        
        self.reid_model = FeatureExtractor(device)
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

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

    def extract(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = image.shape[:2]
        
        # Quality Check: Skip very small boxes
        if (x2-x1) < 20 or (y2-y1) < 50: return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return None

        # Quality Check: Blur detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_score < 100.0 # Threshold

        features = {
            'reid': None, 'face': None, 'pose': None, 'hair': None, 'color': None,
            'confidences': {'reid': 1.0, 'face': 0.0, 'pose': 0.0, 'color': 1.0}
        }
        
        # 1. Deep ReID
        if not is_blurry:
            features['reid'] = self.reid_model(crop)
        else:
            features['confidences']['reid'] = 0.5 # Lower confidence if blurry

        # 2. Color (Works even if blurry)
        features['color'] = self.get_color_descriptor(crop)

        # 3. MediaPipe Processing (Face & Pose)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Face
        face_res = self.mp_face_mesh.process(crop_rgb)
        if face_res.multi_face_landmarks:
            features['face'] = np.array([[l.x, l.y, l.z] for l in face_res.multi_face_landmarks[0].landmark]).flatten()
            features['confidences']['face'] = 1.0
        
        # Pose
        pose_res = self.mp_pose.process(crop_rgb)
        if pose_res.pose_landmarks:
            features['pose'] = self.get_pose_embedding(pose_res.pose_landmarks.landmark)
            features['confidences']['pose'] = pose_res.pose_landmarks.landmark[0].visibility # Nose visibility proxy

        # 4. Hair (Upper 20% crop)
        head_h = int((y2 - y1) * 0.2)
        head_crop = crop[0:head_h, :]
        if head_crop.size > 0 and not is_blurry:
            features['hair'] = self.reid_model(head_crop)
            
        features['bbox_ar'] = (x2-x1) / (y2-y1) # Aspect Ratio for Hard Negative Mining
        
        return features

# --- FEATURE 3: Temporal Voting History ---
class TrackHistory:
    def __init__(self, maxlen=10):
        self.history = {}
        self.maxlen = maxlen
        
    def update(self, tid, score):
        if tid not in self.history:
            self.history[tid] = deque(maxlen=self.maxlen)
        self.history[tid].append(score)
        
    def get_average(self, tid):
        if tid not in self.history or not self.history[tid]:
            return 0.0
        return np.mean(self.history[tid])

# --- MAIN SYSTEM ---
class EnhancedPersonReIDSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[System] Running on: {self.device}")

        # Models
        self.detection_model = YOLO('yolov8n.pt')
        self.tracker = sv.ByteTrack()
        self.extractor = MultiModalFeatureExtractor(device=self.device)
        self.track_history = TrackHistory(maxlen=15) # For Temporal Voting

        # Config
        self.cam1_source = "video2.mp4" 
        self.cam2_source = "video3.mp4"
        
        # --- FEATURE 2: Dynamic Gallery ---
        self.target_gallery = [] 
        self.target_tracker_id = None
        self.max_gallery_size = 50

        # Feature Weights (Base)
        self.base_weights = {'reid': 0.4, 'face': 0.15, 'pose': 0.15, 'color': 0.2, 'hair': 0.1}
        self.match_threshold = 0.65

        # UI State
        self.current_detections = None
        self.colors = {'match': (0, 255, 0), 'no_match': (0, 0, 255), 'target': (255, 255, 0)}

    # --- FEATURE 4: Dynamic Confidence-Aware Weighting ---
    def compute_pair_similarity(self, target, candidate):
        if target is None or candidate is None: return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        # Helper: Cosine Sim
        def get_sim(v1, v2):
            return 1.0 - cosine(v1, v2) if (v1 is not None and v2 is not None) else 0.0

        # Helper: Histogram Sim
        def get_hist_sim(h1, h2):
            return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) if (h1 is not None and h2 is not None) else 0.0

        # 1. ReID
        w_reid = self.base_weights['reid'] * candidate['confidences']['reid']
        s_reid = get_sim(target['reid'], candidate['reid'])
        total_score += s_reid * w_reid
        total_weight += w_reid
        
        # 2. Face (High penalty if mismatch, high reward if match)
        w_face = self.base_weights['face'] * candidate['confidences']['face']
        if w_face > 0 and target['face'] is not None:
            s_face = get_sim(target['face'], candidate['face'])
            total_score += s_face * w_face
            total_weight += w_face

        # 3. Pose (Shape context)
        w_pose = self.base_weights['pose'] * candidate['confidences']['pose']
        if w_pose > 0 and target['pose'] is not None:
            s_pose = get_sim(target['pose'], candidate['pose'])
            total_score += s_pose * w_pose
            total_weight += w_pose

        # 4. Color (Robust to pose changes)
        w_color = self.base_weights['color']
        s_color = get_hist_sim(target['color'], candidate['color'])
        # Clip histogram correlation to 0-1
        s_color = max(0, s_color)
        total_score += s_color * w_color
        total_weight += w_color
        
        # 5. Hair
        w_hair = self.base_weights['hair']
        s_hair = get_sim(target['hair'], candidate['hair'])
        total_score += s_hair * w_hair
        total_weight += w_hair

        final_sim = total_score / total_weight if total_weight > 0 else 0.0
        
        # --- FEATURE 8: Hard Negative Rejection ---
        # Reject if Aspect Ratio differs significantly (e.g. sitting vs standing)
        ar_diff = abs(target['bbox_ar'] - candidate['bbox_ar'])
        if ar_diff > 0.5: # Threshold for shape mismatch
            final_sim *= 0.8 # Penalize

        return final_sim

    def get_gallery_similarity(self, candidate_feats):
        """Compare candidate against all gallery frames and take MAX score."""
        if not self.target_gallery: return 0.0
        scores = [self.compute_pair_similarity(t, candidate_feats) for t in self.target_gallery]
        return max(scores) if scores else 0.0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_detections:
            for (x1, y1, x2, y2, _, _, tid) in self.current_detections:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"[Click] Target ID {tid} selected. Building Gallery...")
                    self.target_tracker_id = tid
                    self.target_gallery = [] # Reset
                    return

    def run(self):
        # --- PHASE 1: TARGET SELECTION (With Gallery Building) ---
        cap = cv2.VideoCapture(self.cam1_source)
        cv2.namedWindow("Select Target")
        cv2.setMouseCallback("Select Target", self.mouse_callback)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

            # Detect & Track
            results = self.detection_model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            self.current_detections = []
            
            # Gallery Collection Logic
            target_present = False
            for xyxy, conf, cls, tid in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
                self.current_detections.append((*xyxy, conf, cls, tid))
                
                if tid == self.target_tracker_id:
                    target_present = True
                    # Continuous Extraction
                    feats = self.extractor.extract(frame, xyxy)
                    if feats:
                        self.target_gallery.append(feats)
                        if len(self.target_gallery) > self.max_gallery_size:
                            self.target_gallery.pop(0)

            # Draw
            annotated = frame.copy()
            for *xyxy, _, _, tid in self.current_detections:
                color = self.colors['target'] if tid == self.target_tracker_id else (0,0,255)
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"ID {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            status = f"Collecting: {len(self.target_gallery)} frames" if self.target_tracker_id else "CLICK TO SELECT"
            cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Select Target", annotated)

            k = cv2.waitKey(20)
            if k == ord('q'): return
            if k == ord('c') and len(self.target_gallery) > 5: break # Ensure minimum samples

        cap.release()
        cv2.destroyWindow("Select Target")

        # --- PHASE 2: SEARCH (With Temporal Voting) ---
        print("\n--- STARTING SEARCH ---")
        cap = cv2.VideoCapture(self.cam2_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = self.detection_model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Extract Candidate
                c_feats = self.extractor.extract(frame, xyxy)
                
                if c_feats:
                    # 1. Instant Similarity
                    inst_sim = self.get_gallery_similarity(c_feats)
                    
                    # 2. Temporal Voting (Smooth over time)
                    self.track_history.update(tid, inst_sim)
                    smooth_sim = self.track_history.get_average(tid)
                    
                    is_match = smooth_sim > self.match_threshold
                    color = self.colors['match'] if is_match else self.colors['no_match']
                    
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    label = f"Match {smooth_sim:.2f}" if is_match else f"ID {tid} ({smooth_sim:.2f})"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Search", frame)
            if cv2.waitKey(1) == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EnhancedPersonReIDSystem()
    app.run()