import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import supervision as sv
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# --- 1. Lightweight Feature Extractor (ResNet50) ---
# This replaces the need for 'torchreid' or 'insightface' C++ compilations
class FeatureExtractor:
    def __init__(self, model_name='resnet50', device='cpu', verbose=False):
        import torchvision.models as models
        import torch.nn as nn
        from torchvision import transforms

        self.device = torch.device(device)
        # Load standard ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final classification layer to get the "embedding"
        backbone.fc = nn.Identity()
        backbone.to(self.device)
        backbone.eval()
        self.model = backbone

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, imgs):
        tensors = []
        for img in imgs:
            if img is None: continue
            # Ensure RGB
            if img.shape[-1] == 3:
                rgb = img[:, :, ::-1] # BGR to RGB
            else:
                rgb = img
            t = self.transform(rgb)
            tensors.append(t)

        if len(tensors) == 0:
            return torch.empty((0, 2048)).to(self.device)

        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        # Normalize features (Crucial for Cosine Similarity)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats

# --- 2. Multi-Modal Extractor (The Brain) ---
class MultiModalFeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        
        # A. Initialize Deep Learning Extractor (Fixes previous AttributeError)
        print(f"[MultiModal] Loading ResNet50 on {device}...")
        self.extractor = FeatureExtractor(device=self.device)
        
        # B. Initialize MediaPipe (Face & Pose)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
        
        print("[MultiModal] Feature extractor initialized")

    def extract_all_features(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Safety Clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return None

        features = {}

        # --- 1. REID (Global Appearance) ---
        try:
            # Returns tensor, convert to numpy
            features['reid'] = self.extractor([crop])[0].cpu().numpy()
        except Exception as e:
            print(f"ReID Error: {e}")
            features['reid'] = None

        # --- 2. HAIR/HEAD (Deep Texture) ---
        # Crop top 20% for head
        head_h = int((y2 - y1) * 0.20)
        head_crop = crop[0:head_h, :]
        if head_crop.size > 0:
            try:
                features['hair'] = self.extractor([head_crop])[0].cpu().numpy()
            except:
                features['hair'] = None
        else:
            features['hair'] = None

        # --- 3. FACE (MediaPipe Geometry) ---
        features['face'] = self._extract_face(image, bbox)

        # --- 4. BODY SHAPE (MediaPipe Geometry) ---
        features['body'] = self._extract_body(image, bbox)

        return features

    def _extract_face(self, image, bbox):
        """Extracts facial landmarks if visible"""
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return None
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        face_data = {}
        try:
            results = self.face_mesh.process(rgb_crop)
            if results.multi_face_landmarks:
                # Get the first 50 landmarks as a flat array
                landmarks = results.multi_face_landmarks[0]
                points = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
                face_data['landmarks'] = points.flatten()[:150] # 50 points * 3 coords
        except: pass
        return face_data if face_data else None

    def _extract_body(self, image, bbox):
        """Extracts shoulder/hip ratios"""
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = image.shape[:2]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return None
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        body_data = {}
        try:
            results = self.pose.process(rgb_crop)
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                # Calculate Ratio: Shoulder Width / Torso Height
                left_sh = lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_sh = lms[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = lms[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                
                # Check visibility
                if left_sh.visibility > 0.5 and right_sh.visibility > 0.5 and left_hip.visibility > 0.5:
                    sh_width = abs(left_sh.x - right_sh.x)
                    torso_h = abs(left_sh.y - left_hip.y)
                    if torso_h > 0:
                        body_data['ratio'] = sh_width / torso_h
        except: pass
        return body_data if body_data else None

# --- 3. Main System ---
class EnhancedPersonReIDSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[System] Running on: {self.device}")

        # Models
        self.detection_model = YOLO('yolov8n.pt')
        self.tracker = sv.ByteTrack()
        self.extractor = MultiModalFeatureExtractor(device=self.device)

        # Config
        self.target_features = None
        self.target_tracker_id = None
        self.cam1_source = "videos/4PTF1.avi" # CHANGE THESE PATHS
        self.cam2_source = "videos/4PTF2.avi" # CHANGE THESE PATHS
        
        # Thresholds
        self.similarity_threshold = 0.60 

        # Colors
        self.colors = {'match': (0, 255, 0), 'no_match': (0, 0, 255), 'target': (255, 255, 0)}
        self.current_frame = None
        self.current_detections = None

    def calculate_similarity(self, target, candidate):
        if target is None or candidate is None: return 0.0
        
        score = 0.0
        weights = 0.0

        # 1. ReID (Weight: 40%)
        if target['reid'] is not None and candidate['reid'] is not None:
            s = 1.0 - cosine(target['reid'], candidate['reid'])
            score += s * 0.4
            weights += 0.4

        # 2. Hair (Weight: 30%) - Strong for uniforms
        if target['hair'] is not None and candidate['hair'] is not None:
            s = 1.0 - cosine(target['hair'], candidate['hair'])
            score += s * 0.3
            weights += 0.3

        # 3. Face (Weight: 20%)
        if target['face'] is not None and candidate['face'] is not None:
            # Compare landmarks
            t_l = target['face'].get('landmarks')
            c_l = candidate['face'].get('landmarks')
            if t_l is not None and c_l is not None:
                s = 1.0 - cosine(t_l, c_l)
                score += s * 0.2
                weights += 0.2

        # 4. Body (Weight: 10%)
        if target['body'] is not None and candidate['body'] is not None:
            t_r = target['body'].get('ratio')
            c_r = candidate['body'].get('ratio')
            if t_r and c_r:
                diff = abs(t_r - c_r)
                s = max(0, 1.0 - diff) # Closer ratio = higher score
                score += s * 0.1
                weights += 0.1

        return score / weights if weights > 0 else 0.0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_detections is None: return
            print(f"[Click] ({x}, {y})")
            
            for (x1, y1, x2, y2, _, class_id, tracker_id) in self.current_detections:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"Extracting ID {tracker_id}...")
                    bbox = [x1, y1, x2, y2]
                    self.target_features = self.extractor.extract_all_features(self.current_frame, bbox)
                    
                    if self.target_features:
                        self.target_tracker_id = tracker_id
                        print(f"✅ TARGET LOCKED: ID {tracker_id}")
                    return

    def run(self):
        # Phase 1: Select
        cap = cv2.VideoCapture(self.cam1_source)
        cv2.namedWindow("Select Target")
        cv2.setMouseCallback("Select Target", self.mouse_callback)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Detect
            results = self.detection_model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)
            
            self.current_frame = frame.copy()
            self.current_detections = []
            
            annotated = frame.copy()
            for xyxy, conf, cls, tid in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
                self.current_detections.append((*xyxy, conf, cls, tid))
                color = self.colors['target'] if tid == self.target_tracker_id else (0,0,255)
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"ID {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(annotated, "CLICK TO SELECT. PRESS 'C' TO SEARCH.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Select Target", annotated)
            
            k = cv2.waitKey(30)
            if k == ord('q'): return
            if k == ord('c') and self.target_tracker_id is not None: break
        
        cap.release()
        cv2.destroyWindow("Select Target")

        # Phase 2: Search
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
                
                # Extract features for candidate
                candidate_feats = self.extractor.extract_all_features(frame, [x1, y1, x2, y2])
                
                # Compare
                sim = self.calculate_similarity(self.target_features, candidate_feats)
                
                is_match = sim > self.similarity_threshold
                color = self.colors['match'] if is_match else self.colors['no_match']
                
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                label = f"Match {sim:.2f}" if is_match else f"ID {tid}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Search", frame)
            if cv2.waitKey(1) == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EnhancedPersonReIDSystem()
    app.run()