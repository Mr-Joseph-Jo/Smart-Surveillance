import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import supervision as sv
try:
    from torchreid.utils import FeatureExtractor  # <--- The safer import
except Exception:
    # Lightweight fallback FeatureExtractor using torchvision/resnet50
    # This avoids a hard dependency on the `torchreid` package while
    # providing usable embeddings for similarity checks.
    class FeatureExtractor:
        def __init__(self, model_name='resnet50', device='cpu', verbose=False):
            import torch
            import torchvision.models as models
            import torch.nn as nn
            from torchvision import transforms

            self.device = torch.device(device if device != 'cuda' else 'cuda')

            # Use ResNet50 backbone, remove final fc
            backbone = models.resnet50(weights=getattr(models, 'ResNet50_Weights', None).DEFAULT if hasattr(models, 'ResNet50_Weights') else None)
            backbone.fc = nn.Identity()
            backbone.to(self.device)
            backbone.eval()
            self.model = backbone

            # Preprocessing similar to ImageNet (resize to person-reid-friendly shape)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),  # (H, W): taller than wide
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __call__(self, imgs):
            """Accepts a list of BGR numpy images and returns a torch.Tensor (N, D).
            This matches the minimal expected behavior used in the repository.
            """
            import torch
            tensors = []
            for img in imgs:
                # img is BGR numpy array from OpenCV; convert to RGB
                if img is None:
                    continue
                rgb = img[:, :, ::-1]
                t = self.transform(rgb)
                tensors.append(t)

            if len(tensors) == 0:
                return torch.empty((0, 2048)).to(self.device)

            batch = torch.stack(tensors, dim=0).to(self.device)
            with torch.no_grad():
                feats = self.model(batch)

            # L2 normalize
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            return feats
from datetime import datetime

class PersonReIDSystem:
    def __init__(self):
        # --- Hardware Config ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[System] Running on: {self.device}")

        # --- Initialize Models ---
        self.detection_model = YOLO('yolov8n.pt')
        self.tracker = sv.ByteTrack()
        
        # USE LIBRARY UTILITY instead of manual build_model
        # This handles the resize (128x256), normalization, and GPU casting automatically.
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            device=self.device,
            verbose=False
        )

        # --- State Management ---
        self.target_embedding = None
        self.target_tracker_id = None
        
        # --- Config ---
        self.similarity_threshold = 0.70
        self.cam1_source = "samplesurv_one.mp4"
        self.cam2_source = "samplesurv_two_.mp4"
        
        # --- Visualization ---
        self.colors = {
            'match': (0, 255, 0),      # Green
            'no_match': (0, 0, 255),   # Red
            'target': (255, 255, 0),   # Cyan/Yellow
            'highlight': (255, 0, 255) # Magenta
        }
        
        self.current_frame = None
        self.current_detections = None

    def get_embedding(self, image, bbox):
        """
        Safer extraction using the library utility.
        Handles boundary checks and preprocessing internally.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Safety Clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1: return None
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return None

        # The FeatureExtractor takes a list of images (batch processing)
        # It returns a tensor of shape (1, 512)
        features = self.extractor([crop]) 
        
        # Convert to flat numpy array for Scipy
        return features[0].cpu().numpy()

    def calculate_similarity(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return 0.0
        # Cosine distance: 0 = same, 1 = different
        # Similarity: 1 = same, 0 = different
        return 1.0 - cosine(embedding1, embedding2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_detections is None: return

            print(f"[Click] Coordinates: ({x}, {y})")
            
            # Check which box was clicked
            for (x1, y1, x2, y2, _, class_id, tracker_id) in self.current_detections:
                if x1 <= x <= x2 and y1 <= y <= y2 and class_id == 0:
                    
                    print(f"[Processing] Extracting features for ID {tracker_id}...")
                    bbox = [x1, y1, x2, y2]
                    
                    # Extract
                    emb = self.get_embedding(self.current_frame, bbox)
                    
                    if emb is not None:
                        self.target_embedding = emb
                        self.target_tracker_id = tracker_id
                        print(f"✅ TARGET LOCKED: ID {tracker_id}")
                    return
            
            print("❌ No person clicked.")

    def process_target_selection(self):
        """Phase 1: Select Target"""
        cap = cv2.VideoCapture(self.cam1_source)
        window_name = "Camera 1: Select Target"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n--- PHASE 1: TARGET SELECTION ---")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue

            # Detect & Track
            results = self.detection_model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            # Store for callback
            self.current_frame = frame.copy()
            # Cache detection data in a simple list for the mouse callback
            self.current_detections = []
            for xyxy, conf, cls, tid in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
                self.current_detections.append((*xyxy, conf, cls, tid))

            # Visuals
            annotated = frame.copy()
            for (x1, y1, x2, y2, _, _, tid) in self.current_detections:
                color = self.colors['highlight'] if tid == self.target_tracker_id else self.colors['no_match']
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, f"ID: {tid}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # UI Text
            msg = "Click Person to Select. Press 'C' to Confirm." if self.target_tracker_id is None \
                  else f"TARGET ID {self.target_tracker_id} SELECTED. Press 'C' to Search."
            cv2.putText(annotated, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, annotated)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): return False
            if key == ord('c') and self.target_tracker_id is not None:
                cap.release()
                cv2.destroyWindow(window_name)
                return True

        return False

    def process_search(self):
        """Phase 2: Search"""
        cap = cv2.VideoCapture(self.cam2_source)
        window_name = "Camera 2: Search Mode"
        
        print("\n--- PHASE 2: SEARCHING ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Detect & Track
            results = self.detection_model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            for box, tracker_id in zip(detections.xyxy, detections.tracker_id):
                x1, y1, x2, y2 = map(int, box)
                
                # Get embedding for current person
                curr_emb = self.get_embedding(frame, box)
                
                # Compare
                sim = self.calculate_similarity(self.target_embedding, curr_emb)
                
                # Visualize
                is_match = sim > self.similarity_threshold
                color = self.colors['match'] if is_match else self.colors['no_match']
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if is_match else 2)
                
                # Draw Label
                label = f"MATCH {sim:.2f}" if is_match else f"ID {tracker_id}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.process_target_selection():
            self.process_search()
        else:
            print("System exited without selection.")

if __name__ == "__main__":
    app = PersonReIDSystem()
    app.run()