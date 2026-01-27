import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import supervision as sv
from torchreid import models
import time
from datetime import datetime

class PersonReIDSystem:
    def __init__(self):
        # Initialize models
        self.detection_model = YOLO('yolov8n.pt')
        self.feature_extractor = self._setup_reid_model()
        self.tracker = sv.ByteTrack()
        
        # Target management
        self.target_embedding = None
        self.target_tracker_id = None
        self.target_camera = "Camera 1"
        self.search_camera = "Camera 2"
        
        # Similarity threshold
        self.similarity_threshold = 0.70
        
        # Video sources
        self.cam1_source = "samplesurv_one.mp4"
        self.cam2_source = "samplesurv.mp4"

        # Visualization settings
        self.colors = {
            'match': (0, 255, 0),      # Green for matches
            'no_match': (0, 0, 255),   # Red for non-matches
            'target': (255, 255, 0),   # Yellow for target selection
            'highlight': (255, 0, 255) # Magenta for highlighted ID
        }
        
        # Store latest frame and detections for mouse callback
        self.current_frame = None
        self.current_detections = None

    def _setup_reid_model(self):
        """Initialize and setup the ReID model"""
        model = models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        model.eval()
        
        # Ensure model is in float32
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).float()
        
        return model

    def extract_features(self, image, bbox):
        """
        Extract feature embeddings from a person crop
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract person crop with boundary checks
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        person_crop = image[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
            
        # Preprocess for ReID model
        person_crop = self._preprocess_image(person_crop)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(person_crop)
            
        # Normalize features to unit length
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()

    def _preprocess_image(self, image):
        """
        Preprocess image for ReID model
        """
        # Resize to standard ReID input size
        image = cv2.resize(image, (128, 256))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize with float32 precision
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # Ensure tensor is float32 and on correct device
        device = next(self.feature_extractor.parameters()).device
        image = image.to(device).float()
        
        return image

    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        emb1_flat = embedding1.flatten()
        emb2_flat = embedding2.flatten()
        
        similarity = 1 - cosine(emb1_flat, emb2_flat)
        return similarity

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for selecting target person in Camera 1
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = self.current_frame
            detections = self.current_detections
            
            if frame is None or detections is None:
                return

            print(f"Mouse clicked at position: ({x}, {y})")
            
            # Find which detection contains the click point
            for i, (x1, y1, x2, y2, confidence, class_id, tracker_id) in enumerate(detections):
                if x1 <= x <= x2 and y1 <= y <= y2 and class_id == 0:  # class_id 0 is 'person'
                    # Extract features from selected person
                    bbox = [x1, y1, x2, y2]
                    target_embedding = self.extract_features(frame, bbox)

                    if target_embedding is not None:
                        self.target_embedding = target_embedding
                        self.target_tracker_id = tracker_id
                        print(f"🎯 Target person selected! ID: {tracker_id}, Embedding shape: {target_embedding.shape}")
                        return
            
            print("No person detected at clicked position.")

    def process_target_camera(self, video_source):
        """
        Process target camera (Camera 1) for person selection via mouse click
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return False

        print("🎥 Playing Camera 1 - Click on a person to select target")
        print("Press 'q' to quit")
        
        cv2.namedWindow(self.target_camera, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.target_camera, self.mouse_callback)
        
        target_selected = False
        
        while not target_selected:
            ret, frame = cap.read()
            if not ret:
                # Reset video if we reach the end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run object detection
            results = self.detection_model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter only person detections (class_id = 0)
            person_detections = detections[detections.class_id == 0]
            
            # Run tracking
            tracked_detections = self.tracker.update_with_detections(person_detections)
            
            # Store current frame and detections for mouse callback
            self.current_frame = frame.copy()
            self.current_detections = [
                (int(x1), int(y1), int(x2), int(y2), float(conf), int(cls), int(tid))
                for (x1, y1, x2, y2), conf, cls, tid in zip(
                    tracked_detections.xyxy, tracked_detections.confidence,
                    tracked_detections.class_id, tracked_detections.tracker_id)
            ]
            
            # Draw detections
            annotated_frame = frame.copy()
            for bbox, confidence, class_id, tracker_id in zip(
                tracked_detections.xyxy, tracked_detections.confidence, 
                tracked_detections.class_id, tracked_detections.tracker_id
            ):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Highlight if this is the selected target
                if tracker_id == self.target_tracker_id:
                    color = self.colors['highlight']
                    thickness = 4
                else:
                    color = self.colors['no_match']
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label = f"ID: {tracker_id}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add instructions and status
            cv2.putText(annotated_frame, "Click on a person to select target", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if self.target_tracker_id is not None:
                cv2.putText(annotated_frame, f"Target Selected: ID {self.target_tracker_id}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['highlight'], 2)
                cv2.putText(annotated_frame, "Press 'c' to continue to Camera 2", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(self.target_camera, annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.target_tracker_id is not None:
                target_selected = True
                break

        cap.release()
        cv2.destroyWindow(self.target_camera)
        return target_selected

    def process_search_camera(self, video_source):
        """
        Process search camera (Camera 2) for person re-identification
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return

        print(f"🔍 Searching for target person (ID: {self.target_tracker_id}) in Camera 2...")
        print("Press 'q' to quit")
        
        cv2.namedWindow(self.search_camera, cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection and tracking
            results = self.detection_model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            person_detections = detections[detections.class_id == 0]
            tracked_detections = self.tracker.update_with_detections(person_detections)
            
            annotated_frame = self._process_search_frame(frame, tracked_detections)
            cv2.imshow(self.search_camera, annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()

    def _process_search_frame(self, frame, detections):
        """Process search camera frame for person re-identification"""
        annotated_frame = frame.copy()
        match_found = False
        best_similarity = 0
        best_tracker_id = None
        
        for bbox, confidence, class_id, tracker_id in zip(
            detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
        ):
            x1, y1, x2, y2 = bbox.astype(int)
            
            if self.target_embedding is not None:
                # Extract features from current detection
                current_embedding = self.extract_features(frame, [x1, y1, x2, y2])
                
                if current_embedding is not None:
                    # Calculate similarity
                    similarity = self.calculate_similarity(self.target_embedding, current_embedding)
                    
                    # Update best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_tracker_id = tracker_id
                    
                    # Check if it's a match
                    if similarity > self.similarity_threshold:
                        # Draw green bounding box for match
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    self.colors['match'], 4)
                        
                        # Add match information
                        current_time = datetime.now().strftime("%H:%M:%S")
                        label = f"MATCH! ID:{tracker_id} Sim:{similarity:.3f}"
                        cv2.putText(annotated_frame, label, (x1, y1-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['match'], 2)
                        
                        match_found = True
                    else:
                        # Draw red bounding box for non-match
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    self.colors['no_match'], 2)
                        label = f"ID: {tracker_id} | Sim: {similarity:.3f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['no_match'], 2)
                else:
                    # Could not extract features
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                self.colors['no_match'], 2)
                    label = f"ID: {tracker_id}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['no_match'], 2)
            else:
                # No target selected
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                            self.colors['no_match'], 2)
                label = f"ID: {tracker_id}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['no_match'], 2)
        
        # Add status information
        status_color = self.colors['match'] if match_found else (255, 255, 255)
        
        if match_found:
            status = f"TARGET FOUND! ID: {best_tracker_id} | Similarity: {best_similarity:.3f}"
            print(f"🎯 Match found! ID: {best_tracker_id}, Similarity: {best_similarity:.3f}")
        else:
            status = f"Searching... Best Sim: {best_similarity:.3f}"
        
        cv2.putText(annotated_frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(annotated_frame, f"Target ID: {self.target_tracker_id}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

    def run(self):
        """Main function to run the ReID system"""
        print("🚀 Initializing Multi-Camera Person Re-Identification System...")
        print("="*60)
        
        # Step 1: Select target person from Camera 1 using mouse
        print("Step 1: Select target person in Camera 1")
        target_selected = self.process_target_camera(self.cam1_source)
        
        if not target_selected or self.target_embedding is None:
            print("❌ No target selected. Exiting.")
            cv2.destroyAllWindows()
            return
        
        # Step 2: Search for target in Camera 2
        print("Step 2: Searching for target in Camera 2")
        self.process_search_camera(self.cam2_source)
        
        cv2.destroyAllWindows()
        print("✅ System shutdown completed.")

if __name__ == "__main__":
    reid_system = PersonReIDSystem()
    reid_system.run()
