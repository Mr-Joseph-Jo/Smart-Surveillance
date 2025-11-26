import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import time
from typing import Dict, List, Tuple, Optional

class PersonReID:
    def __init__(self, device='cpu'):
        self.device = device
        self.feature_extractor = self._load_feature_extractor()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_feature_extractor(self):
        """
        Load a pre-trained model for feature extraction
        You can replace this with a proper ReID model like OSNet
        For now, using ResNet50 as a simple feature extractor
        """
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        return model
    
    def extract_features(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from person crop
        """
        if person_crop.size == 0:
            return np.zeros(2048)  # Return zero vector for empty crops
        
        # Preprocess
        if len(person_crop.shape) == 3:
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        else:
            person_crop_rgb = person_crop
        
        input_tensor = self.transform(person_crop_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = features.view(features.size(0), -1)  # Flatten
            features = features.cpu().numpy().flatten()
        
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        """
        return cosine_similarity([features1], [features2])[0][0]

class GlobalTracker:
    def __init__(self, reid_model: PersonReID, similarity_threshold: float = 0.7):
        self.reid_model = reid_model
        self.similarity_threshold = similarity_threshold
        
        # Global tracking state
        self.global_id_counter = 0
        self.person_database = {}  # global_id: PersonProfile
        self.active_tracks = {1: {}, 2: {}}  # camera_id: {local_id: global_id}
        
        # For temporal consistency
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.last_seen = {}  # global_id: timestamp
        
    def update_tracks(self, camera_id: int, detections: List[Dict], frame: np.ndarray) -> Dict[int, int]:
        """
        Update tracks for a camera and return local_id -> global_id mapping
        
        detections: List of dicts with keys: 'local_id', 'bbox', 'confidence'
        """
        current_time = time.time()
        local_to_global_mapping = {}
        
        for detection in detections:
            local_id = detection['local_id']
            bbox = detection['bbox']  # (x1, y1, x2, y2)
            
            # Extract person crop
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Extract features
            features = self.reid_model.extract_features(person_crop)
            
            # Check if this local track already has a global ID
            if local_id in self.active_tracks[camera_id]:
                global_id = self.active_tracks[camera_id][local_id]
                # Update features in database
                self.person_database[global_id]['features'].append(features)
                self.person_database[global_id]['last_camera'] = camera_id
                self.last_seen[global_id] = current_time
                local_to_global_mapping[local_id] = global_id
            else:
                # Try to match with existing global IDs
                best_global_id, best_similarity = self._find_best_match(features, camera_id)
                
                if best_similarity > self.similarity_threshold:
                    # Match found
                    global_id = best_global_id
                    self.active_tracks[camera_id][local_id] = global_id
                    self.person_database[global_id]['features'].append(features)
                    self.person_database[global_id]['last_camera'] = camera_id
                    self.last_seen[global_id] = current_time
                else:
                    # Create new global ID
                    global_id = self._create_new_global_id()
                    self.active_tracks[camera_id][local_id] = global_id
                    self.person_database[global_id] = {
                        'features': deque([features], maxlen=20),
                        'first_seen': current_time,
                        'last_camera': camera_id,
                        'appearance_count': 1
                    }
                    self.last_seen[global_id] = current_time
                
                local_to_global_mapping[local_id] = global_id
        
        # Clean up inactive tracks
        self._cleanup_inactive_tracks(current_time, timeout=30.0)  # 30 seconds timeout
        
        return local_to_global_mapping
    
    def _find_best_match(self, query_features: np.ndarray, camera_id: int) -> Tuple[Optional[int], float]:
        """
        Find the best matching global ID for the query features
        """
        best_similarity = 0
        best_global_id = None
        
        for global_id, profile in self.person_database.items():
            # Skip if the person was just seen in the same camera (avoid duplicates)
            if profile['last_camera'] == camera_id:
                time_diff = time.time() - self.last_seen.get(global_id, 0)
                if time_diff < 2.0:  # 2 seconds threshold
                    continue
            
            # Compute similarity with stored features
            similarities = []
            for stored_features in profile['features']:
                sim = self.reid_model.compute_similarity(query_features, stored_features)
                similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_global_id = global_id
        
        return best_global_id, best_similarity
    
    def _create_new_global_id(self) -> int:
        """
        Create a new global ID
        """
        self.global_id_counter += 1
        return self.global_id_counter
    
    def _cleanup_inactive_tracks(self, current_time: float, timeout: float = 30.0):
        """
        Remove tracks that haven't been seen for a while
        """
        # Remove from active tracks
        for camera_id in self.active_tracks:
            to_remove = []
            for local_id, global_id in self.active_tracks[camera_id].items():
                if current_time - self.last_seen.get(global_id, current_time) > timeout:
                    to_remove.append(local_id)
            
            for local_id in to_remove:
                del self.active_tracks[camera_id][local_id]
        
        # Remove from global database if not seen for a long time
        to_remove_global = []
        for global_id, last_time in self.last_seen.items():
            if current_time - last_time > timeout * 2:  # Longer timeout for global database
                to_remove_global.append(global_id)
        
        for global_id in to_remove_global:
            if global_id in self.person_database:
                del self.person_database[global_id]
            if global_id in self.last_seen:
                del self.last_seen[global_id]
    
    def get_statistics(self) -> Dict:
        """
        Get tracking statistics
        """
        return {
            'total_global_ids': len(self.person_database),
            'active_tracks_cam1': len(self.active_tracks[1]),
            'active_tracks_cam2': len(self.active_tracks[2]),
            'global_id_counter': self.global_id_counter
        }