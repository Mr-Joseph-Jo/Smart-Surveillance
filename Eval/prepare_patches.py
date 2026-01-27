import os
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import torch

# CONFIGURATION
# Ensure this points to where your DukeMTMC-reID folder actually sits
SOURCE_DATASET = "./duke/dukemtmc-reid/DukeMTMC-reID"  
OUTPUT_FACE = "./duke_patches/face"                   
OUTPUT_HAIR = "./duke_patches/hair"                   

def create_crops():
    # 1. Setup Models
    print("Loading YOLOv8-Pose...")
    # Check for GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO('yolov8n-pose.pt')
    
    # 2. Setup Directories (Mimic ReID structure)
    for subset in ['bounding_box_train', 'bounding_box_test', 'query']:
        os.makedirs(f"{OUTPUT_FACE}/{subset}", exist_ok=True)
        os.makedirs(f"{OUTPUT_HAIR}/{subset}", exist_ok=True)

    # 3. Process all images
    image_paths = glob.glob(f"{SOURCE_DATASET}/**/*.jpg", recursive=True)
    
    print(f"Found {len(image_paths)} images to process...")
    
    for img_path in tqdm(image_paths):
        # Determine subset (train/test/query)
        path_parts = Path(img_path).parts
        if 'bounding_box_train' in path_parts: subset = 'bounding_box_train'
        elif 'bounding_box_test' in path_parts: subset = 'bounding_box_test'
        elif 'query' in path_parts: subset = 'query'
        else: continue # Skip other files (like gallery or distractor sets if present)
            
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Run Inference
        results = pose_model(img, verbose=False, device=device)
        
        # --- BUG FIX START ---
        # Check if results exist AND if keypoints were actually detected
        if not results: 
            continue
            
        res = results[0]
        # Check if keypoints container exists and has at least one detection
        if res.keypoints is None or res.keypoints.xy.shape[0] == 0:
            continue
        # --- BUG FIX END ---
        
        kps = res.keypoints.xy[0].cpu().numpy()
        conf = res.keypoints.conf[0].cpu().numpy()
        
        # Need nose (0) and eyes (1,2) for valid face/hair
        # If overall confidence of keypoints is too low, skip
        if np.mean(conf) < 0.4: continue 
        
        # --- CROP LOGIC ---
        h, w = img.shape[:2]
        nose = kps[0]
        
        # Check if nose was actually detected (confidence > 0 or x,y > 0)
        if nose[0] == 0 and nose[1] == 0: continue

        # FACE CROP (Nose + Eyes region)
        # Simple heuristic: Box around facial keypoints (0..4 are nose, eyes, ears)
        face_pts = kps[:5] 
        valid_face = face_pts[face_pts[:,0] > 0]
        
        if len(valid_face) >= 3:
            x1, y1 = np.min(valid_face, axis=0)
            x2, y2 = np.max(valid_face, axis=0)
            
            # Add padding to get the whole head/face context
            pad_x = int((x2-x1) * 0.5)
            pad_y = int((y2-y1) * 0.5)
            
            y_min = max(0, int(y1-pad_y))
            y_max = min(h, int(y2+pad_y))
            x_min = max(0, int(x1-pad_x))
            x_max = min(w, int(x2+pad_x))
            
            face_crop = img[y_min:y_max, x_min:x_max]
            
            # Only save if crop is valid and not tiny
            if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                cv2.imwrite(f"{OUTPUT_FACE}/{subset}/{filename}", face_crop)

        # HAIR CROP (Top of head to nose)
        # Heuristic: From top of image (0) down to nose (y)
        # Note: In ReID bbox, top of image is usually top of head anyway
        hair_bottom = int(nose[1])
        if hair_bottom > 5: # Ensure we have at least 5 pixels of height
            hair_crop = img[0:hair_bottom, :]
            
            if hair_crop.size > 0:
                cv2.imwrite(f"{OUTPUT_HAIR}/{subset}/{filename}", hair_crop)

if __name__ == "__main__":
    create_crops()