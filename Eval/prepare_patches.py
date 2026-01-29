import os
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import torch

# CONFIGURATION
SOURCE_DATASET = "./duke/dukemtmc-reid/DukeMTMC-reID"
OUTPUT_UPPER = "./duke_patches/upperbody"
OUTPUT_LOWER = "./duke_patches/lowerbody"

def create_crops():
    # 1. Setup Models
    print("Loading YOLOv8-Pose...")
    device = 0 if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO('yolov8n-pose.pt')
    
    # 2. Setup Directories
    for subset in ['bounding_box_train', 'bounding_box_test', 'query']:
        os.makedirs(f"{OUTPUT_UPPER}/{subset}", exist_ok=True)
        os.makedirs(f"{OUTPUT_LOWER}/{subset}", exist_ok=True)

    # 3. Process all images
    image_paths = glob.glob(f"{SOURCE_DATASET}/**/*.jpg", recursive=True)
    print(f"Found {len(image_paths)} images to process...")
    
    for img_path in tqdm(image_paths):
        # Determine subset
        path_parts = Path(img_path).parts
        if 'bounding_box_train' in path_parts: subset = 'bounding_box_train'
        elif 'bounding_box_test' in path_parts: subset = 'bounding_box_test'
        elif 'query' in path_parts: subset = 'query'
        else: continue 
            
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Inference
        results = pose_model(img, verbose=False, device=device)
        if not results: continue
            
        res = results[0]
        if res.keypoints is None or res.keypoints.xy.shape[0] == 0: continue
        
        kps = res.keypoints.xy[0].cpu().numpy()
        conf = res.keypoints.conf[0].cpu().numpy()
        
        # Skip if low overall confidence
        if np.mean(conf) < 0.4: continue 
        
        # --- DEFINE BODY PARTS INDICES (COCO Format) ---
        # 5,6: Shoulders | 11,12: Hips | 15,16: Ankles
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_ANKLE, R_ANKLE = 15, 16

        h_img, w_img = img.shape[:2]

        # =========================================================
        # 1. UPPER BODY CROP (Shoulders to Hips)
        # =========================================================
        # We need at least one shoulder and one hip to define the box
        has_shoulder = (kps[L_SHOULDER][0] > 0 or kps[R_SHOULDER][0] > 0)
        has_hip = (kps[L_HIP][0] > 0 or kps[R_HIP][0] > 0)

        if has_shoulder and has_hip:
            # Gather valid points
            ys_shoulder = [p[1] for p in [kps[L_SHOULDER], kps[R_SHOULDER]] if p[0]>0]
            ys_hip = [p[1] for p in [kps[L_HIP], kps[R_HIP]] if p[0]>0]
            
            # Y-Coords: Top is highest shoulder, Bottom is lowest hip
            y1 = min(ys_shoulder)
            y2 = max(ys_hip)
            
            # X-Coords: Center of torso based on all valid points
            valid_x = [p[0] for p in [kps[L_SHOULDER], kps[R_SHOULDER], kps[L_HIP], kps[R_HIP]] if p[0]>0]
            if len(valid_x) > 0:
                center_x = np.mean(valid_x)
                # Width: either the spread of points OR a fallback width
                spread = max(valid_x) - min(valid_x)
                width = max(spread * 1.5, w_img * 0.4) # Ensure it's not too skinny
                
                # Expand Top to include Head/Neck (go up 20% of torso height)
                torso_h = y2 - y1
                y1_final = max(0, y1 - (torso_h * 0.3)) 
                y2_final = min(h_img, y2 + (torso_h * 0.1)) # Slightly below hips
                
                x1_final = int(max(0, center_x - width/2))
                x2_final = int(min(w_img, center_x + width/2))
                
                upper_crop = img[int(y1_final):int(y2_final), x1_final:x2_final]
                
                # Save if valid
                if upper_crop.size > 0 and upper_crop.shape[0] > 20 and upper_crop.shape[1] > 20:
                    # Optional: Resize to 128x128 for training stability
                    upper_crop = cv2.resize(upper_crop, (128, 128))
                    cv2.imwrite(f"{OUTPUT_UPPER}/{subset}/{filename}", upper_crop)

        # =========================================================
        # 2. LOWER BODY CROP (Hips to Ankles)
        # =========================================================
        # We need hips and ankles (or at least knees if ankles missing, but let's stick to ankles for robustness)
        has_ankle = (kps[L_ANKLE][0] > 0 or kps[R_ANKLE][0] > 0)
        
        if has_hip and has_ankle:
            ys_hip = [p[1] for p in [kps[L_HIP], kps[R_HIP]] if p[0]>0]
            ys_ankle = [p[1] for p in [kps[L_ANKLE], kps[R_ANKLE]] if p[0]>0]
            
            y1 = min(ys_hip)
            y2 = max(ys_ankle)
            
            valid_x = [p[0] for p in [kps[L_HIP], kps[R_HIP], kps[L_ANKLE], kps[R_ANKLE]] if p[0]>0]
            
            if len(valid_x) > 0:
                center_x = np.mean(valid_x)
                spread = max(valid_x) - min(valid_x)
                width = max(spread * 1.6, w_img * 0.45) # Legs need more width for walking stride
                
                y1_final = max(0, y1)
                y2_final = min(h_img, y2 + (y2-y1)*0.1) # Include shoes
                
                x1_final = int(max(0, center_x - width/2))
                x2_final = int(min(w_img, center_x + width/2))
                
                lower_crop = img[int(y1_final):int(y2_final), x1_final:x2_final]
                
                if lower_crop.size > 0 and lower_crop.shape[0] > 20 and lower_crop.shape[1] > 20:
                    lower_crop = cv2.resize(lower_crop, (128, 128))
                    cv2.imwrite(f"{OUTPUT_LOWER}/{subset}/{filename}", lower_crop)

if __name__ == "__main__":
    create_crops()