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
OUTPUT_FACE = "./duke_patches/face"
OUTPUT_HAIR = "./duke_patches/hair"

def create_crops():
    # 1. Setup Models
    print("Loading YOLOv8-Pose...")
    device = 0 if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO('yolov8n-pose.pt')
    
    # 2. Setup Directories
    for subset in ['bounding_box_train', 'bounding_box_test', 'query']:
        os.makedirs(f"{OUTPUT_FACE}/{subset}", exist_ok=True)
        os.makedirs(f"{OUTPUT_HAIR}/{subset}", exist_ok=True)

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
        
        # Skip if low confidence
        if np.mean(conf) < 0.35: continue 
        
        # --- ROBUST ANCHOR LOGIC (Matches your System) ---
        nose = kps[0]
        left_eye = kps[1]
        right_eye = kps[2]
        
        # 1. Find Center X (Anchor)
        if nose[0] > 0:
            center_x = nose[0]
            anchor_y = nose[1]
        elif left_eye[0] > 0 and right_eye[0] > 0:
            center_x = (left_eye[0] + right_eye[0]) / 2
            anchor_y = (left_eye[1] + right_eye[1]) / 2
        else:
            continue # Can't find head center

        # 2. Estimate Head Width
        if left_eye[0] > 0 and right_eye[0] > 0:
            head_width = np.linalg.norm(left_eye - right_eye) * 2.5
        else:
            head_width = img.shape[1] * 0.35 # Fallback: 35% of image width

        half_w = int(head_width / 2)
        img_h, img_w = img.shape[:2]

        # --- FACE PATCH GENERATION ---
        # Face is roughly centered on the anchor (nose/eyes)
        # We take a square box based on head_width
        face_size = int(head_width * 1.5) # slightly larger than head width
        half_face = int(face_size / 2)
        
        fx1 = int(max(0, center_x - half_face))
        fx2 = int(min(img_w, center_x + half_face))
        fy1 = int(max(0, anchor_y - half_face))
        fy2 = int(min(img_h, anchor_y + half_face))
        
        face_crop = img[fy1:fy2, fx1:fx2]
        
        # Save Face if valid size
        if face_crop.size > 0 and face_crop.shape[0] > 15 and face_crop.shape[1] > 15:
            # OPTIONAL: Resize here to save disk space/training time, or let Dataloader do it
            # cv2.resize(face_crop, (128, 128)) 
            cv2.imwrite(f"{OUTPUT_FACE}/{subset}/{filename}", face_crop)

        # --- HAIR PATCH GENERATION ---
        # Hair is from the anchor UPWARDS.
        # X range: Same centered X as face
        # Y range: From well above the head down to the eyes
        
        hx1 = int(max(0, center_x - half_w))
        hx2 = int(min(img_w, center_x + half_w))
        
        # Hair "Bottom" is the eyes/nose
        hy2 = int(anchor_y) 
        # Hair "Top" is guessed based on head width (approx 1.5x width upwards)
        hy1 = int(max(0, hy2 - (head_width * 1.2)))
        
        hair_crop = img[hy1:hy2, hx1:hx2]
        
        # Save Hair if valid size
        if hair_crop.size > 0 and hair_crop.shape[0] > 10 and hair_crop.shape[1] > 10:
             cv2.imwrite(f"{OUTPUT_HAIR}/{subset}/{filename}", hair_crop)

if __name__ == "__main__":
    create_crops()