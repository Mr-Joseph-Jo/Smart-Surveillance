import os
import cv2
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
# Ensure this points to your ORIGINAL DukeMTMC-reID folder
SOURCE_DATASET = "./duke/dukemtmc-reid/DukeMTMC-reID"
# This is where the new "Clean" dataset will be saved
OUTPUT_DATASET = "./duke/dukemtmc-reid/DukeMTMC-Masked"

def create_masked_data():
    # 1. Setup Model
    print("Loading YOLOv8-Pose...")
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n-pose.pt')

    # 2. Process File Structure
    # We maintain the exact folder structure so training scripts work without changes
    subsets = ['bounding_box_train', 'bounding_box_test', 'query', 'gallery']
    
    # Get list of all images
    image_paths = glob.glob(f"{SOURCE_DATASET}/**/*.jpg", recursive=True)
    print(f"Found {len(image_paths)} images. Generating masked dataset...")

    for img_path in tqdm(image_paths):
        # Determine subset (train/test/query) to maintain structure
        path_obj = Path(img_path)
        parts = path_obj.parts
        
        subset = None
        for s in subsets:
            if s in parts:
                subset = s
                break
        
        # Skip files that aren't in the standard folders (e.g. system files)
        if subset is None: continue 

        # Create output directory
        save_dir = f"{OUTPUT_DATASET}/{subset}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        # 3. Pose Inference
        results = model(img, verbose=False, device=device)
        
        # Default: Use original image if pose fails (Safety fallback)
        final_img = img.copy() 
        
        # --- ROBUST CHECK (THE FIX) ---
        # 1. Check if results exist
        # 2. Check if keypoints container exists
        # 3. Check if at least one person was detected (shape[0] > 0)
        if (results and 
            results[0].keypoints is not None and 
            results[0].keypoints.xy.shape[0] > 0):

            # Safe to access the first person
            kps = results[0].keypoints.xy[0].cpu().numpy()
            conf = results[0].keypoints.conf[0].cpu().numpy()
            
            # Filter valid points (> 0.4 confidence)
            # We ignore low-confidence points to avoid drawing masks on noise
            valid_kps = kps[conf > 0.4]
            
            if len(valid_kps) > 4: # Need at least ~4 points to make a meaningful shape
                # 4. Create Convex Hull (The "Shrink Wrap")
                valid_kps = valid_kps.astype(np.int32)
                hull = cv2.convexHull(valid_kps)
                
                # 5. Create Mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                
                # 6. Dilation (The "Safety Padding")
                # We expand the mask by ~10% of image width to include backpacks/loose clothes
                kernel_size = int(w * 0.10) 
                if kernel_size > 0: # Handle edge case for tiny images
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                
                # 7. Apply Background Suppression
                # Convert mask to 3-channel
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Bitwise AND: Keep pixels where mask is white, black out others
                final_img = cv2.bitwise_and(img, mask_3ch)

        # 8. Save
        filename = os.path.basename(img_path)
        cv2.imwrite(f"{save_dir}/{filename}", final_img)

if __name__ == "__main__":
    create_masked_data()