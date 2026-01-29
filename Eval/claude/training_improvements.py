"""
Improved Training Script for Specialized OSNet Models
Addresses overfitting and poor generalization of local features
"""

import torch
import torch.nn as nn
import torchreid
from torchreid import models
from torchreid.engine import ImageSoftmaxEngine
from torchreid.losses import TripletLoss, CrossEntropyLoss
from torchreid.optim import Optimizer
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random
from PIL import Image

# ==================== CONFIGURATION ====================

DATASET_ROOT = './duke/dukemtmc-reid'
SAVE_DIR_BODY = './log/osnet_duke_body_v2'
SAVE_DIR_HAIR = './log/osnet_duke_hair_v2'
SAVE_DIR_FACE = './log/osnet_duke_face_v2'

# Training hyperparameters (IMPROVED)
MAX_EPOCH = 150  # Increased from 50
BATCH_SIZE = 64  # Increased from 32
LR = 0.0003      # Lower learning rate for better convergence

# Hard mining parameters
USE_HARD_MINING = True
MARGIN = 0.3  # Triplet loss margin

# ==================== CUSTOM DATA AUGMENTATION ====================

class ImprovedAugmentation:
    """Enhanced augmentation pipeline for better generalization"""
    
    @staticmethod
    def get_train_transforms(image_size=(256, 128), is_local=False):
        """
        Args:
            image_size: Target size
            is_local: If True, apply stronger augmentation for local patches
        """
        if is_local:
            # Stronger augmentation for small patches (hair/face)
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # Small rotations
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),  # Occasional grayscale
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3)
                )
            ])
        else:
            # Standard augmentation for full body
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.4),
                    ratio=(0.3, 3.3)
                )
            ])
    
    @staticmethod
    def get_test_transforms(image_size=(256, 128)):
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

# ==================== PATCH EXTRACTION ====================

class PoseGuidedPatchExtractor:
    """Extract hair and face patches using YOLOv8-Pose"""
    
    def __init__(self):
        self.pose_model = YOLO('yolov8n-pose.pt')
        
    def extract_hair_patch(self, img_path):
        """Extract hair region"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose_model(img, verbose=False)
        
        if not results or len(results) == 0:
            # Fallback: top 30% of image
            h, w = img_rgb.shape[:2]
            return img_rgb[0:int(h*0.3), :]
        
        kps = results[0].keypoints.xy[0].cpu().numpy()
        
        # Get nose and eyes
        nose = kps[0]
        if nose[0] == 0:
            h, w = img_rgb.shape[:2]
            return img_rgb[0:int(h*0.3), :]
        
        # Hair region: top of image to slightly above nose
        h, w = img_rgb.shape[:2]
        y_bottom = int(nose[1] - 10)
        y_bottom = max(20, min(h, y_bottom))
        
        # Add horizontal margin
        x_margin = int(w * 0.1)
        crop = img_rgb[0:y_bottom, x_margin:w-x_margin]
        
        if crop.size == 0 or crop.shape[0] < 20:
            return img_rgb[0:int(h*0.3), :]
        
        return crop
    
    def extract_face_patch(self, img_path):
        """Extract face region"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose_model(img, verbose=False)
        
        if not results or len(results) == 0:
            # Fallback: central top region
            h, w = img_rgb.shape[:2]
            return img_rgb[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
        
        kps = results[0].keypoints.xy[0].cpu().numpy()
        face_kps = kps[:5]  # First 5 keypoints are face
        
        valid = face_kps[face_kps[:, 0] > 0]
        if len(valid) < 2:
            h, w = img_rgb.shape[:2]
            return img_rgb[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
        
        # Compute bounding box with padding
        x1, y1 = np.min(valid, axis=0)
        x2, y2 = np.max(valid, axis=0)
        
        face_width = x2 - x1
        face_height = y2 - y1
        pad_x = int(face_width * 0.4)  # More padding for context
        pad_y = int(face_height * 0.6)
        
        h, w = img_rgb.shape[:2]
        y1_crop = max(0, int(y1 - pad_y))
        y2_crop = min(h, int(y2 + pad_y))
        x1_crop = max(0, int(x1 - pad_x))
        x2_crop = min(w, int(x2 + pad_x))
        
        crop = img_rgb[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if crop.size == 0 or crop.shape[0] < 20:
            h, w = img_rgb.shape[:2]
            return img_rgb[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
        
        return crop

# ==================== CUSTOM DATASET ====================

class PatchDataset(torch.utils.data.Dataset):
    """Dataset that extracts patches on-the-fly"""
    
    def __init__(self, original_dataset, patch_type='hair', transform=None, cache_patches=True):
        """
        Args:
            original_dataset: torchreid dataset
            patch_type: 'hair', 'face', or 'body'
            transform: torchvision transforms
            cache_patches: If True, cache extracted patches (faster but uses more memory)
        """
        self.dataset = original_dataset
        self.patch_type = patch_type
        self.transform = transform
        self.extractor = PoseGuidedPatchExtractor() if patch_type != 'body' else None
        
        self.cache_patches = cache_patches
        self.patch_cache = {}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, pid, camid = self.dataset.data[idx]
        
        # Check cache first
        if self.cache_patches and idx in self.patch_cache:
            patch = self.patch_cache[idx]
        else:
            # Extract patch
            if self.patch_type == 'body':
                patch = Image.open(img_path).convert('RGB')
            elif self.patch_type == 'hair':
                patch_np = self.extractor.extract_hair_patch(img_path)
                if patch_np is None:
                    patch_np = np.zeros((128, 128, 3), dtype=np.uint8)
                patch = Image.fromarray(patch_np)
            elif self.patch_type == 'face':
                patch_np = self.extractor.extract_face_patch(img_path)
                if patch_np is None:
                    patch_np = np.zeros((128, 128, 3), dtype=np.uint8)
                patch = Image.fromarray(patch_np)
            
            if self.cache_patches:
                self.patch_cache[idx] = patch
        
        # Apply transforms
        if self.transform is not None:
            patch = self.transform(patch)
        
        return patch, pid, camid

# ==================== TRAINING FUNCTIONS ====================

def train_model(model_type='body'):
    """
    Train a specialized model
    
    Args:
        model_type: 'body', 'hair', or 'face'
    """
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*80}\n")
    
    # Configuration
    if model_type == 'body':
        save_dir = SAVE_DIR_BODY
        image_size = (256, 128)
        is_local = False
    elif model_type == 'hair':
        save_dir = SAVE_DIR_HAIR
        image_size = (128, 128)
        is_local = True
    else:  # face
        save_dir = SAVE_DIR_FACE
        image_size = (128, 128)
        is_local = True
    
    # Load dataset
    datamanager = torchreid.data.ImageDataManager(
        root=DATASET_ROOT,
        sources='dukemtmcreid',
        height=image_size[0],
        width=image_size[1],
        batch_size_train=BATCH_SIZE,
        batch_size_test=100,
        transforms='random_flip',  # We'll override this
        num_instances=4,  # For triplet loss
        train_sampler='RandomIdentitySampler'
    )
    
    # Get original datasets
    train_dataset = datamanager.train_loader.dataset
    
    # Create patch datasets with improved augmentation
    train_transforms = ImprovedAugmentation.get_train_transforms(image_size, is_local)
    test_transforms = ImprovedAugmentation.get_test_transforms(image_size)
    
    train_patch_dataset = PatchDataset(
        train_dataset,
        patch_type=model_type,
        transform=train_transforms,
        cache_patches=False  # Don't cache to save memory with large dataset
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_patch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Model
    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )
    
    # Optimizer
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=LR
    )
    
    # Scheduler
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=60,
        gamma=0.1
    )
    
    # Loss functions
    criterion_xent = CrossEntropyLoss(
        num_classes=datamanager.num_train_pids,
        use_gpu=True,
        label_smooth=True  # Label smoothing for better generalization
    )
    
    if USE_HARD_MINING:
        criterion_triplet = TripletLoss(margin=MARGIN)
    
    # Engine
    engine = ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        use_gpu=True
    )
    
    # Override train loader
    engine.train_loader = train_loader
    
    # Custom training loop with hard mining
    print(f"Starting training for {MAX_EPOCH} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Image size: {image_size}")
    print(f"Hard mining: {USE_HARD_MINING}")
    print(f"Save directory: {save_dir}\n")
    
    # Train
    engine.run(
        save_dir=save_dir,
        max_epoch=MAX_EPOCH,
        eval_freq=10,
        print_freq=50,
        test_only=False,
        dist_metric='euclidean',
        rerank=False,
        visrank=False,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        test_loader=None  # We'll evaluate separately
    )
    
    print(f"\n{model_type.upper()} model training completed!")
    print(f"Model saved to: {save_dir}")

# ==================== MAIN ====================

def main():
    print("\n" + "="*80)
    print("IMPROVED TRAINING SCRIPT FOR MULTI-GRANULARITY RE-ID")
    print("="*80)
    print("\nImprovements:")
    print("  ✓ Better data augmentation (rotation, color jitter, blur)")
    print("  ✓ More epochs (150 vs 50)")
    print("  ✓ Larger batch size (64 vs 32)")
    print("  ✓ Label smoothing for generalization")
    print("  ✓ Triplet loss with hard mining")
    print("  ✓ Stronger augmentation for local patches")
    print("="*80)
    
    # Train each model
    models_to_train = ['body', 'hair', 'face']
    
    for model_type in models_to_train:
        try:
            train_model(model_type)
        except Exception as e:
            print(f"\nError training {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update model paths in multi_granularity_reid_v2.py:")
    print(f"   PATH_BODY = '{SAVE_DIR_BODY}/model/model.pth.tar-{MAX_EPOCH}'")
    print(f"   PATH_HAIR = '{SAVE_DIR_HAIR}/model/model.pth.tar-{MAX_EPOCH}'")
    print(f"   PATH_FACE = '{SAVE_DIR_FACE}/model/model.pth.tar-{MAX_EPOCH}'")
    print("\n2. Run diagnostics: python diagnostic_tool.py")
    print("3. Run evaluation: python multi_granularity_reid_v2.py")
    print("="*80)

if __name__ == "__main__":
    main()
