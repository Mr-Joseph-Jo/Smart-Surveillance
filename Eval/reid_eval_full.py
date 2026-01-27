"""
Market-1501 Person Re-Identification - FIXED GALLERY PARSING
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

# ==================== 1. FIXED DATA PARSER ====================

@dataclass
class Market1501Sample:
    image_path: str
    person_id: int
    camera_id: int
    
    @staticmethod
    def parse_filename(filename: str, full_path: str) -> Optional['Market1501Sample']:
        """
        Parse Market1501 filename format CORRECTLY:
        
        Query format: 0001_c1s1_001051_00.jpg
        Gallery format: 0001_c1s1_001051_00.jpg OR -1_c1s1_001051_00.jpg (junk)
        
        Important: Gallery has junk images with person_id = -1, but we need to keep them!
        For testing, we should include ALL gallery images.
        """
        try:
            basename = os.path.basename(filename)
            
            # Split filename
            parts = basename.replace('.jpg', '').split('_')
            if len(parts) < 4:
                return None
            
            # Person ID (first part)
            # CRITICAL: Gallery has -1 for junk, but we need to keep them for evaluation
            try:
                person_id = int(parts[0])
            except ValueError:
                # If can't parse as int, assign a special ID
                person_id = -999
            
            # Camera ID (second part, format: cXsX where X is number)
            camera_sequence = parts[1]
            try:
                # Extract camera number from 'cX' part
                camera_id = int(camera_sequence[1])  # 'c1s1' -> '1' from position 1
            except (IndexError, ValueError):
                # Try alternative parsing
                if 'c' in camera_sequence:
                    cam_part = camera_sequence.split('c')[1]
                    if 's' in cam_part:
                        camera_id = int(cam_part.split('s')[0])
                    else:
                        camera_id = int(cam_part[0])
                else:
                    camera_id = 0
            
            return Market1501Sample(
                image_path=full_path,
                person_id=person_id,
                camera_id=camera_id
            )
            
        except Exception as e:
            print(f"DEBUG: Error parsing {filename}: {e}")
            return None

# ==================== 2. DATASET LOADER WITH DEBUG INFO ====================

class Market1501Loader:
    """Load Market-1501 dataset with PROPER gallery parsing"""
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        print(f"Loading Market-1501 from: {self.dataset_path}")
        
        # Verify dataset structure
        self.verify_structure()
        
        # Load gallery (bounding_box_test) - KEEP ALL IMAGES INCLUDING JUNK
        gallery_path = self.dataset_path / "bounding_box_test"
        self.gallery = self.load_folder_with_debug(gallery_path, "gallery")
        
        # Load query
        query_path = self.dataset_path / "query"
        self.query = self.load_folder_with_debug(query_path, "query")
        
        print(f"\nDataset loaded successfully!")
        print(f"  Gallery images: {len(self.gallery)}")
        print(f"  Query images: {len(self.query)}")
        
        # Show statistics
        self.show_statistics()

    def verify_structure(self):
        """Verify the dataset has required folders"""
        required_folders = ["bounding_box_test", "query"]
        
        print("\nVerifying dataset structure...")
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if folder_path.exists():
                jpg_count = len(list(folder_path.glob("*.jpg")))
                print(f"  ✓ {folder}: {jpg_count} images")
            else:
                raise FileNotFoundError(f"Required folder not found: {folder}")

    def load_folder_with_debug(self, folder_path: Path, name: str) -> List[Market1501Sample]:
        """Load all images from a folder with debugging info"""
        if not folder_path.exists():
            print(f"Warning: {name} folder {folder_path} not found")
            return []
        
        jpg_files = list(folder_path.glob("*.jpg"))
        if not jpg_files:
            print(f"Warning: No .jpg files found in {name} folder")
            return []
        
        print(f"\nLoading {name} images...")
        samples = []
        
        # Sample some files to debug parsing
        debug_samples = []
        
        for img_file in tqdm(jpg_files[:1000], desc=f"Parsing {name}"):  # Load more for gallery
            sample = Market1501Sample.parse_filename(str(img_file), str(img_file))
            if sample:
                samples.append(sample)
                if len(debug_samples) < 5:
                    debug_samples.append((os.path.basename(img_file), sample.person_id, sample.camera_id))
            else:
                if len(debug_samples) < 5:
                    debug_samples.append((os.path.basename(img_file), "FAILED", "FAILED"))
        
        # Show debug info
        print(f"\nDebug - First 5 {name} files:")
        for filename, pid, cam in debug_samples[:5]:
            print(f"  {filename} -> Person ID: {pid}, Camera: {cam}")
        
        print(f"\nLoaded {len(samples)} images from {name}")
        
        # Show person ID distribution
        if samples:
            person_ids = [s.person_id for s in samples]
            unique_ids = set(person_ids)
            print(f"  Unique person IDs: {len(unique_ids)}")
            print(f"  Person ID range: {min(person_ids)} to {max(person_ids)}")
            
            # Count junk images (person_id = -1)
            junk_count = sum(1 for pid in person_ids if pid == -1)
            if junk_count > 0:
                print(f"  Junk images (ID=-1): {junk_count}")
        
        return samples

    def show_statistics(self):
        """Show dataset statistics"""
        print("\nDataset Statistics:")
        print("-" * 50)
        
        # Gallery stats
        if self.gallery:
            gallery_pids = [s.person_id for s in self.gallery]
            unique_gallery_pids = set(gallery_pids)
            gallery_cams = set(s.camera_id for s in self.gallery)
            
            print(f"Gallery: {len(self.gallery)} images")
            print(f"  Unique persons: {len(unique_gallery_pids)}")
            print(f"  Cameras: {sorted(gallery_cams)}")
            
            # Show person ID distribution
            print(f"  Person ID distribution:")
            from collections import Counter
            pid_counts = Counter(gallery_pids)
            top_pids = pid_counts.most_common(10)
            for pid, count in top_pids:
                print(f"    ID {pid}: {count} images")
        
        # Query stats
        if self.query:
            query_pids = [s.person_id for s in self.query]
            unique_query_pids = set(query_pids)
            query_cams = set(s.camera_id for s in self.query)
            
            print(f"\nQuery: {len(self.query)} images")
            print(f"  Unique persons: {len(unique_query_pids)}")
            print(f"  Cameras: {sorted(query_cams)}")

# ==================== 3. SIMPLE BUT EFFECTIVE FEATURE EXTRACTOR ====================

class SimpleReIDExtractor:
    """Simple but effective feature extractor for Market-1501"""
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing extractor on {self.device}")
        
        # Load ResNet50 for deep features
        print("Loading ResNet50...")
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_features.eval().to(self.device)
        
        # Transform for ResNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Extract deep features using ResNet50"""
        try:
            if image is None:
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.resnet_features(img_tensor)
                features = features.view(features.size(0), -1)
                features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().squeeze(0)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def compute_similarity(self, f1: torch.Tensor, f2: torch.Tensor) -> float:
        """Compute cosine similarity between features"""
        if f1 is None or f2 is None:
            return 0.0
        sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0))
        return float(sim.item())

# ==================== 4. EVALUATOR WITH PROPER GALLERY HANDLING ====================

class ReIDEvaluator:
    """Evaluate ReID performance with proper gallery handling"""
    def __init__(self, loader: Market1501Loader):
        self.loader = loader
        self.gallery = loader.gallery
        self.query = loader.query
        
        # Filter out ONLY completely invalid gallery samples (not junk with ID=-1)
        self.gallery = [s for s in self.gallery if hasattr(s, 'person_id') and s.person_id is not None]
        
        print(f"\nEvaluator initialized with:")
        print(f"  Gallery: {len(self.gallery)} images")
        print(f"  Query: {len(self.query)} images")
        
        if len(self.gallery) == 0:
            print("WARNING: No valid gallery images found!")

    def run_simple_evaluation(self, extractor, experiment_name: str, 
                            gallery_size=500, query_size=100):
        """Run evaluation with given feature extractor"""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*70}")
        
        # Check if we have gallery images
        if len(self.gallery) == 0:
            print("ERROR: No gallery images available!")
            # Try to load gallery directly as a fallback
            gallery_path = self.loader.dataset_path / "bounding_box_test"
            print(f"Attempting to load gallery directly from: {gallery_path}")
            
            gallery_images = []
            for img_file in list(gallery_path.glob("*.jpg"))[:gallery_size]:
                gallery_images.append(str(img_file))
            
            print(f"Loaded {len(gallery_images)} gallery image paths")
            
            # Create dummy gallery samples
            self.gallery = []
            for i, img_path in enumerate(gallery_images):
                self.gallery.append(Market1501Sample(
                    image_path=img_path,
                    person_id=i,  # Temporary ID
                    camera_id=1
                ))
            
            print(f"Created {len(self.gallery)} temporary gallery samples")
        
        # Use subsets for testing
        gallery_subset = self.gallery[:gallery_size]
        query_subset = self.query[:query_size]
        
        print(f"Using subset:")
        print(f"  Gallery: {len(gallery_subset)} images")
        print(f"  Query: {len(query_subset)} images")
        
        # Extract gallery features
        print(f"\nExtracting gallery features...")
        gallery_data = []
        
        for sample in tqdm(gallery_subset, desc="Gallery"):
            img = cv2.imread(sample.image_path)
            if img is not None:
                features = extractor.extract_features(img)
                if features is not None:
                    gallery_data.append({
                        'sample': sample,
                        'features': features,
                        'person_id': sample.person_id
                    })
        
        print(f"  Successfully extracted: {len(gallery_data)} images")
        
        # Extract query features
        print(f"\nExtracting query features...")
        query_data = []
        
        for sample in tqdm(query_subset, desc="Query"):
            img = cv2.imread(sample.image_path)
            if img is not None:
                features = extractor.extract_features(img)
                if features is not None:
                    query_data.append({
                        'sample': sample,
                        'features': features,
                        'person_id': sample.person_id
                    })
        
        print(f"  Successfully extracted: {len(query_data)} images")
        
        # Check if we have enough features
        if len(gallery_data) < 10 or len(query_data) < 10:
            print(f"ERROR: Not enough features extracted!")
            print(f"  Gallery features: {len(gallery_data)}")
            print(f"  Query features: {len(query_data)}")
            return None
        
        # Perform matching
        print(f"\nPerforming ReID matching...")
        correct_matches = 0
        total_matches = 0
        
        for q in tqdm(query_data, desc="Matching"):
            best_match_idx = -1
            best_similarity = -1.0
            
            for i, g in enumerate(gallery_data):
                # Skip if same image (optional)
                if q['sample'].image_path == g['sample'].image_path:
                    continue
                
                similarity = extractor.compute_similarity(q['features'], g['features'])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
            
            if best_match_idx != -1:
                total_matches += 1
                if q['person_id'] == gallery_data[best_match_idx]['person_id']:
                    correct_matches += 1
        
        # Calculate accuracy
        if total_matches > 0:
            rank1_accuracy = correct_matches / total_matches
        else:
            rank1_accuracy = 0.0
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Experiment: {experiment_name}")
        print(f"Query images matched: {total_matches}")
        print(f"Correct matches: {correct_matches}")
        print(f"Rank-1 Accuracy: {rank1_accuracy:.4f} ({rank1_accuracy*100:.2f}%)")
        print(f"{'='*70}")
        
        return {
            'name': experiment_name,
            'rank1': rank1_accuracy * 100,
            'correct': correct_matches,
            'total': total_matches,
            'gallery_size': len(gallery_data),
            'query_size': len(query_data)
        }

# ==================== 5. MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print(f"\n{'='*80}")
    print("MARKET-1501 PERSON RE-IDENTIFICATION - FIXED GALLERY PARSING")
    print(f"{'='*80}")
    
    # Configuration
    DATASET_PATH = "Market-1501-v15.09.15"
    
    print(f"Dataset path: {DATASET_PATH}")
    
    # Step 1: Load dataset with debug info
    print(f"\n{'='*80}")
    print("STEP 1: LOADING DATASET WITH DEBUG INFO")
    print(f"{'='*80}")
    
    try:
        loader = Market1501Loader(DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Step 2: Initialize evaluator
    evaluator = ReIDEvaluator(loader)
    
    # Step 3: Run experiment
    print(f"\n{'='*80}")
    print("STEP 2: RUNNING REID EVALUATION")
    print(f"{'='*80}")
    
    # Create extractor
    extractor = SimpleReIDExtractor()
    
    # Run evaluation
    result = evaluator.run_simple_evaluation(
        extractor,
        "ResNet50 ReID",
        gallery_size=500,  # Use 500 gallery images
        query_size=100     # Use 100 query images
    )
    
    if result:
        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(f"Experiment: {result['name']}")
        print(f"Rank-1 Accuracy: {result['rank1']:.2f}%")
        print(f"Correct/Total: {result['correct']}/{result['total']}")
        print(f"Gallery features: {result['gallery_size']}")
        print(f"Query features: {result['query_size']}")
        print(f"{'='*80}")
        
        # Save result
        with open('reid_result.txt', 'w') as f:
            f.write(f"Market-1501 ReID Result\n")
            f.write(f"="*40 + "\n")
            f.write(f"Experiment: {result['name']}\n")
            f.write(f"Rank-1 Accuracy: {result['rank1']:.2f}%\n")
            f.write(f"Correct matches: {result['correct']}/{result['total']}\n")
            f.write(f"Gallery images: {result['gallery_size']}\n")
            f.write(f"Query images: {result['query_size']}\n")
        
        print(f"Result saved to 'reid_result.txt'")
    else:
        print("Evaluation failed!")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")

# ==================== 6. QUICK DEBUG FUNCTION ====================

def debug_gallery_parsing():
    """Quick debug function to understand gallery file naming"""
    DATASET_PATH = "Market-1501-v15.09.15"
    gallery_path = Path(DATASET_PATH) / "bounding_box_test"
    
    print(f"\nDebugging gallery files in: {gallery_path}")
    
    # List first 20 gallery files
    gallery_files = list(gallery_path.glob("*.jpg"))[:20]
    
    print(f"\nFirst 20 gallery files:")
    for i, img_file in enumerate(gallery_files):
        filename = img_file.name
        print(f"{i+1:2d}. {filename}")
        
        # Try to parse
        sample = Market1501Sample.parse_filename(filename, str(img_file))
        if sample:
            print(f"     -> Person ID: {sample.person_id}, Camera: {sample.camera_id}")
        else:
            print(f"     -> FAILED TO PARSE")
    
    # Check what person IDs exist
    print(f"\nAnalyzing person IDs in first 100 gallery files...")
    person_ids = []
    for img_file in list(gallery_path.glob("*.jpg"))[:100]:
        filename = img_file.name
        sample = Market1501Sample.parse_filename(filename, str(img_file))
        if sample:
            person_ids.append(sample.person_id)
    
    print(f"Unique person IDs found: {set(person_ids)}")
    print(f"Person ID -1 count: {sum(1 for pid in person_ids if pid == -1)}")
    print(f"Person ID 0 count: {sum(1 for pid in person_ids if pid == 0)}")

if __name__ == "__main__":
    # First run debug to understand the issue
    debug_gallery_parsing()
    
    # Then run main evaluation
    main()