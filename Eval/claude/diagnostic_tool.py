"""
Feature Quality Diagnostic Tool
Helps diagnose why local features (hair/face) are hurting performance
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import pandas as pd
from collections import defaultdict

# Import from main system
import sys
sys.path.append('/home/claude')
from multi_granularity_reid_v2 import (
    ReIDSample, MultiGranularitySystem, PoseEstimator,
    BodyExtractor, FaceExtractor, HairExtractor,
    PATH_BODY, PATH_HAIR, PATH_FACE, DATASET_PATH
)

class FeatureDiagnostics:
    """Comprehensive diagnostics for feature quality"""
    
    def __init__(self):
        print("Initializing diagnostics...")
        self.pose_model = PoseEstimator()
        self.body_extractor = BodyExtractor(PATH_BODY)
        self.hair_extractor = HairExtractor(PATH_HAIR)
        self.face_extractor = FaceExtractor(PATH_FACE)
        
    def analyze_feature_distribution(self, samples: list, n_samples: int = 100):
        """Analyze feature distributions to detect overfitting"""
        print(f"\nAnalyzing feature distributions ({n_samples} samples)...")
        
        body_feats = []
        hair_feats = []
        face_feats = []
        
        for sample in tqdm(samples[:n_samples]):
            img = cv2.imread(sample.image_path)
            if img is None:
                continue
                
            keypoints = self.pose_model.extract(img)
            
            # Body
            body_feat, _ = self.body_extractor.extract(img)
            if body_feat is not None:
                body_feats.append(body_feat)
            
            if keypoints is not None:
                # Hair
                hair_feat, _ = self.hair_extractor.extract(img, keypoints)
                if hair_feat is not None:
                    hair_feats.append(hair_feat)
                
                # Face
                face_feat, _ = self.face_extractor.extract(img, keypoints)
                if face_feat is not None:
                    face_feats.append(face_feat)
        
        # Convert to arrays
        body_feats = np.array(body_feats)
        hair_feats = np.array(hair_feats) if hair_feats else np.array([[]])
        face_feats = np.array(face_feats) if face_feats else np.array([[]])
        
        print(f"\nFeature Statistics:")
        print(f"  Body: {len(body_feats)} samples extracted")
        print(f"  Hair: {len(hair_feats)} samples extracted")
        print(f"  Face: {len(face_feats)} samples extracted")
        
        # Analyze each modality
        results = {}
        
        for name, feats in [('Body', body_feats), ('Hair', hair_feats), ('Face', face_feats)]:
            if len(feats) == 0 or feats.shape[1] == 0:
                print(f"\n{name}: No features extracted!")
                continue
                
            print(f"\n{name} Features:")
            print(f"  Shape: {feats.shape}")
            print(f"  Mean: {np.mean(feats):.4f}")
            print(f"  Std: {np.std(feats):.4f}")
            print(f"  Min: {np.min(feats):.4f}")
            print(f"  Max: {np.max(feats):.4f}")
            
            # Check for dead features (always zero or always same value)
            feat_stds = np.std(feats, axis=0)
            dead_features = np.sum(feat_stds < 0.001)
            print(f"  Dead features (std < 0.001): {dead_features}/{feats.shape[1]} ({dead_features/feats.shape[1]*100:.1f}%)")
            
            # Check feature norms
            norms = np.linalg.norm(feats, axis=1)
            print(f"  Feature norms - Mean: {np.mean(norms):.4f}, Std: {np.std(norms):.4f}")
            
            # Check sparsity
            zero_ratio = np.sum(np.abs(feats) < 0.001) / feats.size
            print(f"  Sparsity (near-zero values): {zero_ratio*100:.1f}%")
            
            results[name] = {
                'mean': np.mean(feats),
                'std': np.std(feats),
                'dead_features': dead_features,
                'total_features': feats.shape[1],
                'norm_mean': np.mean(norms),
                'norm_std': np.std(norms),
                'sparsity': zero_ratio
            }
        
        return results
    
    def analyze_intra_vs_inter_class_similarity(self, samples: list, n_samples: int = 50):
        """Compare intra-class (same person) vs inter-class (different person) similarities"""
        print(f"\nAnalyzing intra vs inter-class similarities ({n_samples} samples)...")
        
        # Group by person ID
        person_groups = defaultdict(list)
        for sample in samples:
            person_groups[sample.person_id].append(sample)
        
        # Select persons with at least 2 images
        valid_persons = [pid for pid, imgs in person_groups.items() if len(imgs) >= 2]
        selected_persons = valid_persons[:min(n_samples, len(valid_persons))]
        
        intra_body, inter_body = [], []
        intra_hair, inter_hair = [], []
        intra_face, inter_face = [], []
        
        for pid in tqdm(selected_persons):
            images = person_groups[pid][:4]  # Max 4 images per person
            
            # Extract features for all images of this person
            features = []
            for sample in images:
                img = cv2.imread(sample.image_path)
                if img is None:
                    continue
                    
                keypoints = self.pose_model.extract(img)
                
                body_feat, _ = self.body_extractor.extract(img)
                hair_feat, face_feat = None, None
                
                if keypoints is not None:
                    hair_feat, _ = self.hair_extractor.extract(img, keypoints)
                    face_feat, _ = self.face_extractor.extract(img, keypoints)
                
                features.append({
                    'body': body_feat,
                    'hair': hair_feat,
                    'face': face_feat
                })
            
            # Compute intra-class similarities (same person)
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    f1, f2 = features[i], features[j]
                    
                    if f1['body'] is not None and f2['body'] is not None:
                        sim = 1 - cosine(f1['body'], f2['body'])
                        intra_body.append(sim)
                    
                    if f1['hair'] is not None and f2['hair'] is not None:
                        sim = 1 - cosine(f1['hair'], f2['hair'])
                        intra_hair.append(sim)
                    
                    if f1['face'] is not None and f2['face'] is not None:
                        sim = 1 - cosine(f1['face'], f2['face'])
                        intra_face.append(sim)
            
            # Compute inter-class similarities (different persons)
            # Compare with a few other random persons
            other_persons = [p for p in selected_persons if p != pid]
            for other_pid in other_persons[:5]:
                other_images = person_groups[other_pid][:2]
                
                for other_sample in other_images:
                    other_img = cv2.imread(other_sample.image_path)
                    if other_img is None:
                        continue
                    
                    other_kps = self.pose_model.extract(other_img)
                    other_body, _ = self.body_extractor.extract(other_img)
                    other_hair, other_face = None, None
                    
                    if other_kps is not None:
                        other_hair, _ = self.hair_extractor.extract(other_img, other_kps)
                        other_face, _ = self.face_extractor.extract(other_img, other_kps)
                    
                    # Compare with first image of current person
                    f1 = features[0]
                    
                    if f1['body'] is not None and other_body is not None:
                        sim = 1 - cosine(f1['body'], other_body)
                        inter_body.append(sim)
                    
                    if f1['hair'] is not None and other_hair is not None:
                        sim = 1 - cosine(f1['hair'], other_hair)
                        inter_hair.append(sim)
                    
                    if f1['face'] is not None and other_face is not None:
                        sim = 1 - cosine(f1['face'], other_face)
                        inter_face.append(sim)
        
        # Print results
        print("\n" + "="*80)
        print("INTRA-CLASS (Same Person) vs INTER-CLASS (Different Person) Similarity")
        print("="*80)
        print(f"{'Modality':<15} | {'Intra Mean':<12} | {'Inter Mean':<12} | {'Separation':<12}")
        print("-" * 80)
        
        for name, intra, inter in [
            ('Body', intra_body, inter_body),
            ('Hair', intra_hair, inter_hair),
            ('Face', intra_face, inter_face)
        ]:
            if len(intra) > 0 and len(inter) > 0:
                intra_mean = np.mean(intra)
                inter_mean = np.mean(inter)
                separation = intra_mean - inter_mean
                
                print(f"{name:<15} | {intra_mean:>11.4f} | {inter_mean:>11.4f} | {separation:>11.4f}")
                
                # Good separation: intra > inter by significant margin
                if separation < 0.1:
                    print(f"  WARNING: Poor separation! Model may be overfitting or undertrained.")
                elif separation < 0.2:
                    print(f"  CAUTION: Moderate separation. Consider more training or better data.")
                else:
                    print(f"  GOOD: Strong separation between same/different persons.")
            else:
                print(f"{name:<15} | No valid pairs")
        
        print("="*80)
        
        return {
            'body': {'intra': intra_body, 'inter': inter_body},
            'hair': {'intra': intra_hair, 'inter': inter_hair},
            'face': {'intra': intra_face, 'inter': inter_face}
        }
    
    def analyze_correlation_between_modalities(self, samples: list, n_samples: int = 100):
        """Check if local features are redundant with body features"""
        print(f"\nAnalyzing correlation between modalities ({n_samples} samples)...")
        
        # For each pair of same person, compute similarities in each modality
        person_groups = defaultdict(list)
        for sample in samples:
            person_groups[sample.person_id].append(sample)
        
        valid_persons = [pid for pid, imgs in person_groups.items() if len(imgs) >= 2]
        selected_persons = valid_persons[:min(n_samples, len(valid_persons))]
        
        body_sims, hair_sims, face_sims = [], [], []
        
        for pid in tqdm(selected_persons):
            images = person_groups[pid][:3]
            
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    img1 = cv2.imread(images[i].image_path)
                    img2 = cv2.imread(images[j].image_path)
                    
                    if img1 is None or img2 is None:
                        continue
                    
                    kps1 = self.pose_model.extract(img1)
                    kps2 = self.pose_model.extract(img2)
                    
                    # Body
                    body1, _ = self.body_extractor.extract(img1)
                    body2, _ = self.body_extractor.extract(img2)
                    
                    body_sim = None
                    if body1 is not None and body2 is not None:
                        body_sim = 1 - cosine(body1, body2)
                    
                    # Hair
                    hair_sim = None
                    if kps1 is not None and kps2 is not None:
                        hair1, _ = self.hair_extractor.extract(img1, kps1)
                        hair2, _ = self.hair_extractor.extract(img2, kps2)
                        if hair1 is not None and hair2 is not None:
                            hair_sim = 1 - cosine(hair1, hair2)
                    
                    # Face
                    face_sim = None
                    if kps1 is not None and kps2 is not None:
                        face1, _ = self.face_extractor.extract(img1, kps1)
                        face2, _ = self.face_extractor.extract(img2, kps2)
                        if face1 is not None and face2 is not None:
                            face_sim = 1 - cosine(face1, face2)
                    
                    if body_sim is not None:
                        body_sims.append(body_sim)
                        if hair_sim is not None:
                            hair_sims.append(hair_sim)
                        else:
                            hair_sims.append(np.nan)
                        
                        if face_sim is not None:
                            face_sims.append(face_sim)
                        else:
                            face_sims.append(np.nan)
        
        # Compute correlations
        print("\n" + "="*80)
        print("CORRELATION BETWEEN MODALITY SIMILARITIES")
        print("="*80)
        
        # Remove NaN values for correlation
        valid_idx = ~(np.isnan(hair_sims) | np.isnan(face_sims))
        body_clean = np.array(body_sims)[valid_idx]
        hair_clean = np.array(hair_sims)[valid_idx]
        face_clean = np.array(face_sims)[valid_idx]
        
        if len(body_clean) > 10:
            corr_body_hair, _ = pearsonr(body_clean, hair_clean)
            corr_body_face, _ = pearsonr(body_clean, face_clean)
            corr_hair_face, _ = pearsonr(hair_clean, face_clean)
            
            print(f"Body-Hair correlation: {corr_body_hair:.4f}")
            print(f"Body-Face correlation: {corr_body_face:.4f}")
            print(f"Hair-Face correlation: {corr_hair_face:.4f}")
            print()
            
            if corr_body_hair > 0.8:
                print("WARNING: Hair features are highly correlated with body (redundant!)")
            if corr_body_face > 0.8:
                print("WARNING: Face features are highly correlated with body (redundant!)")
            if corr_hair_face > 0.8:
                print("INFO: Hair and face features are correlated (expected)")
        else:
            print("Not enough valid pairs for correlation analysis")
        
        print("="*80)
        
        return {
            'body_hair_corr': corr_body_hair if len(body_clean) > 10 else None,
            'body_face_corr': corr_body_face if len(body_clean) > 10 else None,
            'hair_face_corr': corr_hair_face if len(body_clean) > 10 else None
        }
    
    def test_quality_filtering_impact(self, samples: list, n_samples: int = 50):
        """Test if low quality local features are hurting performance"""
        print(f"\nTesting quality filtering impact ({n_samples} samples)...")
        
        person_groups = defaultdict(list)
        for sample in samples:
            person_groups[sample.person_id].append(sample)
        
        valid_persons = [pid for pid, imgs in person_groups.items() if len(imgs) >= 2]
        selected_persons = valid_persons[:min(n_samples, len(valid_persons))]
        
        # Collect data: quality scores and whether match was correct
        hair_data = {'quality': [], 'helpful': []}
        face_data = {'quality': [], 'helpful': []}
        
        for pid in tqdm(selected_persons):
            images = person_groups[pid][:2]
            if len(images) < 2:
                continue
                
            img1 = cv2.imread(images[0].image_path)
            img2 = cv2.imread(images[1].image_path)
            
            if img1 is None or img2 is None:
                continue
            
            kps1 = self.pose_model.extract(img1)
            kps2 = self.pose_model.extract(img2)
            
            if kps1 is None or kps2 is None:
                continue
            
            # Body baseline
            body1, _ = self.body_extractor.extract(img1)
            body2, _ = self.body_extractor.extract(img2)
            
            if body1 is None or body2 is None:
                continue
            
            body_sim = 1 - cosine(body1, body2)
            
            # Hair
            hair1, qual1_hair = self.hair_extractor.extract(img1, kps1)
            hair2, qual2_hair = self.hair_extractor.extract(img2, kps2)
            
            if hair1 is not None and hair2 is not None:
                hair_sim = 1 - cosine(hair1, hair2)
                avg_quality = (qual1_hair.overall_quality + qual2_hair.overall_quality) / 2
                
                # Did hair improve the ranking?
                combined_sim = 0.7 * body_sim + 0.3 * hair_sim
                helpful = combined_sim > body_sim  # Simplified: did it increase score?
                
                hair_data['quality'].append(avg_quality)
                hair_data['helpful'].append(1 if helpful else 0)
            
            # Face
            face1, qual1_face = self.face_extractor.extract(img1, kps1)
            face2, qual2_face = self.face_extractor.extract(img2, kps2)
            
            if face1 is not None and face2 is not None:
                face_sim = 1 - cosine(face1, face2)
                avg_quality = (qual1_face.overall_quality + qual2_face.overall_quality) / 2
                
                combined_sim = 0.7 * body_sim + 0.3 * face_sim
                helpful = combined_sim > body_sim
                
                face_data['quality'].append(avg_quality)
                face_data['helpful'].append(1 if helpful else 0)
        
        # Analyze
        print("\n" + "="*80)
        print("QUALITY vs HELPFULNESS ANALYSIS")
        print("="*80)
        
        for name, data in [('Hair', hair_data), ('Face', face_data)]:
            if len(data['quality']) == 0:
                print(f"{name}: No data")
                continue
            
            qualities = np.array(data['quality'])
            helpful = np.array(data['helpful'])
            
            print(f"\n{name}:")
            print(f"  Total pairs: {len(qualities)}")
            print(f"  Overall helpful rate: {np.mean(helpful)*100:.1f}%")
            
            # Split by quality
            for thresh in [0.3, 0.5, 0.7]:
                high_qual = qualities >= thresh
                if np.sum(high_qual) > 0:
                    helpful_rate = np.mean(helpful[high_qual]) * 100
                    count = np.sum(high_qual)
                    print(f"  Quality >= {thresh:.1f}: {helpful_rate:.1f}% helpful (n={count})")
        
        print("="*80)

def main():
    print("\n" + "="*80)
    print("FEATURE QUALITY DIAGNOSTICS")
    print("="*80)
    
    # Load a subset of data
    gallery_samples = []
    gallery_folder = DATASET_PATH / "bounding_box_test"
    
    for img_path in sorted(list(gallery_folder.glob("*.jpg")))[:300]:
        sample = ReIDSample.parse(img_path.name, str(img_path))
        if sample:
            gallery_samples.append(sample)
    
    print(f"\nLoaded {len(gallery_samples)} samples for diagnosis")
    
    # Run diagnostics
    diag = FeatureDiagnostics()
    
    print("\n" + "="*80)
    print("TEST 1: Feature Distribution Analysis")
    print("="*80)
    dist_results = diag.analyze_feature_distribution(gallery_samples, n_samples=100)
    
    print("\n" + "="*80)
    print("TEST 2: Intra vs Inter-Class Similarity")
    print("="*80)
    class_results = diag.analyze_intra_vs_inter_class_similarity(gallery_samples, n_samples=50)
    
    print("\n" + "="*80)
    print("TEST 3: Modality Correlation Analysis")
    print("="*80)
    corr_results = diag.analyze_correlation_between_modalities(gallery_samples, n_samples=50)
    
    print("\n" + "="*80)
    print("TEST 4: Quality Filtering Impact")
    print("="*80)
    diag.test_quality_filtering_impact(gallery_samples, n_samples=50)
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nKey Insights to Look For:")
    print("1. Dead Features: If > 20%, model is undertrained or overfitting")
    print("2. Intra-Inter Separation: Should be > 0.2 for good discrimination")
    print("3. Correlation: If > 0.8, local features are redundant")
    print("4. Quality: High quality features should have > 60% helpful rate")
    print("="*80)

if __name__ == "__main__":
    main()
