"""
Complete Market-1501 Evaluation Script
Research Paper Experiments:
1. Pose Only
2. Pose + Hair
3. Pose + Hair + Face
4. Pose + Hair + Face + Color (Full Multi-modal)

Generates: CMC curves, mAP, Rank-1, Rank-5, Rank-10, Rank-20
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


# Import your ReID systems
# from pose_hair_reid_market1501 import MultiModalReID, PoseFeatureExtractor, HairFeatureExtractor


@dataclass
class Market1501Sample:
    """Market-1501 sample"""
    image_path: str
    person_id: int
    camera_id: int
    image: Optional[np.ndarray] = None
    
    @staticmethod
    def parse_filename(filename: str, full_path: str) -> Optional['Market1501Sample']:
        """Parse filename: XXXX_cY_NNNNNN_MM.jpg"""
        try:
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) < 2:
                return None
            
            person_id = int(parts[0])
            camera_id = int(parts[1][1])
            
            # Filter junk
            if person_id <= 0:
                return None
            
            return Market1501Sample(
                image_path=full_path,
                person_id=person_id,
                camera_id=camera_id
            )
        except:
            return None


class Market1501Evaluator:
    """Complete evaluator for Market-1501"""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        
        # FIX 1: Gallery must be the TEST set, not TRAIN set
        self.gallery_dir = self.dataset_root / "bounding_box_test"
        self.query_dir = self.dataset_root / "query"
        
        print("Loading Market-1501...")
        self.gallery = self._load_samples(self.gallery_dir)
        self.query = self._load_samples(self.query_dir)
        
        print(f"✓ Gallery: {len(self.gallery)} images")
        print(f"✓ Query: {len(self.query)} images")
    
    def _load_samples(self, directory: Path) -> List[Market1501Sample]:
        samples = []
        # Support both jpg and png just in case
        for img_path in list(directory.glob("*.jpg")) + list(directory.glob("*.png")):
            sample = Market1501Sample.parse_filename(img_path.name, str(img_path))
            if sample:
                samples.append(sample)
        return samples
    
    def evaluate_experiment(self, reid_system, experiment_name: str) -> Dict:
        """
        Evaluate a single experiment configuration
        """
        print(f"\n{'='*70}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*70}")
        
        # --- 1. Extract Gallery Features ---
        print("Extracting gallery features...")
        gallery_features = []
        failed_gallery = 0
        
        # Pre-process: Resize slightly for YOLO if needed inside your extractor
        # but here we just pass the image
        for sample in tqdm(self.gallery, desc="Gallery"):
            try:
                image = cv2.imread(sample.image_path)
                if image is None:
                    failed_gallery += 1
                    continue
                
                features = reid_system.extract_features(image)
                
                # Check if we got valid features
                if features is not None and any(v is not None for v in features.values()):
                    gallery_features.append({
                        'sample': sample,
                        'features': features
                    })
                else:
                    failed_gallery += 1
            except Exception as e:
                failed_gallery += 1
                continue
        
        print(f"✓ Gallery features: {len(gallery_features)} (failed: {failed_gallery})")
        
        if len(gallery_features) == 0:
            print("CRITICAL ERROR: No gallery features extracted. Check paths or feature extractor.")
            return {'name': experiment_name, 'metrics': {'rank_1': 0, 'mAP': 0}, 'results': []}

        # --- 2. Extract Query Features ---
        print("Extracting query features...")
        query_features = []
        failed_query = 0
        
        for sample in tqdm(self.query, desc="Query"):
            try:
                image = cv2.imread(sample.image_path)
                if image is None:
                    failed_query += 1
                    continue
                
                features = reid_system.extract_features(image)
                
                if features is not None and any(v is not None for v in features.values()):
                    query_features.append({
                        'sample': sample,
                        'features': features
                    })
                else:
                    failed_query += 1
            except Exception:
                failed_query += 1
                continue
        
        print(f"✓ Query features: {len(query_features)} (failed: {failed_query})")
        
        # --- 3. Compute Matching ---
        print("Computing similarities...")
        results = []
        
        for query in tqdm(query_features, desc="Matching"):
            query_sample = query['sample']
            query_feat = query['features']
            
            rankings = []
            for gallery in gallery_features:
                gallery_sample = gallery['sample']
                gallery_feat = gallery['features']
                
                # FIX 2: Standard Market-1501 Evaluation Protocol
                # 1. Skip junk images (ID 0 or -1)
                if gallery_sample.person_id <= 0:
                    continue
                    
                # 2. Skip same person in same camera (The "Junk" Rule)
                # We KEEP "distractors" (different people in same camera)
                if (query_sample.person_id == gallery_sample.person_id and 
                    query_sample.camera_id == gallery_sample.camera_id):
                    continue
                
                # Compute similarity
                try:
                    sim = reid_system.compute_similarity(query_feat, gallery_feat)
                except Exception:
                    sim = 0.0
                
                rankings.append({
                    'gallery_id': gallery_sample.person_id,
                    'camera_id': gallery_sample.camera_id,
                    'similarity': sim,
                    'is_match': gallery_sample.person_id == query_sample.person_id
                })
            
            # Sort by similarity (Highest first)
            rankings = sorted(rankings, key=lambda x: x['similarity'], reverse=True)
            
            if rankings:
                results.append({
                    'query_id': query_sample.person_id,
                    'rankings': rankings
                })
        
        print(f"✓ Matched {len(results)} queries")
        
        # --- 4. Compute Metrics ---
        print("Computing metrics...")
        metrics = self._compute_metrics(results)
        
        self._print_metrics(experiment_name, metrics)
        
        return {
            'name': experiment_name,
            'metrics': metrics,
            'results': results
        }
    
    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute Rank-k and mAP"""
        if not results:
            return {'rank_1': 0.0, 'rank_5': 0.0, 'rank_10': 0.0, 'rank_20': 0.0, 'mAP': 0.0, 'num_queries': 0}

        rank_accuracies = {1: 0, 5: 0, 10: 0, 20: 0}
        aps = []
        
        for result in results:
            rankings = result['rankings']
            
            # Find indices where is_match is True
            correct_indices = [i for i, r in enumerate(rankings) if r['is_match']]
            
            if not correct_indices:
                aps.append(0.0)
                continue
            
            # Rank-k accuracy (Is the first match within top k?)
            first_match_idx = correct_indices[0]
            for k in [1, 5, 10, 20]:
                if first_match_idx < k:
                    rank_accuracies[k] += 1
            
            # Average Precision (AP)
            # AP = (1/num_positive) * sum(precision_at_k * rel_k)
            num_positive = len(correct_indices)
            running_correct = 0
            precision_sum = 0.0
            
            for i, r in enumerate(rankings):
                if r['is_match']:
                    running_correct += 1
                    precision = running_correct / (i + 1)
                    precision_sum += precision
            
            ap = precision_sum / num_positive
            aps.append(ap)
        
        num_queries = len(results)
        
        return {
            'rank_1': rank_accuracies[1] / num_queries,
            'rank_5': rank_accuracies[5] / num_queries,
            'rank_10': rank_accuracies[10] / num_queries,
            'rank_20': rank_accuracies[20] / num_queries,
            'mAP': np.mean(aps) if aps else 0.0,
            'num_queries': num_queries
        }
    
    def _print_metrics(self, name: str, metrics: Dict):
        print(f"\n📊 Results for {name}:")
        print(f"{'─'*50}")
        print(f"  Rank-1:  {metrics['rank_1']*100:.2f}%")
        print(f"  Rank-5:  {metrics['rank_5']*100:.2f}%")
        print(f"  Rank-10: {metrics['rank_10']*100:.2f}%")
        print(f"  Rank-20: {metrics['rank_20']*100:.2f}%")
        print(f"  mAP:     {metrics['mAP']*100:.2f}%")
        print(f"{'─'*50}")

    def plot_cmc_comparison(self, all_experiments: List[Dict], save_path: str = "cmc_comparison.png"):
        """Plot CMC curves for all experiments"""
        if not all_experiments:
            return

        plt.figure(figsize=(10, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        markers = ['o', 's', '^', 'D', 'v']
        
        for idx, experiment in enumerate(all_experiments):
            name = experiment['name']
            results = experiment['results']
            
            if not results:
                continue

            # Compute CMC
            ranks_range = range(1, 21)
            accuracies = []
            
            for k in ranks_range:
                correct = 0
                for result in results:
                    rankings = result['rankings']
                    # Check if any match in top k
                    if any(r['is_match'] for r in rankings[:k]):
                        correct += 1
                accuracy = correct / len(results)
                accuracies.append(accuracy * 100)
            
            plt.plot(ranks_range, accuracies, 
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    markersize=5,
                    label=name)
        
        plt.xlabel('Rank')
        plt.ylabel('Matching Rate (%)')
        plt.title('CMC Curve - Market-1501')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim([1, 20])
        plt.ylim([0, 100])
        plt.savefig(save_path, dpi=300)
        print(f"✓ CMC curve saved: {save_path}")
        plt.close()

    def generate_comparison_table(self, all_experiments: List[Dict], save_path: str = "results_table.csv"):
        """Generate CSV table"""
        data = []
        for experiment in all_experiments:
            m = experiment['metrics']
            data.append({
                'Method': experiment['name'],
                'Rank-1': m['rank_1'],
                'Rank-5': m['rank_5'],
                'mAP': m['mAP']
            })
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"✓ Table saved: {save_path}")

    def generate_latex_table(self, all_experiments: List[Dict], save_path: str = "results_latex.txt"):
        """Generate LaTeX"""
        with open(save_path, 'w') as f:
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\begin{tabular}{l|ccc}\n\\hline\n")
            f.write("Method & Rank-1 & Rank-5 & mAP \\\\\n\\hline\n")
            for exp in all_experiments:
                m = exp['metrics']
                f.write(f"{exp['name']} & {m['rank_1']*100:.1f} & {m['rank_5']*100:.1f} & {m['mAP']*100:.1f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}")
        print(f"✓ LaTeX saved: {save_path}")

    def generate_full_report(self, all_experiments: List[Dict], save_path: str = "full_report.txt"):
        """Generate text report"""
        with open(save_path, 'w') as f:
            f.write("MARKET-1501 EVALUATION REPORT\n")
            f.write("=============================\n\n")
            for exp in all_experiments:
                m = exp['metrics']
                f.write(f"Experiment: {exp['name']}\n")
                f.write(f"  Rank-1: {m['rank_1']*100:.2f}%\n")
                f.write(f"  mAP:    {m['mAP']*100:.2f}%\n\n")
        print(f"✓ Report saved: {save_path}")


# ==================== MAIN EVALUATION SCRIPT ====================

def run_complete_evaluation(dataset_path: str, device: str = 'auto'):
    """
    Run all experiments for research paper
    
    Args:
        dataset_path: Path to Market-1501 dataset
        device: 'auto' (detect GPU), 'cpu', '0', '1', etc.
    
    Experiments:
    1. Pose Only
    2. Pose + Hair
    3. Pose + Hair + Face (TODO)
    4. Pose + Hair + Face + Color (TODO)
    """
    
    print("="*70)
    print("MARKET-1501 MULTI-MODAL RE-IDENTIFICATION EVALUATION")
    print("="*70)
    
    # Check GPU availability
    try:
        import torch
        if device == 'auto':
            if torch.cuda.is_available():
                device = '0'
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA Version: {torch.version.cuda}")
            else:
                device = 'cpu'
                print("⚠ No GPU detected, using CPU (will be slow)")
    except:
        device = 'cpu'
        print("⚠ PyTorch not found, using CPU")
    
    # Import your systems
    from pose_hair_reid_market1501 import MultiModalReID
    
    # Initialize evaluator
    evaluator = Market1501Evaluator(dataset_path)
    
    all_experiments = []
    
    # ===== EXPERIMENT 1: Pose Only =====
    print("\n" + "▓"*70)
    print("▓  EXPERIMENT 1: POSE ONLY")
    print("▓"*70)
    
    pose_only = MultiModalReID(modalities=['pose'], device=device)
    exp1 = evaluator.evaluate_experiment(pose_only, "Pose Only")
    all_experiments.append(exp1)
    
    # ===== EXPERIMENT 2: Pose + Hair =====
    print("\n" + "▓"*70)
    print("▓  EXPERIMENT 2: POSE + HAIR")
    print("▓"*70)
    
    pose_hair = MultiModalReID(modalities=['pose', 'hair'], device=device)
    exp2 = evaluator.evaluate_experiment(pose_hair, "Pose + Hair")
    all_experiments.append(exp2)
    
    # TODO: Add more experiments
    # exp3 = evaluator.evaluate_experiment(pose_hair_face, "Pose + Hair + Face")
    # exp4 = evaluator.evaluate_experiment(full_multimodal, "Full Multi-modal")
    
    # Generate all outputs
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print("="*70)
    
    evaluator.plot_cmc_comparison(all_experiments, "cmc_comparison.png")
    evaluator.generate_comparison_table(all_experiments, "results_table.csv")
    evaluator.generate_latex_table(all_experiments, "results_latex.txt")
    evaluator.generate_full_report(all_experiments, "full_report.txt")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. cmc_comparison.png - CMC curve comparison")
    print("  2. results_table.csv - Results in CSV format")
    print("  3. results_latex.txt - LaTeX table for paper")
    print("  4. full_report.txt - Comprehensive report")
    
    return all_experiments


if __name__ == "__main__":
    # Set your Market-1501 dataset path
    DATASET_PATH = "./Market-1501-v15.09.15"
    
    # Check if path exists
    if not os.path.exists(DATASET_PATH):
        print("ERROR: Dataset path not found!")
        print(f"Please download Market-1501 and set DATASET_PATH to the correct location")
        print("\nExpected structure:")
        print("Market-1501-v15.09.15/")
        print("├── bounding_box_train/")
        print("└── query/")
    else:
        run_complete_evaluation(DATASET_PATH)