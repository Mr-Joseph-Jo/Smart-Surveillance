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
        self.gallery_dir = self.dataset_root / "bounding_box_train"
        self.query_dir = self.dataset_root / "query"
        
        print("Loading Market-1501...")
        self.gallery = self._load_samples(self.gallery_dir)
        self.query = self._load_samples(self.query_dir)
        
        print(f"✓ Gallery: {len(self.gallery)} images")
        print(f"✓ Query: {len(self.query)} images")
    
    def _load_samples(self, directory: Path) -> List[Market1501Sample]:
        samples = []
        for img_path in directory.glob("*.jpg"):
            sample = Market1501Sample.parse_filename(img_path.name, str(img_path))
            if sample:
                samples.append(sample)
        return samples
    
    def evaluate_experiment(self, reid_system, experiment_name: str) -> Dict:
        """
        Evaluate a single experiment configuration
        
        Args:
            reid_system: MultiModalReID instance
            experiment_name: e.g., "Pose Only", "Pose+Hair"
        """
        print(f"\n{'='*70}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*70}")
        
        # Extract gallery features
        print("Extracting gallery features...")
        gallery_features = []
        failed_gallery = 0
        
        for sample in tqdm(self.gallery, desc="Gallery"):
            try:
                image = cv2.imread(sample.image_path)
                if image is None:
                    failed_gallery += 1
                    continue
                
                features = reid_system.extract_features(image)
                
                # Only add if features were successfully extracted
                if features is not None and any(features.values()):
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
        
        # Debug: Check feature quality
        if len(gallery_features) > 0:
            sample_feat = gallery_features[0]['features']
            print(f"  Sample feature keys: {list(sample_feat.keys())}")
            print(f"  Sample feature values: {[type(v).__name__ if v else 'None' for v in sample_feat.values()]}")
        
        # Extract query features
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
                
                # Only add if features were successfully extracted
                if features is not None and any(features.values()):
                    query_features.append({
                        'sample': sample,
                        'features': features
                    })
                else:
                    failed_query += 1
            except Exception as e:
                failed_query += 1
                continue
        
        print(f"✓ Query features: {len(query_features)} (failed: {failed_query})")
        
        # Debug: Check feature quality
        if len(query_features) > 0:
            sample_feat = query_features[0]['features']
            print(f"  Sample feature keys: {list(sample_feat.keys())}")
            print(f"  Sample feature values: {[type(v).__name__ if v else 'None' for v in sample_feat.values()]}")
        
        # Compute matches
        print("Computing similarities...")
        results = []
        
        for query in tqdm(query_features, desc="Matching"):
            query_sample = query['sample']
            query_feat = query['features']
            
            # Skip if no features extracted
            if query_feat is None or not any(v is not None for v in query_feat.values()):
                continue
            
            rankings = []
            for gallery in gallery_features:
                gallery_sample = gallery['sample']
                gallery_feat = gallery['features']
                
                # Skip same camera
                if query_sample.camera_id == gallery_sample.camera_id:
                    continue
                
                # Skip if no features
                if gallery_feat is None or not any(v is not None for v in gallery_feat.values()):
                    continue
                
                # Compute similarity
                try:
                    sim = reid_system.compute_similarity(query_feat, gallery_feat)
                except Exception as e:
                    sim = 0.0
                
                rankings.append({
                    'gallery_id': gallery_sample.person_id,
                    'similarity': sim,
                    'is_match': gallery_sample.person_id == query_sample.person_id
                })
            
            # Sort by similarity
            rankings = sorted(rankings, key=lambda x: x['similarity'], reverse=True)
            
            # Only add if we have rankings
            if rankings:
                results.append({
                    'query_id': query_sample.person_id,
                    'rankings': rankings
                })
        
        print(f"✓ Matched {len(results)} queries")
        
        # Compute metrics
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
        rank_accuracies = {1: 0, 5: 0, 10: 0, 20: 0}
        aps = []
        
        for result in results:
            rankings = result['rankings']
            
            # Find correct match ranks
            correct_ranks = [i for i, r in enumerate(rankings) if r['is_match']]
            
            if not correct_ranks:
                aps.append(0.0)
                continue
            
            # Rank-k accuracy
            first_rank = correct_ranks[0]
            for k in [1, 5, 10, 20]:
                if first_rank < k:
                    rank_accuracies[k] += 1
            
            # Average Precision
            precisions = []
            for i, rank in enumerate(correct_ranks):
                precision = (i + 1) / (rank + 1)
                precisions.append(precision)
            
            aps.append(np.mean(precisions) if precisions else 0.0)
        
        num_queries = len(results)
        
        return {
            'rank_1': rank_accuracies[1] / num_queries,
            'rank_5': rank_accuracies[5] / num_queries,
            'rank_10': rank_accuracies[10] / num_queries,
            'rank_20': rank_accuracies[20] / num_queries,
            'mAP': np.mean(aps),
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
        plt.figure(figsize=(12, 7))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        markers = ['o', 's', '^', 'D', 'v']
        
        for idx, experiment in enumerate(all_experiments):
            name = experiment['name']
            results = experiment['results']
            
            # Compute CMC
            ranks = range(1, 21)
            accuracies = []
            
            for k in ranks:
                correct = 0
                for result in results:
                    rankings = result['rankings']
                    top_k_correct = any(r['is_match'] for r in rankings[:k])
                    if top_k_correct:
                        correct += 1
                accuracy = correct / len(results)
                accuracies.append(accuracy * 100)
            
            plt.plot(ranks, accuracies, 
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    linewidth=2.5,
                    markersize=6,
                    label=name)
        
        plt.xlabel('Rank', fontsize=12, fontweight='bold')
        plt.ylabel('Matching Rate (%)', fontsize=12, fontweight='bold')
        plt.title('Cumulative Matching Characteristic (CMC) Curve\nMarket-1501 Dataset', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([1, 20])
        plt.ylim([0, 100])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ CMC curve saved: {save_path}")
        plt.close()
    
    def generate_comparison_table(self, all_experiments: List[Dict], 
                                  save_path: str = "results_table.csv"):
        """Generate comparison table"""
        data = []
        
        for experiment in all_experiments:
            name = experiment['name']
            metrics = experiment['metrics']
            
            data.append({
                'Method': name,
                'Rank-1 (%)': f"{metrics['rank_1']*100:.2f}",
                'Rank-5 (%)': f"{metrics['rank_5']*100:.2f}",
                'Rank-10 (%)': f"{metrics['rank_10']*100:.2f}",
                'Rank-20 (%)': f"{metrics['rank_20']*100:.2f}",
                'mAP (%)': f"{metrics['mAP']*100:.2f}"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        
        print(f"\n✓ Results table saved: {save_path}")
        print("\n" + df.to_string(index=False))
        
        return df
    
    def generate_latex_table(self, all_experiments: List[Dict],
                            save_path: str = "results_latex.txt"):
        """Generate LaTeX table for paper"""
        with open(save_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Comparison on Market-1501 Dataset}\n")
            f.write("\\label{tab:market1501_results}\n")
            f.write("\\begin{tabular}{l|ccccc}\n")
            f.write("\\hline\n")
            f.write("Method & Rank-1 & Rank-5 & Rank-10 & Rank-20 & mAP \\\\\n")
            f.write("\\hline\n")
            
            for experiment in all_experiments:
                name = experiment['name']
                m = experiment['metrics']
                f.write(f"{name} & {m['rank_1']*100:.2f} & {m['rank_5']*100:.2f} & "
                       f"{m['rank_10']*100:.2f} & {m['rank_20']*100:.2f} & {m['mAP']*100:.2f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"✓ LaTeX table saved: {save_path}")
    
    def generate_full_report(self, all_experiments: List[Dict],
                           save_path: str = "full_report.txt"):
        """Generate comprehensive report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MARKET-1501 MULTI-MODAL RE-IDENTIFICATION EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: Market-1501\n")
            f.write(f"Gallery Size: {len(self.gallery)} images\n")
            f.write(f"Query Size: {len(self.query)} images\n\n")
            
            f.write("EXPERIMENTS CONDUCTED:\n")
            for i, exp in enumerate(all_experiments, 1):
                f.write(f"{i}. {exp['name']}\n")
            f.write("\n")
            
            for experiment in all_experiments:
                name = experiment['name']
                metrics = experiment['metrics']
                
                f.write("-"*70 + "\n")
                f.write(f"{name}\n")
                f.write("-"*70 + "\n")
                f.write(f"Rank-1 Accuracy:  {metrics['rank_1']*100:.2f}%\n")
                f.write(f"Rank-5 Accuracy:  {metrics['rank_5']*100:.2f}%\n")
                f.write(f"Rank-10 Accuracy: {metrics['rank_10']*100:.2f}%\n")
                f.write(f"Rank-20 Accuracy: {metrics['rank_20']*100:.2f}%\n")
                f.write(f"Mean Average Precision (mAP): {metrics['mAP']*100:.2f}%\n")
                f.write(f"Queries Evaluated: {metrics['num_queries']}\n\n")
            
            # Performance improvement analysis
            if len(all_experiments) > 1:
                f.write("="*70 + "\n")
                f.write("PERFORMANCE IMPROVEMENT ANALYSIS\n")
                f.write("="*70 + "\n\n")
                
                baseline = all_experiments[0]
                baseline_rank1 = baseline['metrics']['rank_1']
                baseline_map = baseline['metrics']['mAP']
                
                for exp in all_experiments[1:]:
                    rank1_improvement = (exp['metrics']['rank_1'] - baseline_rank1) * 100
                    map_improvement = (exp['metrics']['mAP'] - baseline_map) * 100
                    
                    f.write(f"{exp['name']} vs {baseline['name']}:\n")
                    f.write(f"  Rank-1 improvement: {rank1_improvement:+.2f}%\n")
                    f.write(f"  mAP improvement: {map_improvement:+.2f}%\n\n")
        
        print(f"✓ Full report saved: {save_path}")


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