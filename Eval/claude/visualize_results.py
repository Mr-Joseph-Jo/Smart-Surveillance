"""
Result Visualization and Comparison Tool
Helps visualize the improvement (or lack thereof) in multi-granularity fusion
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def plot_comparison(results_file='multi_granularity_results.json'):
    """Plot comparison of different fusion strategies"""
    
    # Check if results exist
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("Please run multi_granularity_reid_v2.py first!")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    methods = [r['name'] for r in results]
    r1 = [r['r1'] * 100 for r in results]
    r5 = [r['r5'] * 100 for r in results]
    r10 = [r['r10'] * 100 for r in results]
    map_scores = [r['map'] * 100 for r in results]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Granularity Person Re-ID Results Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Colors
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Plot 1: Rank-1 Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.barh(methods, r1, color=colors)
    ax1.set_xlabel('Rank-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Rank-1 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.axvline(x=86, color='red', linestyle='--', linewidth=2, label='Baseline (86%)')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, r1)):
        ax1.text(val + 0.5, i, f'{val:.2f}%', va='center', fontweight='bold')
    
    # Plot 2: Rank-5 Accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.barh(methods, r5, color=colors)
    ax2.set_xlabel('Rank-5 Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Rank-5 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars2, r5)):
        ax2.text(val + 0.5, i, f'{val:.2f}%', va='center', fontweight='bold')
    
    # Plot 3: mAP
    ax3 = axes[1, 0]
    bars3 = ax3.barh(methods, map_scores, color=colors)
    ax3.set_xlabel('mAP (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Average Precision (mAP) Comparison', fontsize=14, fontweight='bold')
    ax3.axvline(x=70.62, color='red', linestyle='--', linewidth=2, label='Baseline (70.62%)')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars3, map_scores)):
        ax3.text(val + 0.5, i, f'{val:.2f}%', va='center', fontweight='bold')
    
    # Plot 4: Improvement over baseline
    ax4 = axes[1, 1]
    baseline_r1 = r1[0]  # Assume first result is baseline
    improvements = [r - baseline_r1 for r in r1]
    
    colors_imp = ['gray' if imp <= 0 else 'green' for imp in improvements]
    bars4 = ax4.barh(methods, improvements, color=colors_imp)
    ax4.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Rank-1 Improvement Analysis', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='red', linestyle='-', linewidth=2)
    ax4.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars4, improvements)):
        color = 'green' if val > 0 else 'red'
        symbol = '+' if val >= 0 else ''
        ax4.text(val + (0.2 if val > 0 else -0.2), i, 
                f'{symbol}{val:.2f}%', va='center', 
                fontweight='bold', color=color,
                ha='left' if val > 0 else 'right')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'results_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    best_r1_idx = np.argmax(r1)
    best_map_idx = np.argmax(map_scores)
    
    print(f"\n🏆 Best Rank-1: {methods[best_r1_idx]} ({r1[best_r1_idx]:.2f}%)")
    print(f"🏆 Best mAP: {methods[best_map_idx]} ({map_scores[best_map_idx]:.2f}%)")
    
    if improvements[best_r1_idx] > 0:
        print(f"\n✅ SUCCESS: {improvements[best_r1_idx]:.2f}% improvement over baseline!")
        print(f"   Recommended strategy: {methods[best_r1_idx]}")
    else:
        print(f"\n❌ PROBLEM: No improvement over baseline")
        print(f"   Best result still {improvements[best_r1_idx]:.2f}% below baseline")
        print("\n   Recommendations:")
        print("   1. Run diagnostic_tool.py to check feature quality")
        print("   2. Verify you have three separately trained models")
        print("   3. Consider retraining with training_improvements.py")
        print("   4. Check TROUBLESHOOTING_GUIDE.md for detailed solutions")
    
    print("="*80)
    
    return fig

def analyze_diagnostic_results():
    """Provide recommendations based on diagnostic results"""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC INTERPRETATION GUIDE")
    print("="*80)
    
    print("\n📊 Feature Distribution Analysis:")
    print("   - Dead features > 20%: Model is undertrained or overfitting")
    print("   - Feature norms highly variable: Poor normalization during training")
    print("   - High sparsity (>50%): Model may be too regularized")
    
    print("\n📊 Intra vs Inter-Class Similarity:")
    print("   - Separation < 0.1: CRITICAL - Model cannot distinguish people")
    print("   - Separation 0.1-0.2: MODERATE - Model needs improvement")
    print("   - Separation > 0.2: GOOD - Model has discriminative power")
    
    print("\n📊 Modality Correlation:")
    print("   - Body-Hair/Face > 0.8: Local features are redundant")
    print("   - Body-Hair/Face 0.5-0.8: Some complementary information")
    print("   - Body-Hair/Face < 0.5: Strong complementary features")
    
    print("\n📊 Quality Filtering Impact:")
    print("   - High quality helpful rate > 60%: Quality filtering is effective")
    print("   - High quality helpful rate < 40%: Features not discriminative")
    
    print("\n🔧 Recommended Actions:")
    print("   1. If dead features > 20%:")
    print("      → Retrain with more data augmentation")
    print("      → Increase training epochs")
    print("      → Check if patches are too small")
    
    print("\n   2. If separation < 0.15:")
    print("      → Add triplet loss with hard mining")
    print("      → Increase batch size for better negative sampling")
    print("      → Use larger patches (e.g., 160x160 instead of 128x128)")
    
    print("\n   3. If correlation > 0.8:")
    print("      → Features are redundant, won't help")
    print("      → Try different body parts (hands, shoes)")
    print("      → Use attention mechanism instead of simple fusion")
    
    print("\n   4. If quality filtering doesn't help:")
    print("      → Model quality itself is poor, not just detection")
    print("      → Retrain with better data and augmentation")
    
    print("="*80)

def create_training_comparison():
    """Compare original vs improved training configuration"""
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION COMPARISON")
    print("="*80)
    
    configs = {
        'Configuration': ['Epochs', 'Batch Size', 'Learning Rate', 'Augmentation', 
                         'Loss Function', 'Hard Mining', 'Label Smoothing'],
        'Original': ['50', '32', '0.0003', 'Basic (flip only)', 'CrossEntropy only', 
                    'No', 'No'],
        'Improved': ['150', '64', '0.0003', 'Advanced (8 types)', 
                    'CrossEntropy + Triplet', 'Yes (margin=0.3)', 'Yes'],
        'Impact': ['↑ Generalization', '↑ Gradient quality', '→ Same', 
                  '↑↑ Robustness', '↑ Discrimination', '↑↑ Hard cases', 
                  '↑ Generalization']
    }
    
    print(f"\n{'Configuration':<20} | {'Original':<25} | {'Improved':<25} | {'Impact':<20}")
    print("-" * 95)
    
    for i in range(len(configs['Configuration'])):
        print(f"{configs['Configuration'][i]:<20} | {configs['Original'][i]:<25} | "
              f"{configs['Improved'][i]:<25} | {configs['Impact'][i]:<20}")
    
    print("\n" + "="*80)
    print("Expected Improvement from Better Training:")
    print("  - Feature quality: +20-30%")
    print("  - Intra-Inter separation: +0.05-0.10")
    print("  - Dead features: -10-15%")
    print("  - Final Re-ID performance: +2-4%")
    print("="*80)

def main():
    """Main visualization and analysis"""
    
    print("\n" + "="*80)
    print("MULTI-GRANULARITY RE-ID: RESULT ANALYSIS & VISUALIZATION")
    print("="*80)
    
    # Try to load and visualize results
    try:
        plot_comparison()
        plt.show()
    except Exception as e:
        print(f"\nCould not visualize results: {e}")
        print("Run multi_granularity_reid_v2.py first to generate results!")
    
    # Show diagnostic guide
    analyze_diagnostic_results()
    
    # Show training comparison
    create_training_comparison()
    
    print("\n" + "="*80)
    print("QUICK CHECKLIST FOR SUCCESS")
    print("="*80)
    print("□ Three separate trained models (not same model!)")
    print("□ Diagnostics show good feature quality")
    print("□ Intra-Inter separation > 0.2")
    print("□ Features are complementary (correlation < 0.7)")
    print("□ Using Quality-Aware or Adaptive Gating fusion")
    print("□ Baseline results reproducible (86%)")
    print("="*80)

if __name__ == "__main__":
    main()
