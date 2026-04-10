import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_history(checkpoint_dir):
    """Load training history from checkpoint directory."""
    history_path = checkpoint_dir / 'training_history.pkl'
    if not history_path.exists():
        print(f"Warning: History file not found at {history_path}")
        return None
    with open(history_path, 'rb') as f:
        return pickle.load(f)

def plot_comparison():
    """Create comparison plots for all experiments found in the checkpoints directory."""
    checkpoint_root = Path('checkpoints')
    
    # Dynamically find all experiment directories that contain a history file
    experiments = [d.name for d in checkpoint_root.iterdir() if d.is_dir() and (d / 'training_history.pkl').exists()]
    histories = {}
    
    print(f"Found {len(experiments)} experiments: {', '.join(experiments)}")
    for exp in experiments:
        hist = load_history(checkpoint_root / exp)
        if hist is not None:
            histories[exp] = hist
    
    if not histories:
        print("No training histories found in checkpoints directory.")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Deep Photonics Reliability: Experiment Comparison', fontsize=16, fontweight='bold')
    
    # Define a color map for known experiments, fall back to default for others
    color_map = {
        'tri_channel': '#d62728', # Red
        'dual_channel': '#1f77b4', # Blue
        'original_only': '#ff7f0e', # Orange
        'fft_only': '#2ca02c' # Green
    }
    
    # Plot 1: Validation F1 Score
    for exp in histories:
        c = color_map.get(exp, None)
        axes[0, 0].plot(histories[exp]['val_f1'], label=exp, marker='o', markersize=3, linewidth=2, color=c)
    axes[0, 0].set_title('Validation F1 Score (Primary Metric)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    for exp in histories:
        c = color_map.get(exp, None)
        axes[0, 1].plot(histories[exp]['val_acc'], label=exp, marker='o', markersize=3, linewidth=2, color=c)
    axes[0, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training F1 Score
    for exp in histories:
        c = color_map.get(exp, None)
        axes[1, 0].plot(histories[exp]['train_f1'], label=exp, marker='o', markersize=3, linewidth=2, color=c)
    axes[1, 0].set_title('Training F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Validation Loss
    for exp in histories:
        c = color_map.get(exp, None)
        axes[1, 1].plot(histories[exp]['val_loss'], label=exp, marker='o', markersize=3, linewidth=2, color=c)
    axes[1, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = Path('experiment_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {comparison_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    
    summary_data = []
    for exp in sorted(histories.keys()):
        hist = histories[exp]
        
        best_f1_idx = np.argmax(hist['val_f1'])
        best_f1 = hist['val_f1'][best_f1_idx]
        best_f1_epoch = best_f1_idx + 1
        
        best_acc_idx = np.argmax(hist['val_acc'])
        best_acc = hist['val_acc'][best_acc_idx]
        best_acc_epoch = best_acc_idx + 1
        
        final_f1 = hist['val_f1'][-1]
        final_acc = hist['val_acc'][-1]
        final_loss = hist['val_loss'][-1]
        
        print(f"\n{exp.upper()}:")
        print(f"  Best Val F1:  {best_f1:.4f} (epoch {best_f1_epoch})")
        print(f"  Best Val Acc: {best_acc:.4f} (epoch {best_acc_epoch})")
        print(f"  Final Val F1:  {final_f1:.4f}")
        print(f"  Final Val Acc: {final_acc:.4f}")
        print(f"  Final Val Loss: {final_loss:.4f}")
        print(f"  Total Epochs Trained: {len(hist['val_f1'])}")
        
        summary_data.append({
            'experiment': exp,
            'best_f1': best_f1,
            'best_f1_epoch': best_f1_epoch,
            'final_f1': final_f1,
            'final_acc': final_acc
        })
    
    # Ranking
    print("\n" + "="*80)
    print("RANKINGS (by Best F1 Score):")
    print("="*80)
    sorted_summary = sorted(summary_data, key=lambda x: x['best_f1'], reverse=True)
    for i, data in enumerate(sorted_summary, 1):
        print(f"{i}. {data['experiment']:20s} - Best F1: {data['best_f1']:.4f} (epoch {data['best_f1_epoch']})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    plot_comparison()
