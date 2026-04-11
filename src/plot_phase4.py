import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def plot_phase4_results():
    project_root = Path.cwd()
    history_path = project_root / 'checkpoints' / 'phase4_physics' / 'training_history.pkl'
    save_path = project_root / 'results' / 'final_evaluation' / 'phase4_training_curves.png'
    
    if not history_path.exists():
        print("History not found.")
        return

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss Curve
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], color=color, label='Total Loss', linewidth=2)
    ax1.plot(history['physics_loss'], color='tab:orange', linestyle='--', label='Physics Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # F1 Curve
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val F1', color=color)
    ax2.plot(history['val_f1'], color=color, label='Val F1 Score', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Phase 4: Physics-Constrained Training Dynamics')
    fig.tight_layout()
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    plot_phase4_results()
