import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import pickle
from utils import load_config, setup_device
from data_pipeline import build_loaders
from model import PhotonicResNet18
from grad_cam import GradCAM, denormalize

def plot_training_curves(history_path, save_path):
    if not history_path.exists():
        return
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], color=color, label='Total Loss', linewidth=2)
    if 'physics_loss' in history:
        ax1.plot(history['physics_loss'], color='tab:orange', linestyle='--', label='Physics Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val F1', color=color)
    ax2.plot(history['val_f1'], color=color, label='Val F1 Score', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Phase 4: Optimization Dynamics')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_final_evaluation():
    # 1. Setup
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config = load_config(str(script_dir / 'config.yaml'))
    device = setup_device()
    
    experiment_type = config.get('experiment', {}).get('name', 'tri_channel')
    # Load the absolute best model from Phase 4
    model_path = project_root / 'checkpoints' / 'phase4_physics' / 'best_model.pth'
    
    results_dir = project_root / 'results' / 'final_evaluation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # 2. Build Test Loader
    # Ensure is_physics=True to handle the 4-tuple return if necessary, 
    # but for simple evaluation we just need image and labels.
    # We use build_loaders with is_physics=False to get the standard Image/Label pairs for simpler eval
    _, _, test_loader, input_channels = build_loaders(config, project_root, experiment_type, is_physics=False)
    
    # 3. Load Model
    num_classes = 4
    model = PhotonicResNet18(
        input_channels=input_channels, 
        num_classes=num_classes,
        dropout_prob=0.0 # No dropout during eval
    ).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    
    # 4. Global Metrics Gathering
    print("\nRunning test set inference...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 5. Statistical Analysis
    target_names = ['Normal', 'Minor-Defect', 'Moderate-Defect', 'Major-Defect']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save Report
    with open(results_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "="*30)
    print("FINAL TEST SET REPORT")
    print("="*30)
    print(report)
    
    # 6. Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Final Confusion Matrix (Phase 4 Physics-Aware)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(results_dir / 'confusion_matrix.png', dpi=300)
    print(f"Confusion Matrix saved to {results_dir / 'confusion_matrix.png'}")
    
    # 7. Quality Check: Random Test Visuals
    print("\nGenerating blind-test visualizations...")
    cam_extractor = GradCAM(model, model.model.layer4[-1])
    norm_mean = [config['stats']['train_original_mean'], config['stats']['train_fft_mean'], config['stats']['train_enhanced_mean']]
    norm_std = [config['stats']['train_original_std'], config['stats']['train_fft_std'], config['stats']['train_enhanced_std']]
    
    indices = np.random.choice(len(test_loader.dataset), 12, replace=False)
    
    for i, idx in enumerate(indices):
        img_tensor, label = test_loader.dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        cam, pred, logits = cam_extractor(input_tensor)
        conf = F.softmax(logits.unsqueeze(0), dim=1)[0, pred].item()
        
        img_cpu = img_tensor.clone().cpu()
        denormed = denormalize(img_cpu, norm_mean[:input_channels], norm_std[:input_channels]).numpy()
        raw_img = np.clip(denormed[0], 0, 1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(raw_img, cmap='gray')
        plt.imshow(cam, cmap='jet', alpha=0.45)
        plt.axis('off')
        
        title = f"Test Sample\nTrue: {target_names[label]} | Pred: {target_names[pred]} ({conf:.1%})"
        if label == pred:
            plt.title(title, color='green')
        else:
            plt.title(title, color='red')
            
        plt.savefig(results_dir / f'blind_test_{i}.jpg', bbox_inches='tight')
        plt.close()

    print(f"Blind Test Visuals saved to {results_dir}")
    
    # 8. Training Metrics Visualization
    print("Generating training optimization curves...")
    history_path = project_root / 'checkpoints' / 'phase4_physics' / 'training_history.pkl'
    plot_training_curves(history_path, results_dir / 'phase4_training_curves.png')
    
    print("\nEvaluation Complete!")

if __name__ == '__main__':
    run_final_evaluation()
