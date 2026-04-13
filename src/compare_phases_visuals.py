import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import add_config_argument, load_runtime_config, setup_device
from data_pipeline import build_loaders
from model import PhotonicResNet18
from grad_cam import GradCAM, denormalize

def compare_models(config_arg: str):
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config, config_path = load_runtime_config(config_arg, project_root)
    device = setup_device()
    
    experiment_type = config.get('experiment', {}).get('name', 'tri_channel')
    
    
    p3_path = project_root / 'checkpoints' / experiment_type / 'best_model.pth'
    p4_path = project_root / 'checkpoints' / 'phase4_physics' / 'best_model.pth'
    
    if not p3_path.exists() or not p4_path.exists():
        raise FileNotFoundError(
            f"Comparison requires both checkpoints. Missing one of: {p3_path}, {p4_path}"
        )

    
    # Force batch_size=1 and deterministic
    config['batch_size'] = 1
    _, val_loader, _, input_channels = build_loaders(config, project_root, experiment_type)
    
    
    num_classes = len(config.get('train_class_count', {0: 1, 1: 1, 2: 1, 3: 1}))
    
    model_p3 = PhotonicResNet18(input_channels=input_channels, num_classes=num_classes).to(device)
    model_p3.load_state_dict(torch.load(str(p3_path), map_location=device))
    model_p3.eval()
    
    model_p4 = PhotonicResNet18(input_channels=input_channels, num_classes=num_classes).to(device)
    model_p4.load_state_dict(torch.load(str(p4_path), map_location=device))
    model_p4.eval()

    
    cam_p3 = GradCAM(model_p3, model_p3.model.layer4[-1])
    cam_p4 = GradCAM(model_p4, model_p4.model.layer4[-1])
    
    norm_mean = [config['stats']['train_original_mean'], config['stats']['train_fft_mean'], config['stats']['train_enhanced_mean']]
    norm_std = [config['stats']['train_original_std'], config['stats']['train_fft_std'], config['stats']['train_enhanced_std']]
    
    # Process  samples
    output_dir = project_root / 'data' / 'phase_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating phase comparison images...")
    
    
    sample_count = 0
    max_samples = 10
    
    for idx in range(len(val_loader.dataset)):
        img_tensor, label = val_loader.dataset[idx]
        if label == 0: continue # Skip normal cells
        
        img_path = val_loader.dataset.data['path'][idx]
        base_name = Path(img_path).stem
        
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get CAMs
        cam3, pred3, logits3 = cam_p3(input_tensor)
        cam4, pred4, logits4 = cam_p4(input_tensor)
        
        conf3 = F.softmax(logits3.unsqueeze(0), dim=1)[0, pred3].item()
        conf4 = F.softmax(logits4.unsqueeze(0), dim=1)[0, pred4].item()
        
        # Denormalize image for plots
        img_cpu = img_tensor.clone().cpu()
        denormed = denormalize(img_cpu, norm_mean[:input_channels], norm_std[:input_channels]).numpy()
        raw_img = np.clip(denormed[0], 0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Image: {base_name} | Label: {label}", fontsize=14)
        
        axes[0].imshow(raw_img, cmap='gray')
        axes[0].set_title("Raw Input")
        
        axes[1].imshow(raw_img, cmap='gray')
        axes[1].imshow(cam3, cmap='jet', alpha=0.5)
        axes[1].set_title(f"Phase 3 (Standard)\nPred: {pred3} Conf: {conf3:.2f}")
        
        axes[2].imshow(raw_img, cmap='gray')
        axes[2].imshow(cam4, cmap='jet', alpha=0.5)
        axes[2].set_title(f"Phase 4 (Physics-Aware)\nPred: {pred4} Conf: {conf4:.2f}")
        
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{base_name}_comparison.jpg", bbox_inches='tight')
        plt.close()
        
        sample_count += 1
        print(f"  [OK] Compared {base_name}")
        if sample_count >= max_samples: break

    print(f"\nComparison complete! Check the folder: {output_dir} | Config: {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Phase 3 and Phase 4 Grad-CAM outputs.")
    add_config_argument(parser, default="src/config.yaml")
    args = parser.parse_args()
    compare_models(args.config)
