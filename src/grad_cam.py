import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_config, setup_device
from data_pipeline import build_loaders
from model import PhotonicResNet18

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            # Generate CAM for predicted class if none is provided
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        
        # We target a specific index
        target = output[0, target_class]
        target.backward(retain_graph=True)
        
        # Pull out gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Weight activations by the spatial mean metric of gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to CAM (we only care about positive influences for the target class)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Apply prescribed Min-Max Normalization
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max + 1e-8)
        
        return cam, target_class, output[0].detach()

def create_pseudo_mask(cam, percentile=75, kernel_size=3):
    # Calculate threshold strictly based on prescribed percentile
    threshold = np.percentile(cam, percentile)
    
    # Binary thresholding
    mask = (cam > threshold).astype(np.uint8) * 255
    
    # Morphological cleaning
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Morph EX OPEN: removes noise
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Morph EX CLOSE: fills small holes inside the object
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_and_save_visuals(raw_img, fft_img, enhanced_img, cam, mask, out_path, pred_class, title_prefix):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Clip tensors to [0, 1] range to avoid display warnings if stats slightly overshoot
    axes[0].imshow(np.clip(raw_img, 0, 1), cmap='gray')
    axes[0].set_title(f"{title_prefix}\\nRaw | Pred: {pred_class}")
    
    axes[1].imshow(np.clip(fft_img, 0, 1), cmap='gray')
    axes[1].set_title("FFT Filtered")
    
    axes[2].imshow(np.clip(enhanced_img, 0, 1), cmap='gray')
    axes[2].set_title("Enhanced")
    
    axes[3].imshow(np.clip(enhanced_img, 0, 1), cmap='gray')
    im3 = axes[3].imshow(cam, cmap='jet', alpha=0.5)
    axes[3].set_title("Grad-CAM")
    
    axes[4].imshow(mask, cmap='gray')
    axes[4].set_title("Pseudo Mask")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = script_dir / 'config.yaml'
    config = load_config(str(config_path))
    device = setup_device()

    # Configuration for CAM and Pseudo Masks
    cam_cfg = config.get('cam_params', {})
    percentile_threshold = cam_cfg.get('percentile_threshold', 75)
    morph_kernel_size = cam_cfg.get('morph_kernel_size', 3)
    
    out_dir = Path(cam_cfg.get('output_dir', project_root / 'data' / 'pseudo_masks'))
    vis_dir = out_dir / 'visuals'
    mask_dir = out_dir / 'masks'
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build DataLoaders (batch_size=1 necessary for CAM easily associating with image path)
    # Temporarily override dataloader in config to bs=1
    config['batch_size'] = 1
    config['dataloader']['num_workers'] = 0 # Safe sequential execution
    config['dataloader']['persistent_workers'] = False
    experiment_type = config.get('experiment', {}).get('name', 'tri_channel')
    
    train_loader, val_loader, test_loader, input_channels = build_loaders(config, project_root, experiment_type)
    
    # CRITICAL: Overwrite training transforms to be deterministic for CAM generation.
    # This ensures masks align with original raw images instead of random augmentations.
    train_loader.dataset.transform = val_loader.dataset.transform
    
    # 2. Build and Load Model
    num_classes = len(config.get('train_class_count', {0: 1, 1: 1, 2: 1, 3: 1}))
    model = PhotonicResNet18(input_channels=input_channels, num_classes=num_classes).to(device)
    
    checkpoint_dir = project_root / 'checkpoints' / experiment_type
    best_model_name = config.get('experiment', {}).get('best_model_name', 'best_model.pth')
    checkpoint_path = checkpoint_dir / best_model_name
    
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}. Using untrained weights.")

    model.eval()

    # Target Layer: typically the final convolution in ResNet layer4
    target_layer = model.model.layer4[-1]
    cam_extractor = GradCAM(model, target_layer)

    # We need normalization stats to plot visually correct images
    norm_mean = [config['stats']['train_original_mean'], config['stats']['train_fft_mean'], config['stats']['train_enhanced_mean']]
    norm_std = [config['stats']['train_original_std'], config['stats']['train_fft_std'], config['stats']['train_enhanced_std']]

    loaders = {'train': train_loader, 'val': val_loader}
    
    for split_name, loader in loaders.items():
        print(f"\nProcessing {split_name} split with deterministic transforms...")
        os.makedirs(vis_dir / split_name, exist_ok=True)
        os.makedirs(mask_dir / split_name, exist_ok=True)
        
        # Sequentially process images to map masks back to original file names
        for idx in tqdm(range(len(loader.dataset))):
            img_tensor, label = loader.dataset[idx] 
            img_path = loader.dataset.data['path'][idx]
            base_name = Path(img_path).stem
            
            # Formulate [1, C, H, W] tensor for model
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Compute Grad-CAM for predicted class
            cam, pred_class, logits = cam_extractor(input_tensor, target_class=None)
            
            # Calculate confidence score
            probs = F.softmax(logits.unsqueeze(0), dim=1)
            confidence = probs[0, pred_class].item()
            
            # Generate Pseudo Mask based on Grad-CAM activations
            mask = create_pseudo_mask(cam, percentile=percentile_threshold, kernel_size=morph_kernel_size)
            
            # Save raw mask for downstream physics-constrained training/analysis
            cv2.imwrite(str(mask_dir / split_name / f"{base_name}_mask.png"), mask)
            
            # Generate visualization for qualitative analysis
            img_tensor_cpu = img_tensor.clone().cpu()
            denormed_img = denormalize(img_tensor_cpu, norm_mean[:input_channels], norm_std[:input_channels]).numpy()
            
            raw_channel = denormed_img[0]
            fft_channel = denormed_img[1] if input_channels > 1 else raw_channel
            enh_channel = denormed_img[2] if input_channels == 3 else fft_channel
            
            plot_and_save_visuals(
                raw_channel, fft_channel, enh_channel, cam, mask, 
                vis_dir / split_name / f"{base_name}_cam.jpg", 
                pred_class, 
                f"Conf: {confidence:.2f}"
            )

    print("Grad-CAM generation and Pseudo-mask creation complete. Results saved to data/pseudo_masks.")

if __name__ == '__main__':
    main()
