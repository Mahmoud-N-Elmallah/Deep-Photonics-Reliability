import sys
import yaml
import torch
from torchvision import transforms
from pathlib import Path


script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(script_dir))

from dataset import DatasetMaker, FftTransform

config_path = script_dir / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("Loading Dataset with tri_channel mode...")

temp_train_ds = DatasetMaker(
    config['data_path']['train'],
    transforms=transforms.Compose([
        FftTransform(
            width=config['fft_params']['notch_width'], 
            depth=config['fft_params']['notch_depth'],
            apply_bilateral=config['fft_params']['apply_bilateral'],
            mode="tri_channel",
            enhancement_method=config['fft_params'].get('enhancement_method', 'clahe'),
            clahe_clip_limit=config['fft_params'].get('clahe_clip_limit', 3.0)
        ),
        transforms.ToTensor()
    ]), 
    project_root=project_root
)

print(f"Stacking dataset ({len(temp_train_ds)} images)...")
all_images = [temp_train_ds[i][0] for i in range(len(temp_train_ds))]
a = torch.stack(all_images, dim=0)

print(f"Stacked tensor shape: {a.shape}") 

train_original_mean = a[:, 0, :, :].mean().item()
train_original_std = a[:, 0, :, :].std().item()

train_fft_mean = a[:, 1, :, :].mean().item()
train_fft_std = a[:, 1, :, :].std().item()

train_enhanced_mean = a[:, 2, :, :].mean().item()
train_enhanced_std = a[:, 2, :, :].std().item()

print(f"Original -> Mean: {train_original_mean:.4f}, Std: {train_original_std:.4f}")
print(f"FFT      -> Mean: {train_fft_mean:.4f}, Std: {train_fft_std:.4f}")
print(f"Enhanced -> Mean: {train_enhanced_mean:.4f}, Std: {train_enhanced_std:.4f}")

config['stats']['train_original_mean'] = round(train_original_mean, 4)
config['stats']['train_original_std'] = round(train_original_std, 4)

config['stats']['train_fft_mean'] = round(train_fft_mean, 4)
config['stats']['train_fft_std'] = round(train_fft_std, 4)

config['stats']['train_enhanced_mean'] = round(train_enhanced_mean, 4)
config['stats']['train_enhanced_std'] = round(train_enhanced_std, 4)

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config file updated successfully!")
