import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import transforms
from typing import Dict
from pathlib import Path
from physics_utils import *
from dataset import *

def build_loaders(config: Dict, project_root: Path = None, experiment_type: str = None):
    # Determine experiment type from config if not provided
    if experiment_type is None:
        experiment_type = config.get('experiment', {}).get('name', 'dual_channel')
    
    # Map experiment type to FFT mode and input channels
    if experiment_type == "dual_channel":
        fft_mode = "dual"
        input_channels = 2
    elif experiment_type == "original_only":
        fft_mode = "original_only"
        input_channels = 1
    elif experiment_type == "fft_only":
        fft_mode = "fft_only"
        input_channels = 1
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # 1 Transforms
    train_transform = transforms.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomChoice([
            v2.RandomRotation((0, 0)),
            v2.RandomRotation((90, 90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270))
        ]),
        v2.ElasticTransform(alpha=20.0, sigma=2.5), 
        FftTransform(
            width=config['fft_params']['notch_width'], 
            depth=config['fft_params']['notch_depth'],
            apply_bilateral=config['fft_params']['apply_bilateral'],
            mode=fft_mode
        ),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        v2.Normalize(
            mean=[config['stats']['train_original_mean'], config['stats']['train_fft_mean']],
            std=[config['stats']['train_original_std'], config['stats']['train_fft_std']])])
    
    val_transform = transforms.Compose([
        FftTransform(
            width=config['fft_params']['notch_width'], 
            depth=config['fft_params']['notch_depth'],
            apply_bilateral=config['fft_params']['apply_bilateral'],
            mode=fft_mode),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(
            mean=[config['stats']['train_original_mean'], config['stats']['train_fft_mean']],
            std=[config['stats']['train_original_std'], config['stats']['train_fft_std']])])

    # 2 Datasets
    if project_root is None:
        project_root = Path.cwd()
    train_ds = DatasetMaker(config['data_path']['train'], transforms=train_transform, project_root=project_root, experiment_type=experiment_type)
    val_ds = DatasetMaker(config['data_path']['val'], transforms=val_transform, project_root=project_root, experiment_type=experiment_type)
    test_ds = DatasetMaker(config['data_path']['test'], transforms=val_transform, project_root=project_root, experiment_type=experiment_type)

    # 3 Loaders (class imbalance handled via loss function weights)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])
    
    return train_loader, val_loader, test_loader, input_channels
    
    return train_loader, val_loader,test_loader