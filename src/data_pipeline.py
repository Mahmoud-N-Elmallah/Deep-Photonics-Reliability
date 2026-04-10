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
    
    # Map experiment type to FFT mode, input channels, and normalization stats
    if experiment_type == "tri_channel":
        fft_mode = "tri_channel"
        input_channels = 3
        norm_mean = [config['stats']['train_original_mean'], config['stats']['train_fft_mean'], config['stats']['train_enhanced_mean']]
        norm_std = [config['stats']['train_original_std'], config['stats']['train_fft_std'], config['stats']['train_enhanced_std']]
    elif experiment_type == "dual_channel":
        fft_mode = "dual"
        input_channels = 2
        norm_mean = [config['stats']['train_original_mean'], config['stats']['train_fft_mean']]
        norm_std = [config['stats']['train_original_std'], config['stats']['train_fft_std']]
    elif experiment_type == "original_only":
        fft_mode = "original_only"
        input_channels = 1
        norm_mean = [config['stats']['train_original_mean']]
        norm_std = [config['stats']['train_original_std']]
    elif experiment_type == "fft_only":
        fft_mode = "fft_only"
        input_channels = 1
        norm_mean = [config['stats']['train_fft_mean']]
        norm_std = [config['stats']['train_fft_std']]
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # 1 Transforms
    train_transform = transforms.Compose([
        v2.Resize((config['augmentations']['resize'], config['augmentations']['resize'])),
        v2.RandomHorizontalFlip(p=config['augmentations']['horizontal_flip_prob']),
        v2.RandomVerticalFlip(p=config['augmentations']['vertical_flip_prob']),
        v2.RandomChoice([
            v2.RandomRotation((0, 0)),
            v2.RandomRotation((90, 90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270))
        ]),
        v2.ElasticTransform(alpha=config['augmentations']['elastic_alpha'], sigma=config['augmentations']['elastic_sigma']), 
        v2.ColorJitter(brightness=config['augmentations']['color_jitter_brightness'], contrast=config['augmentations']['color_jitter_contrast']),
        FftTransform(
            width=config['fft_params']['notch_width'], 
            depth=config['fft_params']['notch_depth'],
            apply_bilateral=config['fft_params']['apply_bilateral'],
            mode=fft_mode,
            enhancement_method=config['fft_params'].get('enhancement_method', 'clahe'),
            clahe_clip_limit=config['fft_params'].get('clahe_clip_limit', 3.0)
        ),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(
            mean=norm_mean,
            std=norm_std)])
    
    val_transform = transforms.Compose([
        v2.Resize((config['augmentations']['resize'], config['augmentations']['resize'])),
        FftTransform(
            width=config['fft_params']['notch_width'], 
            depth=config['fft_params']['notch_depth'],
            apply_bilateral=config['fft_params']['apply_bilateral'],
            mode=fft_mode,
            enhancement_method=config['fft_params'].get('enhancement_method', 'clahe'),
            clahe_clip_limit=config['fft_params'].get('clahe_clip_limit', 3.0)
         ),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(
            mean=norm_mean,
            std=norm_std)])

    # 2 Datasets
    if project_root is None:
        project_root = Path.cwd()
    train_ds = DatasetMaker(config['data_path']['train'], transforms=train_transform, project_root=project_root, experiment_type=experiment_type)
    val_ds = DatasetMaker(config['data_path']['val'], transforms=val_transform, project_root=project_root, experiment_type=experiment_type)
    test_ds = DatasetMaker(config['data_path']['test'], transforms=val_transform, project_root=project_root, experiment_type=experiment_type)

    # 3 Loaders (class imbalance handled via loss function weights)
    dl_config = config.get('dataloader', {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True})
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, 
                              num_workers=dl_config['num_workers'], pin_memory=dl_config['pin_memory'], persistent_workers=dl_config['persistent_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                            num_workers=dl_config['num_workers'], pin_memory=dl_config['pin_memory'], persistent_workers=dl_config['persistent_workers'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], 
                             num_workers=dl_config['num_workers'], pin_memory=dl_config['pin_memory'], persistent_workers=dl_config['persistent_workers'])
    
    return train_loader, val_loader, test_loader, input_channels