import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision import transforms
from typing import Dict
from pathlib import Path
from physics_utils import *
from dataset import *

def build_loaders(config: Dict, project_root: Path = None):

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
    v2.ElasticTransform(alpha=50.0, sigma=5.0), 
    FftTransform(
        width=config['fft_params']['notch_width'], 
        depth=config['fft_params']['notch_depth'],
        apply_bilateral=config['fft_params']['apply_bilateral'],
        dual_channel=config['fft_params']['dual_channel'] 
    ),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.Normalize(
        mean=[config['stats']['train_original_mean'], config['stats']['train_fft_mean']],
        std=[config['stats']['train_original_std'], config['stats']['train_fft_std']])])
   

    val_transform = transforms.Compose([
        FftTransform(
        width=config['fft_params']['notch_width'], 
        notch_depth=config['fft_params']['notch_depth'],
        apply_bilateral=config['fft_params']['apply_bilateral'],
        dual_channel=config['fft_params']['dual_channel'] ),
        v2.Normalize(
        mean=[config['stats']['train_original_mean'], config['stats']['train_fft_mean']],
        std=[config['stats']['train_original_std'], config['stats']['train_fft_std']])])

    # 2 Datasets
    if project_root is None:
        project_root = Path.cwd()
    train_ds = DatasetMaker(config['data_path']['train'], transforms=train_transform, project_root=project_root)
    val_ds = DatasetMaker(config['data_path']['val'], transforms=val_transform, project_root=project_root)
    test_ds = DatasetMaker(config['data_path']['test'], transforms=val_transform, project_root=project_root) 

    # 3 Sampler (Weighted to handle imbalancedd classes)
    sample_weights = [config['class_weight'][label] for label in train_ds.data['label']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    # 4 loaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader=DataLoader(test_ds,batch_size=config['batch_size'])
    
    return train_loader, val_loader,test_loader