import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from typing import Dict
from physics_utils import *
from dataset_dataloader import *

def build_loaders(config: Dict):

    # 1. Transforms
    train_transform = transforms.Compose([
    FftTransform(
        width=config['fft_params']['notch_width'], 
        notch_depth=config['fft_params']['notch_depth'],
        apply_bilateral=config['fft_params']['apply_bilateral']),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # Rotations (90 deg increase) to not destroy the fft transform
    transforms.RandomChoice([
        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270))
    ]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[config['stats']['train_mean']], 
        std=[config['stats']['train_std']])])
   
    val_transform =transforms.Compose([FftTransform(width=config['fft_params']['notch_width'], notch_depth=config['fft_params']['notch_depth'],
                                                    apply_bilateral=config['fft_params']['apply_bilateral']),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[config['stats']['train_mean']], std=[config['stats']['train_std']])])

    # 2. Datasets
    train_ds = DatasetMaker(config['data_path']['train'], transforms=train_transform)
    val_ds = DatasetMaker(config['data_path']['val'], transforms=val_transform) 
    test_ds=DatasetMaker(config['data_path']['test'], transforms=val_transform) 

    # 3. Sampler (Weighted to handle imbalanced Solar Fault classes)
    sample_weights = [config['class_weight'][label] for label in train_ds.data['label']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    # 4. Loaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader=DataLoader(test_ds,batch_size=config['batch_size'])
    
    return train_loader, val_loader,test_loader