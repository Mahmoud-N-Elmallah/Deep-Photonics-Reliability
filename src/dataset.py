import torch 
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader,Dataset
import pandas as pd 
import cv2 
import numpy as np 
from pathlib import Path
from physics_utils import *
from PIL import Image

class FftTransform(object):
    def __init__(self, width=0.05, depth=0.99, apply_bilateral=True, mode="tri_channel", enhancement_method="clahe", clahe_clip_limit=3.0):
        self.width = width
        self.notch_depth = depth
        self.apply_bilateral = apply_bilateral
        self.mode = mode
        self.enhancement_method = enhancement_method
        self.clahe_clip_limit = clahe_clip_limit
    
    def __call__(self, img):
        if self.mode == "original_only": return img
        img = np.array(img).astype(np.uint8)
        dft_shift, spectrum = compute_fft_spectrum(img)
        filtered_dft, mask = apply_gaussian_notch_filter(dft_shift, width=self.width, notch_depth=self.notch_depth)
        cleaned_img = reconstruct_image(filtered_dft, apply_bilateral=self.apply_bilateral)
        if self.mode == "tri_channel":
            enhanced_img = enhance_defects_clahe(cleaned_img, clip_limit=self.clahe_clip_limit) if self.enhancement_method == "clahe" else cleaned_img
            tri_channel = np.stack([img, cleaned_img, enhanced_img], axis=2)
            return Image.fromarray(tri_channel.astype(np.uint8))
        elif self.mode == "dual":
            dual_channel = np.stack([img, cleaned_img], axis=2)
            return Image.fromarray(dual_channel.astype(np.uint8))
        return Image.fromarray(cleaned_img)

class JointTransform:
    def __init__(self, config, norm_mean, norm_std, fft_mode, is_train=True):
        self.is_joint = True
        self.is_train = is_train
        self.resize = v2.Resize((config['augmentations']['resize'], config['augmentations']['resize']))
        self.fft = FftTransform(width=config['fft_params']['notch_width'], depth=config['fft_params']['notch_depth'], 
                                apply_bilateral=config['fft_params']['apply_bilateral'], mode=fft_mode)
        self.normalize = v2.Normalize(mean=norm_mean, std=norm_std)
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.hflip = v2.RandomHorizontalFlip(p=config['augmentations']['horizontal_flip_prob'])
        self.vflip = v2.RandomVerticalFlip(p=config['augmentations']['vertical_flip_prob'])

    def __call__(self, img, mask):
        img, mask = self.resize(img), self.resize(mask)
        if self.is_train:
            state = torch.get_rng_state()
            img = self.hflip(img)
            torch.set_rng_state(state)
            mask = self.hflip(mask)
            state = torch.get_rng_state()
            img = self.vflip(img)
            torch.set_rng_state(state)
            mask = self.vflip(mask)
        img = self.fft(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        if not isinstance(mask, torch.Tensor): mask = transforms.functional.to_tensor(mask)
        return img, mask

class DatasetMaker(Dataset):
    def __init__(self, data_csv_path, transforms=None, project_root=None, experiment_type="dual_channel"):
        super().__init__()
        self.transform = transforms
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.data = pd.read_csv(self.project_root / data_csv_path)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_path = self.project_root / 'data' / self.data['path'][idx]
        img = Image.open(img_path).convert('L')
        y = self.data['label'][idx]
        if self.transform: img = self.transform(img)
        return img, y

class PhysicsDataset(Dataset):
    def __init__(self, main_csv_path, mask_mapping_csv_path, transform=None, project_root=None, coverage_threshold=0.35):
        super().__init__()
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.main_data = pd.read_csv(self.project_root / main_csv_path)
        self.mask_mapping = pd.read_csv(self.project_root / mask_mapping_csv_path)
        self.main_data['image_name'] = self.main_data['path'].apply(lambda x: Path(x).name)
        self.data = pd.merge(self.main_data, self.mask_mapping, on='image_name')
        self.transform = transform
        self.coverage_threshold = coverage_threshold # Filter out noisy "hallucinated" masks

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(self.project_root / 'data' / row['path']).convert('L')
        mask = Image.open(self.project_root / 'data' / row['mask_rel_path']).convert('L')
        label = row['label']
        
        # QUALITY FILTER
        # Cracks shouldn't cover 35% of the total cell area
        mask_np = np.array(mask)
        coverage = np.mean(mask_np > 127) 
        
        # is_valid_mask: 0 noisy, 1  good quality
        # handle this in the loss function to  zero the physics loss
        is_valid = 1.0
        if coverage > self.coverage_threshold or coverage < 0.001:
            is_valid = 0.0 # Mark as unreliable
        
        if self.transform:
            if hasattr(self.transform, 'is_joint'): img, mask = self.transform(img, mask)
            else: img = self.transform(img)
        
        if not isinstance(mask, torch.Tensor): mask = transforms.functional.to_tensor(mask)
            
        return img, mask, label, is_valid
