import torch 
import torchvision
from torchvision import transforms
###
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
        """FFT Transform with multiple modes.
        Args:
            mode: "dual" (original + FFT), "fft_only" (FFT only), "original_only" (original only), "tri_channel"
            enhancement_method: "clahe" or "sobel" or "none" (used for tri_channel)
            clahe_clip_limit: Limit for contrast enhancement
        """
        self.width = width
        self.notch_depth = depth
        self.apply_bilateral = apply_bilateral
        self.mode = mode
        self.enhancement_method = enhancement_method
        self.clahe_clip_limit = clahe_clip_limit
    
    def __call__(self, img):
        if self.mode == "original_only":
            # Return original image as-is
            return img
        
        # Process FFT for dual_channel, fft_only, and tri_channel modes
        img = np.array(img).astype(np.uint8)
        dft_shift, spectrum = compute_fft_spectrum(img)
        filtered_dft, mask = apply_gaussian_notch_filter(dft_shift, width=self.width, notch_depth=self.notch_depth)
        cleaned_img = reconstruct_image(filtered_dft, apply_bilateral=self.apply_bilateral)
        
        if self.mode == "tri_channel":
            if self.enhancement_method == "sobel":
                enhanced_img = enhance_defects_sobel(cleaned_img)
            elif self.enhancement_method == "clahe":
                enhanced_img = enhance_defects_clahe(cleaned_img, clip_limit=self.clahe_clip_limit)
            else:
                enhanced_img = cleaned_img # Fallback to just the cleaned image if "none"
            
            tri_channel = np.stack([img, cleaned_img, enhanced_img], axis=2)
            return Image.fromarray(tri_channel.astype(np.uint8))
        elif self.mode == "dual":
            dual_channel = np.stack([img, cleaned_img], axis=2)
            return Image.fromarray(dual_channel.astype(np.uint8))
        elif self.mode == "fft_only":
            return Image.fromarray(cleaned_img)

class DatasetMaker(Dataset):
    def __init__(self, data_csv_path, transforms=None, project_root=None, experiment_type="dual_channel"):
        super().__init__()
        self.transform = transforms
        self.experiment_type = experiment_type
        if project_root is None:
            project_root = Path.cwd()
        self.project_root = Path(project_root)
        self.path = self.project_root / data_csv_path
        self.data = pd.read_csv(self.path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.project_root / 'data' / self.data['path'][idx]
        img = Image.open(img_path).convert('L')
        y = self.data['label'][idx]
        if self.transform:
            img = self.transform(img)
        return img, y
    
