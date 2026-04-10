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
    def __init__(self, width=0.05, depth=0.99, apply_bilateral=True, mode="dual"):
        """FFT Transform with multiple modes.
        Args:
            mode: "dual" (original + FFT), "fft_only" (FFT only), "original_only" (original only)
        """
        self.width = width
        self.notch_depth = depth
        self.apply_bilateral = apply_bilateral
        self.mode = mode
    
    def __call__(self, img):
        if self.mode == "original_only":
            # Return original image as-is
            return img
        
        # Process FFT for dual_channel and fft_only modes
        img = np.array(img).astype(np.uint8)
        dft_shift, spectrum = compute_fft_spectrum(img)
        filtered_dft, mask = apply_gaussian_notch_filter(dft_shift, width=self.width, notch_depth=self.notch_depth)
        cleaned_img = reconstruct_image(filtered_dft, apply_bilateral=self.apply_bilateral)
        
        if self.mode == "dual":
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
    
