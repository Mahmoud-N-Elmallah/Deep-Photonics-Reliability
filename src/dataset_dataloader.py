import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import pandas as pd 
import cv2 
import numpy as np 
from pathlib import Path
from physics_utils import *
from PIL import Image

class FftTransform(object):
    def __init__(self, width=5, notch_depth=0.95,apply_bilateral=False):
        self.width=width
        self.notch_depth=notch_depth
        self.apply_bilateral=apply_bilateral
    
    def __call__(self,img):
        img = np.array(img)
        dft_shift,spectrum = compute_fft_spectrum(img)
        filtered_dft, mask = apply_gaussian_notch_filter(dft_shift,width=self.width,notch_depth=self.notch_depth)
        cleaned_img = reconstruct_image(filtered_dft,apply_bilateral=self.apply_bilateral)
        return Image.fromarray(cleaned_img)
 
class DatasetMaker(Dataset):
    def __init__(self,data_csv_path,transforms=None,project_root=None):
        super().__init__()
        self.transform=transforms
        if project_root is None:
            project_root = Path.cwd()
        self.project_root = Path(project_root)
        self.path = self.project_root / data_csv_path
        self.data=pd.read_csv(self.path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.project_root / 'data' / self.data['path'][idx]
        img=Image.open(img_path).convert('L')
        y=self.data['label'][idx]
        if self.transform:
            img=self.transform(img)
        return img , y
    
