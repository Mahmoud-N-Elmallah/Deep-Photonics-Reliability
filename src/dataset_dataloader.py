import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import pandas as pd 
import cv2 
import numpy as np 
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
    def __init__(self,data_csv_path,transforms=None):
        super().__init__()
        self.transform=transforms
        self.path=data_csv_path
        self.data=pd.read_csv(self.path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img=Image.open(f'../data/{self.data['path'][idx]}').convert('L')
        y=self.data['label'][idx]
        if self.transform:
            img=self.transform(img)
        return img , y
    
temp_train_ds=DatasetMaker('../data/train_data.csv',transforms=transforms.Compose([FftTransform(),transforms.ToTensor()]))

all=[temp_train_ds[i][0] for i in range(len(temp_train_ds))]
a=torch.stack(all,dim=0)
train_mean=a[:,0,:,:].mean()
train_std=a[:,0,:,:].std()  ## add these stats to the config file for further use

config_content = f"""
TRAIN_MEAN = {round(train_mean.item(),3)}
TRAIN_STD = {round(train_std.item(),3)}
NOTCH_WIDTH = 5
NOTCH_DEPTH = 0.95
APPLY_BILATERAL = False
"""

with open('../src/config.py', 'w') as f:
    f.write(config_content.strip())

print("Config saved successfully.")