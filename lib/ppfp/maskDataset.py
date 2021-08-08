# Dataset class for inference using pretrained maks-rcnn
# ref - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch.utils.data import Dataset
import torch
from PIL import Image
import os

class MaskDataset(Dataset) :
    def __init__(self,rootDir,transform):
        self.rootDir = rootDir
        self.transform = transform

        self.allImages = os.listdir(self.rootDir)

    def __len__(self):
        return len(self.allImages)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.rootDir,self.allImages[idx])
        img = Image.open(img_name).convert('RGB')
        sample = self.transform(img)
        return {
            "fileName" : img_name, 
            "value" : sample
        }


