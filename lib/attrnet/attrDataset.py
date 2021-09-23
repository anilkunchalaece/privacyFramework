from torch.utils.data import Dataset
import torch
import PIL
import os
from torchvision.transforms import transforms

class AttrDataset(Dataset):
    def __init__(self,attr,transform):
        self.attr = attr
        self.transform = transform

    def __len__(self):
        return len(self.attr)
    
    def __getitem__(self, idx) :
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = PIL.Image.open(self.attr[idx][0]).convert('RGB')
        labels = torch.tensor(self.attr[idx][1],dtype=torch.float)

        return{
            "image" : self.transform(img),
            "labels" : labels 
        }


if __name__ == "__main__" :
    from preprocess import processData

    srcDir = "/home/akunchala/Documents/z_Datasets/RAP"
    data = processData(srcDir)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(degrees=45)])
    aD = AttrDataset(data["dataset"], transform)
    print(aD[0]["image"].shape)
    print(aD[0]["labels"].shape)

