import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from sklearn.model_selection import train_test_split

from lib.sinet.tripletDataset import TripletDataset
from lib.sinet.similarityNet import SimilarityNet
from lib.sinet.image_pairs import ImagePairGen

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.manual_seed(42)
print(F"running with {device}")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomRotation(degrees=45)])
root_dir = "/home/akunchala/Downloads/MARS_Dataset/bbox_train"
triplets = ImagePairGen(root_dir,limit_ids=20 ,max_frames=None)
dataset = TripletDataset(triplets.generateTripletsRandomly(), transform)

train, test = train_test_split(dataset,shuffle=True)
train_dataloader = DataLoader(train,batch_size=10,shuffle=True)

model = SimilarityNet()
model = model.to(device)
opt = torch.optim.Adam(model.parameters(),lr = 0.00001 )
criterion = nn.TripletMarginLoss(margin=0.1)
model.train()

for epoch in range(0,200) :
    losses = []
    total_loss = 0
    for idx, data in enumerate(train_dataloader):
        anchorImgs = data["anchorImg"].to(device)
        positiveImgs = data["positiveImg"].to(device)
        negativeImgs = data["negativeImg"].to(device)

        opt.zero_grad()
        a_f,p_f,n_f = model(anchorImgs,positiveImgs,negativeImgs)
        loss = criterion(a_f,p_f,n_f)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(np.mean(losses))




