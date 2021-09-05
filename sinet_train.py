"""
last updated : 5 Sep 2021
Author : Kunchala Anil
Email : anilkunchalaece@gmail.com

This script is used to train the sinet implementation with resnet(18) backbone
"""
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
import json

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.manual_seed(42)
print(F"running with {device}")

def train():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(degrees=45)])
    root_dir = "/home/akunchala/Downloads/MARS_Dataset/bbox_train"
    triplets = ImagePairGen(root_dir,limit_ids=250 ,max_frames=None)
    train, valid = train_test_split(triplets.generateTripletsRandomly(),shuffle=True)

    train_dataset = TripletDataset(train, transform)
    valid_dataset = TripletDataset(valid, transform)

    train_dataloader = DataLoader(train_dataset,batch_size=100,shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=100,shuffle=True,drop_last=True)

    model = SimilarityNet()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr = 0.00001 )
    criterion = nn.TripletMarginLoss(margin=0.1)

    # earlyStopping params
    patience = 10 # wait for this many epochs before stopping the training
    validLossPrev = float("inf") #used for early stopping
    badEpoch = 0

    # dict to store losses
    lossDict = {
            "train" : [],
            "valid" : []
        }

    for epoch in range(0,100) :
        
        tl = []
        vl = []
        
        # training
        model.train()
        for idx, data in enumerate(train_dataloader):
            anchorImgs = data["anchorImg"].to(device)
            positiveImgs = data["positiveImg"].to(device)
            negativeImgs = data["negativeImg"].to(device)

            opt.zero_grad()
            a_f,p_f,n_f = model(anchorImgs,positiveImgs,negativeImgs)
            loss = criterion(a_f,p_f,n_f)
            loss.backward()
            opt.step()
            tl.append(loss.cpu().detach().item())
        
        # validation
        model.eval()
        for idx, data in enumerate(valid_dataloader):
            anchorImgs = data["anchorImg"].to(device)
            positiveImgs = data["positiveImg"].to(device)
            negativeImgs = data["negativeImg"].to(device)
            a_f,p_f,n_f = model(anchorImgs,positiveImgs,negativeImgs)
            loss = criterion(a_f,p_f,n_f)
            vl.append(loss.cpu().detach().item())

        # collect losses
        _tl, _vl = np.mean(tl) , np.mean(vl)
        lossDict["train"].append(_tl)
        lossDict["valid"].append(_vl)
        print(F"epoch:{epoch}, tl:{_tl}, vl:{_vl}")

        # earlyStopping
        if _vl < validLossPrev : # if there is a decrease in validLoss all is well 
            badEpoch = 0 # reset bad epochs

            #save model
            torch.save(model.state_dict(),"models/sinet.pth")

        else :
            if _vl - validLossPrev >= 0.0001 : # min delta
                badEpoch = badEpoch + 1

            if badEpoch >= patience :
                print(F"Training stopped early due to overfitting in epoch {epoch}")
                break
        validLossPrev = _vl # store current valid loss

    # dump losses into file
    lossFileName = "sinet_losses.json"
    with open(lossFileName,"w") as fd:
        json.dump(lossDict,fd)
    print(F"Training and validation losses are saved in {lossFileName}")


def plotLoss():
    fileName = "sinet_losses.json"
    try :
        with open(fileName) as fd:
            d = json.load(fd)
            plt.plot(d["train"],label="Training")
            plt.plot(d["valid"],label="Validation")
            plt.title("SiNet Training loss")
            plt.legend()
            plt.show()
    except Exception as e :
        print(F"unable to open {fileName} with exception {str(e)}")

if __name__ == "__main__" :
    train()
    plotLoss()