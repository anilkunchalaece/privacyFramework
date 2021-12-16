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
import torchvision

from sklearn.model_selection import train_test_split

from lib.sinet.tripletDataset import TripletDataset, PairDataset
from lib.sinet.similarityNet import SimilarityNet
from lib.sinet.image_pairs import ImagePairGen

import matplotlib.pyplot as plt
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.manual_seed(42)
print(F"running with {device}")

img_width = 40
img_height = 120
batchSize = 100
N_EPOCH = 100

def train():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((img_height,img_width)),# height,width
                                    transforms.RandomRotation(degrees=45)])
    root_dir = "/home/akunchala/Documents/z_Datasets/MARS_Dataset/bbox_train"
    t = ImagePairGen(root_dir,limit_ids=100 ,max_frames=None)
    triplets = t.generateTripletsRandomly()
    train, valid = train_test_split(triplets,shuffle=True)

    train_dataset = TripletDataset(train, transform)
    valid_dataset = TripletDataset(valid, transform)

    train_dataloader = DataLoader(train_dataset,batch_size=batchSize,shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batchSize,shuffle=True,drop_last=True)

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

    for epoch in range(0,N_EPOCH) :
        
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


def eval():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((img_height,img_width))# height,width
                                    ])
    root_dir = "/home/akunchala/Documents/z_Datasets/MARS_Dataset/bbox_test"
    t = ImagePairGen(root_dir,limit_ids=50 ,max_frames=None)
    triplets = t.generateTripletsRandomly()
    # print(triplets[:2])
    

    triplet_dataset = TripletDataset(triplets, transform)

    triplet_dataloader = DataLoader(triplet_dataset,batch_size=batchSize,shuffle=True,drop_last=True)
    
    model = SimilarityNet()
    model.load_state_dict(torch.load('models/sinet.pth'))
    model = model.to(device)
    model.eval()

    for idx, data in enumerate(triplet_dataloader):
        anchorImgs = data["anchorImg"].to(device)
        positiveImgs = data["positiveImg"].to(device)
        negativeImgs = data["negativeImg"].to(device)
        a_f,p_f,n_f = model(anchorImgs,positiveImgs,negativeImgs)
        # print(a_f.shape)

        ap_si = nn.functional.cosine_similarity(a_f, p_f,dim=1)
        an_si = nn.functional.cosine_similarity(a_f, n_f,dim=1)
        ap_si = ap_si.detach().cpu().numpy()
        an_si = an_si.detach().cpu().numpy()
        # print(ap_si)
        print(F" ap_si : {np.mean(ap_si)} an_si : {np.mean(an_si)}")
        # print(an_si)
        # print(F"an_si : {np.mean(an_si)}")
        break

def evalResults(gtDir,predDir):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((img_height,img_width))# height,width
                                    ])

    p = ImagePairGen(gtDir)
    dirDict = {
        "gt" : gtDir,
        "pred" : predDir #"/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/blur_body"
    }

    pairs = p.generatePairsForEval(dirDict)

    pairs_dataset = PairDataset(pairs, transform)

    pairs_dataloader = DataLoader(pairs_dataset,batch_size=batchSize,shuffle=True,drop_last=True)
    
    model = SimilarityNet()
    model.load_state_dict(torch.load('/home/akunchala/Documents/PhDStuff/PrivacyFramework/models/sinet.pth'))
    model = model.to(device)
    model.eval()

    ap_si_total = []

    for idx, data in enumerate(pairs_dataloader):
        anchorImgs = data["anchorImg"].to(device)
        positiveImgs = data["positiveImg"].to(device)
        a_f,p_f = model(anchorImgs,positiveImgs)
        # print(a_f.shape)

        ap_si = nn.functional.cosine_similarity(a_f, p_f,dim=1)
        ap_si = ap_si.detach().cpu().numpy().tolist()
        # print(F" ap_si : {np.mean(ap_si)}")
        # ap_si_total = np.append(ap_si_total,ap_si,axis=0)
        ap_si_total.extend(ap_si)
        # print(an_si)
        # print(F"an_si : {np.mean(an_si)}")
    # print(len(ap_si_total))
    # print(np.mean(np.array(ap_si_total)))
    return np.mean(np.array(ap_si_total))




if __name__ == "__main__" :
    # train()
    # plotLoss()
    # eval()
    evalResults()
