"""
This script is used to train attributeNet implementation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from lib.attrnet.attrDataset import AttrDataset
from lib.attrnet.attributeNet import AttributeNet
from lib.attrnet.preprocess import loadDataFromFile,ATTR_OF_INTREST_IDX

import matplotlib.pyplot as plt
import numpy as np
import json
import sys,os
import pickle

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.manual_seed(42)
print(F"running with {device}")

img_width = 40
img_height = 120
batchSize = 100
N_EPOCH = 15

def train():
    srcDir = "/home/akunchala/Documents/z_Datasets/RAP"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((img_height,img_width)),# height,width
                                    transforms.RandomRotation(degrees=45)])

    # _d = processData(srcDir)
    # data = _d["dataset"][:400]
    # labelNames = _d["label_names"]
    # print(F"Original dataset length is {len(data)}")

    # train, valid = train_test_split(data,shuffle=True)
    dataFileName = os.path.join(srcDir,"dataset_modified.pkl")
    _d = loadDataFromFile(dataFileName)
    train_dataset = AttrDataset(_d["train"], transform)
    valid_dataset = AttrDataset(_d["valid"], transform)

    train_dataloader = DataLoader(train_dataset,batch_size=batchSize,shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batchSize,shuffle=True,drop_last=True)
    
    model = AttributeNet()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr = 0.0001 )
    criterion = nn.BCEWithLogitsLoss()

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
            img = data["image"].to(device)
            labels = data["labels"].to(device)

            opt.zero_grad()
            pred_labels = model(img)

            loss = criterion(pred_labels,labels)
            loss.backward()
            opt.step()
            tl.append(loss.cpu().detach().item())
        
        # validation
        model.eval()
        for idx, data in enumerate(valid_dataloader):
            img_v = data["image"].to(device)
            labels_v = data["labels"].to(device)

            pred_label_v = model(img_v)
            loss = criterion(pred_label_v,labels_v)
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
            torch.save(model.state_dict(),"models/attrnet.pth")

        else :
            if _vl - validLossPrev >= 0.0001 : # min delta
                badEpoch = badEpoch + 1

            if badEpoch >= patience :
                print(F"Training stopped early due to overfitting in epoch {epoch}")
                break
        validLossPrev = _vl # store current valid loss

    # dump losses into file
    lossFileName = "attrnet_losses.json"
    with open(lossFileName,"w") as fd:
        json.dump(lossDict,fd)
    print(F"Training and validation losses are saved in {lossFileName}")



def eval():
    # This function will run inference on the test set and saves the orig and pred labels into file 
    srcDir = "/home/akunchala/Documents/z_Datasets/RAP"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((img_height,img_width))
                                    ])
    dataFileName = os.path.join(srcDir,"dataset_modified.pkl")
    _d = loadDataFromFile(dataFileName)
    test = _d["test"]
    labelNames = _d["label_names"]

    testDataset = AttrDataset(test, transform)
    testDataLoader = DataLoader(testDataset,batch_size=batchSize,shuffle=True,drop_last=True)

    model = AttributeNet()
    model.load_state_dict(torch.load('models/attrnet.pth'))
    model = model.to(device)
    model.eval()

    results = {
        "orig" : np.empty(shape=(0,len(labelNames))),
        "pred" : np.empty(shape=(0,len(labelNames))),
        "label_names" : labelNames
    }

    for idx, data in enumerate(testDataLoader):
        img = data["image"].to(device)
        labels = data["labels"].to(device)
        pred_labels = model(img)
        pred_labels = torch.sigmoid(pred_labels)

        labels = labels.detach().cpu().numpy()
        pred_labels = pred_labels.detach().cpu().numpy()
        pred_labels[pred_labels >= 0.50] = 1
        pred_labels[pred_labels < 0.50] = 0

        results["orig"] = np.append(results["orig"],labels,axis=0)
        results["pred"] = np.append(results["pred"],pred_labels,axis=0)
        # break
    # print(results["pred"].shape)
    fileNameToSave = os.path.join(srcDir,os.path.basename(srcDir)+"_eval_results.pkl")
    with open(fileNameToSave,'wb') as fd:
        pickle.dump(results,fd)

def calculateF1Scores(fileName):
    data = loadDataFromFile(fileName)
    attr_idxs = [5,15,21,29,33,34] # staring ids for all attributes to split labels into subgroups
    f1score = f1_score(data["orig"],data["pred"],average=None)
    acc_score = accuracy_score(data["orig"],data["pred"])
    print(f1score)
    print(acc_score)



def plotLoss():
    fileName = "attrnet_losses.json"
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
    # plotLoss()
    # eval()
    # fileName = "/home/akunchala/Documents/z_Datasets/RAP/RAP_eval_results.pkl"
    # calculateF1Scores(fileName)