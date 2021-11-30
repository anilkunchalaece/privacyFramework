# This script used for the following
# 1. Extract the best matched Ground truth BBOX from Predicted bbox
#    a. IoU ( Intersection over union ) is used to match bboxes
# 2. 

import pandas as pd
import torchvision
import torch
import os
import json
from multiprocessing import Pool
import cv2


def loadBboxFile(fName,t,cols) :
    # cols -> cols to extract for given dataset
    df = pd.read_csv(fName,sep=" ",header=None)
    if t == "gt" :
        df.columns = ["class","scores","x1","y1","x2","y2"]
    elif t == "pred" :
        df.columns = ["class","scores","x1","y1","x2","y2"]
    return df[cols]

def getMatchingBboxes(fName,bboxDir="bbox",dirToSave=None):
    # fName = "000001.txt"
    cols = ["scores","x1","y1","x2","y2"]
    gt = loadBboxFile(os.path.join("groundtruths",fName),"gt",cols)
    pred = loadBboxFile(os.path.join("detections",fName),"pred",cols)

    # print(gt)
    gt = gt.drop(gt[gt["scores"] <  0.75].index)
    gt = gt[cols[1:]]
    pred = pred[cols[1:]]

    gt = torch.tensor(gt.values)
    pred = torch.tensor(pred.values)
    # print(F"gt =>{gt.shape} , pred => {pred.shape}")

    iou_values = torchvision.ops.box_iou(gt,pred)
    iou_thres = iou_values >= 0.5
    indices = torch.nonzero(iou_thres)
    # iou_thres.nonzero()
    out = {
        "fileName" : fName,
        "gt_src" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled",
        "pred_src" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled_output",
        "bboxDir" : bboxDir,
        "matches" : []
    }

    for i in indices :
        # print(gt[i.numpy()[0]])
        # print(pred[i.numpy()[1]])
        # print("-----")
        out["matches"].append({
            "gt" : [int(x) for x in gt[i.numpy()[0]].tolist()],
            "pred" : [ int(x) for x in pred[i.numpy()[1]].tolist()]
        })

    # print(indices)
    if dirToSave != None :
        with open(os.path.join(dirToSave,fName.replace(".txt", ".json")),"w") as fd:
            json.dump(out,fd)
    return out

def cropImage(bData):
    gt_img = cv2.imread(os.path.join(bData["gt_src"],bData["fileName"].replace(".txt", ".png")))
    pred_img = cv2.imread(os.path.join(bData["pred_src"],bData["fileName"].replace(".txt", ".png")))

    gt_t_path = os.path.join(bData["bboxDir"],"gt",F'{bData["fileName"].split(".")[0]}')
    pred_t_path = os.path.join(bData["bboxDir"],"pred",F'{bData["fileName"].split(".")[0]}')

    try :
        os.mkdir(gt_t_path)
        os.mkdir(pred_t_path)
    except :
        pass

    for i,x in enumerate(bData["matches"]):

        # print(F"{i} => {x}")
        gt_c = gt_img[x["gt"][1]:x["gt"][3],x["gt"][0]:x["gt"][2]]
        pred_c = pred_img[x["pred"][1]:x["pred"][3], x["pred"][0]:x["pred"][2]]


        # write cropped image to file
        cv2.imwrite(os.path.join(gt_t_path,F'{i}.png'),gt_c)
        cv2.imwrite(os.path.join(pred_t_path,F'{i}.png'),pred_c)

def drawbbox(frame, bbox):
    for box in bbox :
        cv2.rectangle(frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 0, 255),
                    thickness=2)
    return frame
    

def cropBboxImages():
    fName = "000001.txt"
    bboxDir = "bbox"

    try :
        # create dirs to save cropped images
        os.makedirs(os.path.join(bboxDir,"gt"))
        os.makedirs(os.path.join(bboxDir,"pred"))
    except :
        print("bbox dir already exists")
        
    bboxData = getMatchingBboxes(fName,bboxDir)

    # used to plot bbox in image
    # imgEx = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled/000001.png"
    # print([ x["gt"] for x in bboxData["matches"]])
    # out_img = drawbbox(cv2.imread(imgEx), [ x["gt"] for x in bboxData["matches"]])
    # cv2.imwrite("out_img.png", out_img)
    # cropImage(bboxData)
    
    with Pool() as pool:
        pool.map(cropImage,[getMatchingBboxes(x) for x in os.listdir("groundtruths")])




if __name__ == "__main__" :
    # fName = "groundtruths/000000.txt"
    # print(loadBboxFile(fName))
    dirToSave = "matchingBboxes"
    bboxDir = "bbox"
    # getMatchingBboxes(dirToSave)
    cropBboxImages()