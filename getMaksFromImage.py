"""
Date : 3 Aug 2021
Author : Kunchala Anil

This script is used to extrace semantic segmentation masks for persons in given images using pytorch pretrained mask-rcnn using resnet-50-fcn network
ref - https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
"""

import torch
import torchvision
import cv2
import argparse

from PIL import Image
from torchvision.transforms import transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
import numpy as np
import json,os,sys

# my libs
from lib.ppfp.config import getConfig as config
from lib.ppfp.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from lib.ppfp.maskDataset import MaskDataset


class GetMask():

    def __init__(self):

        #Config
        self.config = config()["ppfp"]
        self.batchSize = self.config["batch_size"]
        self.score_threshold = self.config["score_threshold"]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(F"running model using {self.device} ")

        # define the pre-trained maskrcnn model from pytorch
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,progress=True)
        self.model = self.model.to(self.device) # send it to gpu

        # we are currenlty interested in inference only , so set the model to eval 
        self.model.eval()

        self.transform = transforms.Compose([transforms.ToTensor()])

    def runPretrainedModel(self,imageDir,dirToSave):
        dataset = MaskDataset(imageDir, self.transform)
        dataLoader = DataLoader(dataset,batch_size=self.batchSize,shuffle=False)
        outputs = {
            "scores" : [], "masks" : [],"labels" : [], "boxes" : [], "fileNames" : []
        }

        for idx, data in enumerate(dataLoader) :
            with torch.no_grad():
                # print(data["fileName"])
                img_name = data["fileName"]
                data = data["value"].to(self.device)
                out = self.model(data)
                # print(f"data {data.shape} , out {len(out)}")
                # get all scores, labels, masks and bboxes 
                for k in outputs.keys():
                    _vals = []
                    for i, x in enumerate(out) :
                        if k == "masks" :
                            # convert masks into binary
                            outputs[k].append((x[k] >= 0.5).detach().cpu().numpy())
                        elif k == "fileNames":
                            outputs[k].append(img_name[i])
                        else:
                            outputs[k].append(x[k].detach().cpu().numpy())
        return outputs

    def generateMasks(self,imageDir,dirToSave) :

        # get object detection results for images using mask-rcnn 
        detectionResults = self.runPretrainedModel(imageDir,dirToSave)
        colors = self.generateRandomColors()
        
        for idx in range(len(detectionResults["fileNames"])) :
            fName = detectionResults["fileNames"][idx]
            
            # check scores with >= score_threshold
            scoreIdxs = np.argwhere(detectionResults["scores"][idx] >= self.config["score_threshold"]).squeeze()
            # print(F"Found {scoreIdxs.shape} scores above threshold {self.config['score_threshold']}")

            # based on threshold_score_idxs get label indexes which are only human
            labelIdxs = np.argwhere(detectionResults["labels"][idx][scoreIdxs] == 1).squeeze() # label "1" is person TODO -> need to improve this part of code
            # print(F"Found {labelIdxs.shape} person lables above threshold")

            #getMasksbased on labelIdx
            masks = detectionResults["masks"][idx][labelIdxs].squeeze()
            #create a tensor with mask and convert it to bool -> suitable for draw_segementation_masks
            masks = torch.from_numpy(masks).type(torch.bool)

            orgImg = np.array(Image.open(fName)).transpose(2,0,1)
            imgEmpty = torch.zeros(orgImg.shape,dtype=torch.uint8) # may be need to replace with 1's to only masks with white background TODO -> need to check this again

            imgWithMasks = draw_segmentation_masks(imgEmpty, masks,colors=colors).numpy().transpose(1,2,0)
            
            outFName = os.path.join(dirToSave,os.path.basename(fName))
            cv2.imwrite(outFName, imgWithMasks)
            # print(F"img with mask saved to {outFName}")

    def generateRandomColors(self,maxColors=25) :
        return [tuple(val) for val in np.random.choice(range(256),size=(maxColors,3))]

if __name__ == "__main__" :
    m = GetMask()
    d_out = m.generateMasks("school_girls_org","out")