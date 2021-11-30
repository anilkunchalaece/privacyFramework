# Last modified : 23 Nov 2021
# This script uses keypoint-RCNN to detect the keypoints in the given image

import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import transforms as transforms
from PIL import Image
from maskDataset import MaskDataset
from torch.utils.data import DataLoader
from config import getConfig as config
import os
import pandas as pd
import shutil



class PersonDetector:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config()["eval"]
        self.batchSize = self.config["batch_size"]
        # get keypoint rcnn from torch pretrained models
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detector = self.detector.to(self.device)

        # set the model into evaluation
        self.detector = self.detector.eval()
        
        self.transform = transforms.Compose([transforms.ToTensor()])

    
    def detectPerson(self,imgs):
        # print(imgs.size)
        imgs = self.transform(imgs)
        imgs = imgs.to(self.device)
        # print(imgs.size)
        imgs = torch.unsqueeze(imgs, 0)
        # print(imgs.shape)
        pred = self.detector(imgs)
        print(len(pred[0]['boxes']))
        return pred

    def generateDetections(self,imageDir,dirToSave):
        dataset = MaskDataset(imageDir, self.transform)
        dataLoader = DataLoader(dataset,batch_size=self.batchSize,shuffle=False)

        # out_names = []
        # out_bbox = []
        out_list = []

        for idx, data in enumerate(dataLoader) :
            with torch.no_grad():
                # print(data["fileName"])
                img_names = data["fileName"]
                data = data["value"].to(self.device)
                out = self.detector(data)
                out_list = []
                for idx, eachDetection in enumerate(out) :
                    fName = img_names[idx]

                    # check scores with >= score_threshold
                    scoreIdxs = np.argwhere(eachDetection["scores"].detach().cpu().numpy() >= self.config["score_threshold"]).squeeze()
                    # print(F"Found {scoreIdxs.shape} scores above threshold {self.config['score_threshold']}")

                    # based on threshold_score_idxs get label indexes which are only human
                    labelIdxs = np.argwhere(eachDetection["labels"][scoreIdxs].detach().cpu().numpy() == 1).squeeze() # label "1" is person TODO -> need to improve this part of code
                    # print(F"Found {labelIdxs.shape} person lables above threshold")

                    boxes = eachDetection["boxes"][labelIdxs].detach().cpu().numpy().squeeze().tolist()
                    scores = eachDetection["scores"][labelIdxs].detach().cpu().numpy().squeeze().tolist()

                    # print(boxes)

                    i_final = []
                    for i,b in enumerate(boxes) :
                        x = []
                        x.extend([fName,"person"])
                        x.extend(b)
                        x.append(scores[i])

                        i_final.append(x)

                        with open(F"{dirToSave}/{os.path.basename(fName).replace('.png','.txt')}",'a') as fd:
                            fd.write("person ")
                            fd.write(str(scores[i]))
                            fd.write(" ")
                            fd.write(" ".join([str(x) for x in b]))
                            fd.write("\n")
                    
                    # print(len(i_final))
                    # print(i_final[0])
                    
                    out_list.extend(i_final)
                # break
        # print(out_list[0])
        out_pd = pd.DataFrame(out_list,columns=['image','class_label','x_top_left','y_top_left','x_bottom_right','y_bottom_right','score'])
        # print(out_pd)
        out_pd.to_pickle("orig_images_scaled_output.pkl")

if __name__ == "__main__" :
    # import sys
    # imgName = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled/000000.png"
    # p = PersonDetector()
    # img = Image.open(imgName)
    # img = img.convert("RGB")
    # print(img.mode)
    # ps = p.detectPerson(img)
    # sys.exit()
    # im = cv2.imread(imgName)
    p = PersonDetector()
    imageDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled_output"
    dirToSave = "out"

    # c = [
        
    #     {
    #         "imageDir" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled_output",
    #         "dirToSave" : "detections"
    #     },
    #     {
    #         "imageDir" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled",
    #         "dirToSave" : "groundtruths"
    #     }
    # ]
    c = []
    srcDir = "annomizedImgs"
    for dirName in os.listdir(srcDir) :
        c.append({
            "imageDir" : os.path.join(srcDir,dirName),
            "dirToSave" : os.path.join(srcDir,F"{dirName}_detections")
        })

    # print(c)

    for x in c :
        if os.path.isdir(x["dirToSave"]) :
            shutil.rmtree(x["dirToSave"])
        os.mkdir(x["dirToSave"])
        print(F"generating {x['dirToSave']}")
        print(x["imageDir"], x["dirToSave"])
        try :
            p.generateDetections(x["imageDir"], x["dirToSave"])    
        except :
            print(F"unable to generate detection for {x['imageDir']}")
