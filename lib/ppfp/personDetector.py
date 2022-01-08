# Last modified : 23 Nov 2021
# This script uses keypoint-RCNN to detect the keypoints in the given image

import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import transforms as transforms
from PIL import Image
from lib.ppfp.maskDataset import MaskDataset
from torch.utils.data import DataLoader
from lib.ppfp.config import getConfig as config
import os,json
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
                    #if there are no detections, skip the loop
                    if len(boxes) == 0 :
                        continue
                    
                    # if we detect only one person, convert it into list
                    if type(boxes[0]) != list :
                        boxes = [boxes]
                        scores = [scores]
                    

                    i_final = []
                    for i,b in enumerate(boxes) :
                        
                        # print(b)
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

    def generateDetections2(self,imageDir,dirToSave,fToSave):
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
                for idx, d in enumerate(out) :
                    im_name = img_names[idx]
                    boxes = torchvision.ops.box_convert(d["boxes"], "xyxy", "xywh")
                    areas = torchvision.ops.box_area(d["boxes"])

                    for i in range(d["boxes"].shape[0]) :
                        if d["scores"][i].item() >= 0.5 and d["labels"][i].item() == 1:
                            _kp = {
                                "image_id" : int(os.path.basename(im_name).split(".")[0]),
                                "category_id" : d["labels"][i].item(),
                                # "keypoints" : d["keypoints"][i,:,:].reshape(1,-1).tolist()[0],
                                "score" : d["scores"][i].item(),
                                "bbox" : boxes[i].tolist(),
                                "area" : areas[i].item()
                            }
                            out_list.append(_kp)

        outFilePath = os.path.join(dirToSave,fToSave)

        with open(outFilePath,'w') as fd :
            json.dump(out_list,fd)
        
        del self.detector
        torch.cuda.empty_cache()
        
        print(F"Bboxes are saved to {outFilePath}")
        return outFilePath

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
    #     {'imageDir': 'annomizedImgs/pixelate_body', 
    #     'dirToSave': 'annomizedImgs/pixelate_body_detections'
    #     },
    #     # {
    #     # 'imageDir': 'annomizedImgs/dct_body_detections',
    #     # 'dirToSave': 'annomizedImgs/dct_body_detections_detections'
    #     # }
    # ]
    c = []
    srcDir = "annomizedImgs"
    for dirName in os.listdir(srcDir) :
        c.append({
            "imageDir" : os.path.join(srcDir,dirName),
            "dirToSave" : os.path.join(srcDir,F"{dirName}_detections")
        })

    print(c)

    for x in c :
        if os.path.isdir(x["dirToSave"]) :
            shutil.rmtree(x["dirToSave"])
        os.mkdir(x["dirToSave"])
        print(F"generating {x['dirToSave']}")
        # print(x["imageDir"], x["dirToSave"])
        # try :
        p.generateDetections(x["imageDir"], x["dirToSave"])    
        # except :
            # print(F"unable to generate detection for {x['imageDir']}")
