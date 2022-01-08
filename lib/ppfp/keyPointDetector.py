# Last modified : 17 Oct 2021W
# This script uses keypoint-RCNN to detect the keypoints in the given image

import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import transforms as transforms
import torchvision
from PIL import Image
from lib.ppfp.config import getConfig as config
import os,json

from lib.ppfp.maskDataset import MaskDataset
from torch.utils.data import DataLoader


class KeyPointDetector:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get keypoint rcnn from torch pretrained models
        self.detector = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.detector = self.detector.to(self.device)
        # set the model into evaluation
        self.detector = self.detector.eval()
        self.config = config()["eval"]
        self.batchSize = self.config["batch_size"]
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def detectKeyPoints(self,imgs):
        print(imgs.size)
        imgs = self.transform(imgs)
        print(imgs.size)
        imgs = torch.unsqueeze(imgs, 0)
        print(imgs.shape)
        pred = self.detector(imgs)
        print(pred[0]["keypoints"].shape)
        print(pred[0]["labels"])
        print(pred[0]["scores"])
        return pred[0]["keypoints"]
    
    def generateDetections(self,imageDir,outDir,fileName):
        dataset = MaskDataset(imageDir, self.transform)
        dataLoader = DataLoader(dataset,batch_size=self.batchSize,shuffle=False)
        keyPointsOut = []
        for idx, data in enumerate(dataLoader) :
            with torch.no_grad():
                img_names = data["fileName"]
                data = data["value"].to(self.device)
                out = self.detector(data)
                for idx,d in enumerate(out) :
                    # print(img_names[idx])
                    # print(len(d["keypoints"].tolist()))
                    # print(d.keys())
                    im_name = img_names[idx]
                    boxes = torchvision.ops.box_convert(d["boxes"], "xyxy", "xywh")
                    areas = torchvision.ops.box_area(d["boxes"])
                    # print(d["keypoints"].shape)
                    # print(d["keypoints"].shape[0])
                    
                    for i in range(d["keypoints"].shape[0]) :
                        if d["scores"][i].item() >= 0.5 :
                            _kp = {
                                "image_id" : int(os.path.basename(im_name).split(".")[0]),
                                "category_id" : d["labels"][i].item(),
                                "keypoints" : d["keypoints"][i,:,:].reshape(1,-1).tolist()[0],
                                "score" : d["scores"][i].item(),
                                "bbox" : boxes[i].tolist(),
                                "area" : areas[i].item()
                            }
                            keyPointsOut.append(_kp)
                        # print(_kp)

        outFilePath = os.path.join(outDir,fileName)

        with open(outFilePath,'w') as fd :
            json.dump(keyPointsOut,fd)
        
        print(F"Keypoints are saved to {outFilePath}")
        del self.detector
        torch.cuda.empty_cache()
        return outFilePath
                    
                
                


if __name__ == "__main__":
    dirName = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled_output/"
    # dirName = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annomizedImgs/blur_body"
    outDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/keyPoints"
    conf = {
        "dirName" : dirName,
        "outDir" : outDir,
        "fileName" : "pred_keypoints.json"
    }

    # fileName = "000000.png"
    # imgName = F"{dirName}/000000.png"
    kd = KeyPointDetector()
    # img = Image.open(imgName)
    # img = img.convert("RGB")
    # # print(img.mode)
    # kp = kd.detectKeyPoints(img)
    # print(F"shape {kp.shape}")
    # im = cv2.imread(imgName)
    # # print(cv2.KeyPoint_convert(kp[0][:,:-1].tolist()))
    # # print(kp[0].tolist())
    # cv2.drawKeypoints(im, cv2.KeyPoint_convert(kp[0][:,:-1].tolist()), im,color=(255,0,0))
    # cv2.imwrite("kp_pe.png",im)
    kd.generateDetections(dirName,outDir,"pred_keypoints.json")


    