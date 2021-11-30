# Last modified : 17 Oct 2021W
# This script uses keypoint-RCNN to detect the keypoints in the given image

import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import transforms as transforms
from PIL import Image


class KeyPointDetector:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get keypoint rcnn from torch pretrained models
        self.detector = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        # set the model into evaluation
        self.detector = self.detector.eval()
        
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def detectKeyPoints(self,imgs):
        print(imgs.size)
        imgs = self.transform(imgs)
        print(imgs.size)
        imgs = torch.unsqueeze(imgs, 0)
        print(imgs.shape)
        pred = self.detector(imgs)
        print(pred[0]["keypoints"].shape)
        return pred[0]["keypoints"]




if __name__ == "__main__":
    imgName = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled_output/000000.png"
    kd = KeyPointDetector()
    img = Image.open(imgName)
    img = img.convert("RGB")
    print(img.mode)
    kp = kd.detectKeyPoints(img)
    im = cv2.imread(imgName)
    print(cv2.KeyPoint_convert(kp[0][:,:-1].tolist()))
    print(kp[0].tolist())
    cv2.drawKeypoints(im, cv2.KeyPoint_convert(kp[0][:,:-1].tolist()), im,color=(255,0,0))
    cv2.imwrite("kp_pe.png",im)
    