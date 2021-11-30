## This function is used to extract all faces and save it in a files
# later these faces  i.e face bboxes are used to pixelate, blur and mask the faces in order to annonomize the dataset

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os,json
from multiprocessing import Pool


class InsightFaceDetector:

    def __init__(self,srcDir,outDir="faces"):
        self.app = FaceAnalysis(allowed_modules=["detection"])
        self.app.prepare(ctx_id=0,det_thresh=0.25)
        self.srcDir = srcDir
        self.outDir = outDir

        # create a outDir if not exist
        if not os.path.isdir(self.outDir):
            os.mkdir(self.outDir)
    
    # imPath -> image path relative to srcDir
    def detectFace(self,imPath) :
        imgPath = os.path.join(self.srcDir,imPath)
        im = cv2.imread(imgPath)

        # detect faces in image
        faces = self.app.get(im)
        
        out = {
            "fileName" : imgPath,
            "facesBbox" : []
        }

        for f in faces :
            out["facesBbox"].append(f["bbox"].tolist())

        fOut = os.path.join(self.outDir,imPath.replace(".png",".json"))
        with open(fOut,'w') as fd:
            json.dump(out,fd)
    
    def detectAllFacesInDir(self):
        # with Pool() as pool:
        #     pool.map(self.detectFace,os.listdir(self.srcDir))
        for fName in os.listdir(self.srcDir) :
            self.detectFace(fName)





if __name__ == "__main__":
    srcDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled"
    outDir = "faces"
    fName = "000001.png"

    fd = InsightFaceDetector(srcDir)
    # fd.detectFace(fName)
    fd.detectAllFacesInDir()




# app = FaceAnalysis()
# app.prepare(ctx_id=0,det_thresh=0.75)
# # img = ins_get_image(testImgPath)
# img = cv2.imread(testImgPath)
# faces = app.get(img)

# print(faces[0]["embedding"].shape)
# o_img = fd._draw(img,[faces[0]["bbox"]],None)

# cv2.imwrite("out.png",o_img)

# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)
