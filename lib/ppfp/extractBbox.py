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
import matplotlib.pyplot as plt


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

    annonBasePath = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annomizedImgs"
    annonDirs = {
                "blur_body",
                "blur_faces",
                "dct_faces",
                "pixelate_body",
                "pixelate_faces",
    }

    annonDirs = [os.path.join(annonBasePath, d) for d in annonDirs]

    ann_imgs = {}

    for aD in annonDirs :
        ann_imgs[aD] = cv2.imread(os.path.join(aD,bData["fileName"].replace(".txt", ".png")))

    # annon_t_paths = [os.path.join(bData["bboxDir"], d, F'{bData["fileName"].split(".")[0]}') for d in annonDirs]
    annon_t_paths = {}
    
    for aD in annonDirs:
        annon_t_paths[aD] = os.path.join(bData["bboxDir"], os.path.basename(aD), F'{bData["fileName"].split(".")[0]}')

    gt_t_path = os.path.join(bData["bboxDir"],"gt",F'{bData["fileName"].split(".")[0]}')
    pred_t_path = os.path.join(bData["bboxDir"],"pred",F'{bData["fileName"].split(".")[0]}')

    try :
        os.mkdir(gt_t_path)
        os.mkdir(pred_t_path)
        for k in annon_t_paths.keys() :
            os.makedirs(annon_t_paths[k])
    except Exception as e :
        print(e)
        # raise

    for i,x in enumerate(bData["matches"]):


        # print(F'height {x["gt"][3] - x["gt"][1]} width {x["gt"][2] - x["gt"][0]}')
        
        # print(F"{i} => {x}")
        gt_c = gt_img[x["gt"][1]:x["gt"][3],x["gt"][0]:x["gt"][2]]
        pred_c = pred_img[x["pred"][1]:x["pred"][3], x["pred"][0]:x["pred"][2]]

        for k in ann_imgs.keys():
            i_c = ann_imgs[k][x["gt"][1]:x["gt"][3],x["gt"][0]:x["gt"][2]]
            cv2.imwrite(os.path.join(annon_t_paths[k],F'{i}.png'), i_c)

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

class BBoxExtractor:
    
    def __init__(self,gtImgsLocation,predImgsLocation,annonDir,gtDetFile,predDetFile,segmentationLocation,neuralArtLocation,min_size=20):
        self.gtLocation = gtImgsLocation
        self.predLocation = predImgsLocation
        self.segmLocation = segmentationLocation
        self.neuralArtLocation = neuralArtLocation
        self.annonDir = annonDir
        self.gtFile = gtDetFile
        self.predFile = predDetFile
        self.min_size = min_size # images with height and width smaller than this will be removed
        self.matchingBboxes = {} # dict to save output matching bboxes
    
    def getMatchingBboxes(self):
        gtBbox = self.processBBoxFile(self.gtFile)
        predBbox = self.processBBoxFile(self.predFile)
        out = {}

        for k in gtBbox.keys():
            try :
                _gtV = torch.tensor(gtBbox[k])
                _predV = torch.tensor(predBbox[k])

                _gtV = torchvision.ops.box_convert(_gtV,"xywh","xyxy")
                _predV = torchvision.ops.box_convert(_predV,"xywh","xyxy")

                # remove small bboxes
                _gtV = _gtV[torchvision.ops.remove_small_boxes(_gtV,self.min_size),:]
                _predV = _predV[torchvision.ops.remove_small_boxes(_predV,self.min_size),:]

                iou_values = torchvision.ops.box_iou(_gtV,_predV)
                iou_thres = iou_values >= 0.5
                indices = torch.nonzero(iou_thres)

                _gt = []
                _pred = []
                for i in indices :
                    _gt.append([int(x) for x in _gtV[i.numpy()[0]].tolist()])
                    _pred.append([ int(x) for x in _predV[i.numpy()[1]].tolist()])

                out[k] = {
                    "gt" : _gt,
                    "pred" : _pred
                }
            except :
                pass
        return out
    

    def cropBbox(self,k) :
        bboxes = self.matchingBboxes[k]
        
        gt_d = os.path.join(self.annonDir,"gt","bbox",k)
        pred_d = os.path.join(self.annonDir,"pred","bbox",k)
        segm_d = os.path.join(self.annonDir,"segm","bbox",k)
        na_d = os.path.join(self.annonDir,"neuralArt","bbox",k)
                
        gt_img = cv2.imread(os.path.join(self.gtLocation,F"{int(k):06d}.png"))
        pred_img = cv2.imread(os.path.join(self.predLocation,F"{int(k):06d}.png"))
        segm_img = cv2.imread(os.path.join(self.segmLocation,F"{int(k):06d}.png"))
        na_img = cv2.imread(os.path.join(self.neuralArtLocation,F"{int(k):06d}.png"))

        self.createDirIfNotExist(gt_d)
        self.createDirIfNotExist(pred_d)
        self.createDirIfNotExist(segm_d)
        self.createDirIfNotExist(na_d)

        # create dir for blur bboxes
        for d in os.listdir(os.path.join(self.annonDir,"blur")) :
            self.createDirIfNotExist(os.path.join(self.annonDir,"blur",d,"bbox",k))

        # create dir for blur bboxes
        for d in os.listdir(os.path.join(self.annonDir,"pix")) :
            self.createDirIfNotExist(os.path.join(self.annonDir,"pix",d,"bbox",k))
        
        for i,x in enumerate(bboxes["gt"]) :

            gt_c = gt_img[bboxes["gt"][i][1]:bboxes["gt"][i][3],bboxes["gt"][i][0]:bboxes["gt"][i][2]]
            pred_c = pred_img[bboxes["pred"][i][1]:bboxes["pred"][i][3], bboxes["pred"][i][0]:bboxes["pred"][i][2]]
            segm_c = segm_img[bboxes["gt"][i][1]:bboxes["gt"][i][3], bboxes["gt"][i][0]:bboxes["gt"][i][2]]
            na_c = na_img[bboxes["gt"][i][1]:bboxes["gt"][i][3], bboxes["gt"][i][0]:bboxes["gt"][i][2]]

            for d in os.listdir(os.path.join(self.annonDir,"blur")) :
                
                # print(os.path.join(self.annonDir,"blur",d,"blur_body",F"{int(k):06d}.png"))
                b_img = cv2.imread(os.path.join(self.annonDir,"blur",d,"blur_body",F"{int(k):06d}.png"))
                b_img_c = b_img[bboxes["gt"][i][1]:bboxes["gt"][i][3],bboxes["gt"][i][0]:bboxes["gt"][i][2]]
                cv2.imwrite(os.path.join(self.annonDir,"blur",d,"bbox",k,F'{i}.png'),b_img_c)

            for d in os.listdir(os.path.join(self.annonDir,"pix")) :
                b_img = cv2.imread(os.path.join(self.annonDir,"pix",d,"pixelate_body",F"{int(k):06d}.png"))
                b_img_c = b_img[bboxes["gt"][i][1]:bboxes["gt"][i][3],bboxes["gt"][i][0]:bboxes["gt"][i][2]]
                cv2.imwrite(os.path.join(self.annonDir,"pix",d,"bbox",k,F'{i}.png'),b_img_c)



            cv2.imwrite(os.path.join(gt_d,F'{i}.png'),gt_c)
            cv2.imwrite(os.path.join(pred_d,F'{i}.png'),pred_c)
            cv2.imwrite(os.path.join(segm_d,F'{i}.png'),segm_c)
            cv2.imwrite(os.path.join(na_d,F'{i}.png'),na_c)

    def createDirIfNotExist(self,d):
        try :
            if not os.path.isdir(d) :
                try :
                    os.mkdir(d)
                except :
                    os.makedirs(d)
        except :
            pass
    
    def extractAllBboxes(self) :
        self.matchingBboxes = self.getMatchingBboxes()
        # print(self.matchingBboxes)
        # self.cropBbox("1")
        with Pool() as pool:
            pool.map(self.cropBbox, self.matchingBboxes.keys())
    
    def plotSinetValues(self,fileName) :
        y = []
        with open(fileName) as fd :
            data = json.load(fd)

            k = sorted(data, key=data.get)
            for i in k :
                y.append(data[i])
            
            plt.plot(k,y)
            plt.xticks(rotation=90)
            plt.show()

    
    def processBBoxFile(self,fileName) :
        with open(fileName) as fd:
            data = json.load(fd)

            outData = {}
            for d in data :
                k = str(d["image_id"]) # why string ? number are not allowed as dict keys
                if outData.get(k,0) == 0 :
                    outData[k] = []
                outData[k].append(d["bbox"])

            return outData


if __name__ == "__main__" :
    # fName = "groundtruths/000000.txt"
    # print(loadBboxFile(fName))
    # dirToSave = "matchingBboxes"
    # bboxDir = "bbox"
    # getMatchingBboxes(dirToSave)
    # cropBboxImages()
    gtImgsLocation = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/orig_images_scaled"
    predImgsLocation = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/orig_images_scaled_output"

    annonDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs"

    gtDetFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/personDetectorOut/orig.json"
    predDetFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/personDetectorOut/pred.json"


    bb = BBoxExtractor(gtImgsLocation,predImgsLocation,annonDir,gtDetFile,predDetFile)
    # bb.extractAllBboxes()
    bb.plotSinetValues("/home/akunchala/Documents/PhDStuff/PrivacyFramework/pixEval.json")