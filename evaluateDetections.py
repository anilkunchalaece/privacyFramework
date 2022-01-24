####
# this script is used as a evalution script for the privacyFramework
####

from lib.objectDetectionMetrics import pascalvoc
from sinet_train import evalResults
import json,os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from lib.ppfp.cocoConverter import COCOConverter 
from lib.ppfp.personDetector import PersonDetector
from lib.ppfp.keyPointDetector import KeyPointDetector
import sinet_train as sinet
import torch

# torch.set_default_tensor_type(torch.cuda.HalfTensor)

FILE_TO_SAVE_RESULTS = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annomizedImgs/pixelate_body_detections/evaluateResults.json"

def evaluate():
    dPairs = getDirPairs()
    outDict = {}

    for k in dPairs :
        ap = pascalvoc.calculateAP(gtFolder=dPairs[k]["person"]["gt"], predFolder=dPairs[k]["person"]["det"],gt_t=0,det_t=0)
        si = evalResults(gtDir=dPairs[k]["sinet"]["gt"], predDir=dPairs[k]["sinet"]["det"])
        print(F"{k} => ap: {ap}, si, {si}")
        outDict[k] = {
            "ap" : ap,
            "si" : si
        }
    with open(FILE_TO_SAVE_RESULTS,"w") as fd :
        json.dump(outDict,fd)


def plotResults():
    with open(FILE_TO_SAVE_RESULTS) as fd:
        data = json.load(fd)

    labels = ["blur_face","pix_face","wireframes","blur_body","pix_body"]
    ap = []
    si = []
    pur = []
    for k in labels :
        _ap = data[k]["ap"]
        _si = 1 - data[k]["si"]
        ap.append(_ap)
        si.append(_si)
        pur.append(_si/_ap)
    print(labels)
    print(F"ap => {ap}")
    print(F"si => {si}")
    print(F"pur => {pur}")
    # plt.plot(labels,ap,label="Utility")
    # plt.plot(labels,si,label="Privacy")
    plt.plot(ap,si)
    plt.xlabel("Utility")
    plt.ylabel("Privacy")
    # plt.plot(labels,pur,label="PUR")
    plt.legend()
    plt.grid()
    plt.show()
    



def getDirPairs():
    return {
        "wireframes" : {
            "person" :{
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/groundtruths",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/detections"
            },
            "sinet" : {
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/gt",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/pred"
            }
        },
        "blur_face" : {
            "person" :{
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/groundtruths",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annomizedImgs/blur_faces_detections"
            },
            "sinet" : {
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/gt",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/blur_faces"
            }
        },
        "blur_body" : {
            "person" :{
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/groundtruths",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annomizedImgs/blur_body_detections"
            },
            "sinet" : {
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/gt",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/blur_body"
            }
        },
        "pix_face" : {
            "person" :{
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/groundtruths",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annomizedImgs/pixelate_faces_detections"
            },
            "sinet" : {
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/gt",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/pixelate_faces"
            }
        },
        "pix_body" : {
            "person" :{
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/groundtruths",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annomizedImgs/pixelate_body_detections"
            },
            "sinet" : {
                "gt" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/gt",
                "det" : "/home/akunchala/Documents/PhDStuff/PrivacyFramework/bbox/pixelate_body"
            }
        }

    }

# used to calcualte scores for person and keypoint detections
def detectionEval():
    # annFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/keyPoints/orig_keypoints_coco.json"
    # resFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/keyPoints/pred_keypoints.json"
    # resFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/keyPoints/pix_body_keypoints.json"
    # resFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/keyPoints/blur_body_keypoints.json"
    # resFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/keyPoints/orig_keypoints.json"
    
    srcDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/keyPointDetectorOut/"
    
    resList = [
        "blur_faces.json",
        "pixelate_faces.json",
        "pred.json",
        "pixelate_body.json",
        "blur_body.json",
    ]

    results = {}
    outFileName = os.path.join(srcDir,"results.json")

    for fName in resList :
        annFile = os.path.join(srcDir,"orig.json")
        resFile = os.path.join(srcDir,fName)
        print(fName)
        
        cObj = COCOConverter(annFile)
        annFile = cObj.convert()
        
        cocoGt=COCO(annFile)
        cocoDt=cocoGt.loadRes(resFile)
        cocoEval = COCOeval(cocoGt,cocoDt,"keypoints")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        results[fName] = cocoEval.stats.tolist()
    with open(outFileName,"w") as fd:
        json.dump(results,fd)

def getCocoEval(gtFile, resFile, key) :
    cocoGt=COCO(gtFile)
    cocoDt=cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt,cocoDt,key)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats.tolist()    

def blurPrivacyEval(annonBodyDir):
    baseDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs"
    gtDir = os.path.join(baseDir,"gt","bbox")
    predDir = os.path.join(baseDir,"pred","bbox")
    blurDir = os.path.join(baseDir,"blur")

    gt_imgs = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/orig_images_scaled"
    pred_imgs = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/orig_images_scaled_output"

    pd = PersonDetector()
    kd = KeyPointDetector()

    gt_pd_det_path = pd.generateDetections2(gt_imgs,os.path.join(baseDir,"gt"),"person_det.json")
    gt_kd_det_path = kd.generateDetections(gt_imgs,os.path.join(baseDir,"gt"),"keypoint_det.json")


    cObj = COCOConverter(gt_pd_det_path)
    gt_pd_det_path = cObj.convert()

    cObjKd = COCOConverter(gt_kd_det_path)
    gt_kd_det_path = cObjKd.convert()

    pd = PersonDetector()
    kd = KeyPointDetector()
    pred_pd_det_path = pd.generateDetections2(pred_imgs,predDir,"person_det.json")
    pred_kd_det_path = kd.generateDetections(pred_imgs,predDir,"keypoint_det.json")

    results = {}

    pred_v = sinet.evalResults(gtDir,predDir)


    results["pred_v"] = {
    "sinet" :   pred_v,
    "person" : getCocoEval(gt_pd_det_path,pred_pd_det_path,"bbox"),
    "keypoint" : getCocoEval(gt_kd_det_path,pred_kd_det_path,"keypoints"),
    }

    for d in os.listdir(blurDir):
        c_d = os.path.join(blurDir,d,"bbox")
        p_v = sinet.evalResults(gtDir,c_d)

        pd = PersonDetector()
        kd = KeyPointDetector()
        _pDetectionsPath = pd.generateDetections2(os.path.join(blurDir,d,"blur_body"),os.path.join(blurDir,d),"person_det.json")
        _kDetectionsPath = kd.generateDetections(os.path.join(blurDir,d,"blur_body"),os.path.join(blurDir,d),"keypoint_det.json")

        del pd
        del kd
        torch.cuda.empty_cache()

        results[F"blur_{d}"] = {
         "sinet" :  p_v,
        "person" : getCocoEval(gt_pd_det_path,_pDetectionsPath,"bbox"),
        "keypoint" : getCocoEval(gt_kd_det_path,_kDetectionsPath,"keypoints")
        }

    with open("blurEval.json","w") as fd :
        json.dump(results,fd)

def removeSmallBboxes(fName):
    out = []
    with open(fName) as fd:
        data = json.load(fd)
        _h = 30
        for d in data :
            if d["bbox"][2] > 40 and d["bbox"][3] > 40 :
                out.append(d)
        with open(fName.replace(".json","_removed.json"),"w")  as fw :
            json.dump(out,fw)
            return fName.replace(".json","_removed.json")


def evaluateFiles(blurDir) :

    person_gt = removeSmallBboxes("/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs/gt/person_det.json")
    cObj = COCOConverter(person_gt)
    person_gt_coco = cObj.convert()

    keypoint_gt = removeSmallBboxes("/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs/gt/keypoint_det.json")
    cObj = COCOConverter(keypoint_gt)
    keypoint_gt_coco = cObj.convert()

    person_pred = removeSmallBboxes("/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs/pred/person_det.json")
    keypoint_pred = removeSmallBboxes("/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs/pred/keypoint_det.json")

    print(getCocoEval(person_gt_coco,person_pred,"bbox"))
    print(getCocoEval(keypoint_gt_coco,keypoint_pred,"keypoints"))


def pixPrivacyEval():
    baseDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs"
    gtDir = os.path.join(baseDir,"gt","bbox")
    predDir = os.path.join(baseDir,"pred","bbox")
    pixDir = os.path.join(baseDir,"pix")

    gt_imgs = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/orig_images_scaled"
    pred_imgs = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/orig_images_scaled_output"

    pd = PersonDetector()
    kd = KeyPointDetector()

    gt_pd_det_path = pd.generateDetections2(gt_imgs,os.path.join(baseDir,"gt"),"person_det.json")
    gt_kd_det_path = kd.generateDetections(gt_imgs,os.path.join(baseDir,"gt"),"keypoint_det.json")


    cObj = COCOConverter(gt_pd_det_path)
    gt_pd_det_path = cObj.convert()

    cObjKd = COCOConverter(gt_kd_det_path)
    gt_kd_det_path = cObjKd.convert()

    pd = PersonDetector()
    kd = KeyPointDetector()
    pred_pd_det_path = pd.generateDetections2(pred_imgs,predDir,"person_det.json")
    pred_kd_det_path = kd.generateDetections(pred_imgs,predDir,"keypoint_det.json")

    results = {}

    pred_v = sinet.evalResults(gtDir,predDir)


    results["pred_v"] = {
    "sinet" :   pred_v,
    "person" : getCocoEval(gt_pd_det_path,pred_pd_det_path,"bbox"),
    "keypoint" : getCocoEval(gt_kd_det_path,pred_kd_det_path,"keypoints"),
    }

    for d in os.listdir(pixDir):
        c_d = os.path.join(pixDir,d,"bbox")
        p_v = sinet.evalResults(gtDir,c_d)

        pd = PersonDetector()
        kd = KeyPointDetector()
        _pDetectionsPath = pd.generateDetections2(os.path.join(pixDir,d,"pixelate_body"),os.path.join(pixDir,d),"person_det.json")
        _kDetectionsPath = kd.generateDetections(os.path.join(pixDir,d,"pixelate_body"),os.path.join(pixDir,d),"keypoint_det.json")

        del pd
        del kd
        torch.cuda.empty_cache()

        results[F"pix_{d}"] = {
         "sinet" :  p_v,
        "person" : getCocoEval(gt_pd_det_path,_pDetectionsPath,"bbox"),
        "keypoint" : getCocoEval(gt_kd_det_path,_kDetectionsPath,"keypoints")
        }

    with open("pixEval.json","w") as fd :
        json.dump(results,fd)


def pixPrivacyEval2():
    baseDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs"
    gtDir = os.path.join(baseDir,"gt","bbox")
    predDir = os.path.join(baseDir,"pred","bbox")
    pixDir = os.path.join(baseDir,"pix")

    results = {}

    pred_v = sinet.evalResults(gtDir,predDir)
    results["pred_v"] = pred_v

    for d in os.listdir(pixDir):
        c_d = os.path.join(pixDir,d,"bbox")
        _v = sinet.evalResults(gtDir,c_d)
        results[F"pix_{d}"] = _v

    with open("pixEval.json","w") as fd :
        json.dump(results,fd)

def plotResults2():
    with open("blurEval.json") as fd:
        data = json.load(fd)

        k = sorted(data, key= lambda x:data[x]["sinet"])
        # k.remove("pred_v")

        sinet = []
        pd = []
        kd = []
        ut = []

        for i in k :
            sinet.append(1-data[i]["sinet"])
            pd.append(data[i]["person"][4])
            kd.append(data[i]["keypoint"][4])
            print(F'{data[i]["person"][4]} {data[i]["keypoint"][4]}')
            ut.append((data[i]["person"][4]+data[i]["keypoint"][4])/2)
        
        plt.plot(ut,sinet,marker='o',label="sinet")
        plt.xlabel("Utilization")
        plt.ylabel("Privacy")

        for i, txt in enumerate(k):
            plt.annotate(txt.replace("lur",""), (ut[i]+0.001, sinet[i]+0.001))

        # plt.scatter(k,sinet)
        # plt.plot(k,pd,label="pd")
        # plt.plot(k,kd,label="kd")
        # plt.plot(k,ut,marker='o',label="utility")
        # plt.scatter(k,ut)
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()

            

        

if __name__ == "__main__":
    # evaluate()
    # plotResults()
    # detectionEval()
    # blurPrivacyEval()
    # pixPrivacyEval()
    plotResults2()
    # evaluateFiles("x")
    # removeSmallBboxes("/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs/gt/person_det.json")