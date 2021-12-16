####
# this script is used as a evalution script for the privacyFramework
####

from lib.objectDetectionMetrics import pascalvoc
from sinet_train import evalResults
import json 
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
    # evaluate()
    plotResults()