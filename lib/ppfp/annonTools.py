## Tools used to annomize the faces in images

import cv2
import os, json
import numpy as np
from multiprocessing import Pool


# ref - https://sefiks.com/2020/11/07/face-and-background-blurring-with-opencv-in-python/
def blur_img(img, factor = 20):
    kW = int(img.shape[1] / factor)
    kH = int(img.shape[0] / factor)

    #ensure the shape of the kernel is odd
    if kW % 2 == 0: kW = kW - 1
    if kH % 2 == 0: kH = kH - 1

    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    return blurred_img


# srcDir -> contains JSON files with faceBBoxes
# desDir -> dir to store image with blurFace
def blurFacesInDir(srcDir,desDir) :
    pass

# get bbox information and fileName from given file
def getDataFromFile(fName,k):
    origImgs = '/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled'

    if fName.split(".")[-1] == "json" :
        outDir = F"annomizedImgs/{k}_faces"
        imgName = os.path.join(origImgs,os.path.basename(fName).replace(".json",".png"))
        createDirIfNotExist(outDir)

        with open(fName) as fd:
            d = json.load(fd)
            return imgName , d["facesBbox"], outDir
    else :
        outDir =  F"annomizedImgs/{k}_body"  
        createDirIfNotExist(outDir)
        imgName = os.path.join(origImgs,os.path.basename(fName).replace(".txt",".png"))
        with open(fName) as fd:
            b = []
            for l in fd:
                b.append([float(x.replace("[","").replace("]","").replace(",", "").replace("}", "")) for x in l.split(" ")[-4:]])
            return imgName, b, outDir   

def blurBbox(imgFile):
    imgName, bboxes,outDir = getDataFromFile(imgFile,"blur")
    # print(imgName)
    img = cv2.imread(imgName)

    for box in bboxes :
        x1,y1,x2,y2 = [int(x) for x in box]
        face = img[y1:y2,x1:x2]
        try :
            face_blurred = blur_img(face,factor=70)
        except :
            # print("factor")
            try :
                face_blurred = blur_img(face,factor=7)
            except:
                # face_blurred = blur_img(face,factor=1)
                # unable to blur, face is too small
                face_blurred = face
        img[y1:y2,x1:x2] = face_blurred
    
    outPath = os.path.join(outDir,os.path.basename(imgName))
    cv2.imwrite(outPath,img)
    # return img

def anonymize_face_pixelate(image):
	# divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    # print(F"h {h} w {w}")
    xblocks = int(w/3)
    yblocks = int(h/3)

    xSteps = np.linspace(0, w, xblocks + 1, dtype="int")
    ySteps = np.linspace(0, h, yblocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)
    # return the pixelated blurred image
    return image

def pixelateBbox(imgFile):
    imgName, bboxes,outDir = getDataFromFile(imgFile,"pixelate")

    img = cv2.imread(imgName)
    for box in bboxes :
        x1,y1,x2,y2 = [int(x) for x in box]
        face = img[y1:y2,x1:x2]
        # try :
        face_pixelated = anonymize_face_pixelate(face)
        # except :
            # face_pixelated = anonymize_face_pixelate(face,blocks=6)
        img[y1:y2,x1:x2] = face_pixelated

    outPath = os.path.join(outDir,os.path.basename(imgName))
    cv2.imwrite(outPath,img) 
    # return img

# Descrete cosine transformation
def scrambleUsingDCT(imgFile):
    imgName, bboxes, outDir = getDataFromFile(imgFile,"dct")

    img = cv2.imread(imgName)
    for box in bboxes :
        x1,y1,x2,y2 = [int(x) for x in box]
        face = img[y1:y2,x1:x2]
        # face = np.float32(face)
        try :
            f1,f2,f3 = cv2.split(face)
            f1_o = np.uint8(cv2.dct(np.float32(f1)))
            f2_o = np.uint8(cv2.dct(np.float32(f2)))
            f3_o = np.uint8(cv2.dct(np.float32(f3)))

            face_scrambled = cv2.merge([f1_o,f2_o,f3_o])
        except :
            face_scrambled = face
        
        img[y1:y2,x1:x2] = face_scrambled

    # return img
    outPath = os.path.join(outDir,os.path.basename(imgName))
    cv2.imwrite(outPath,img)

def createDirIfNotExist(d) :
    try :
        if not os.path.isdir(d) :
            try :
                os.mkdir(d)
            except :
                os.makedirs(d)
    except :
        pass

def annomizeImgsInDir(outDir,facesDir,personDir) :
    outDir = os.path.join(os.getcwd(),"annomizedImgs") 
    facesDir = os.path.join(os.getcwd(),"faces")
    personDir = os.path.join(os.getcwd(),"groundtruths")

    createDirIfNotExist(outDir)

    # print(getDataFromFile("/home/akunchala/Documents/PhDStuff/PrivacyFramework/faces/000622.json", "k")[0])

    for d in [facesDir,personDir] :
        print(F"processing {d}")
        # process faces
        with Pool() as pool:
            pool.map(blurBbox,[ os.path.join(d,x) for x in os.listdir(d)])
        
        with Pool() as pool:
            pool.map(pixelateBbox,[ os.path.join(d,x) for x in os.listdir(d)])

        # with Pool() as pool:
        #     pool.map(scrambleUsingDCT,[ os.path.join(d,x) for x in os.listdir(d)])



if __name__ == "__main__":
    annomizeImgsInDir()
    # fName = "faces/000001.json"
    # with open(fName) as fd:
    #     d = json.load(fd)
    # img = pixelateFace(d["fileName"], d["facesBbox"])
    # img2 = blurFace(d["fileName"], d["facesBbox"])
    # img3 = scrambleUsingDCT(d["fileName"], d["facesBbox"])

    # cv2.imwrite("pixelated_out.png", img)
    # cv2.imwrite("blur_out.png", img2)
    # cv2.imwrite("dct_out.png", img3)

    # with open("groundtruths/000001.txt") as fd:
    #     b = []
    #     for l in fd:
    #         b.append([float(x) for x in l.split(" ")[-4:]])
    #     # print(b)
    # img4 = blurFace(d["fileName"], b,factor=15)
    # cv2.imwrite("blur_full_out.png", img4)

    # img5 = pixelateFace(d["fileName"], b,blocks=3)
    # cv2.imwrite("pixelate_full_out.png", img5)

    # img6 = scrambleUsingDCT(d["fileName"],b)
    # cv2.imwrite("scramble_full_out.png", img6)