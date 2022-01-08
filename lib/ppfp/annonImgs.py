import cv2
import os, json
import numpy as np
from multiprocessing import Pool


class AnnonImgs:
    def __init__(self,srcDir,outDir,blurFactor=None,pixelateFactor=None) :
        self.origImgs = srcDir
        self.outDir = outDir
        self.blurFactor = blurFactor
        self.pixelateFactor = pixelateFactor
    
    # ref - https://sefiks.com/2020/11/07/face-and-background-blurring-with-opencv-in-python/
    def blur_img(self,img, factor = 3):

        if self.blurFactor != None :
            factor = self.blurFactor

        kW = int(img.shape[1] / factor)
        kH = int(img.shape[0] / factor)

        #ensure the shape of the kernel is odd
        if kW % 2 == 0: kW = kW - 1
        if kH % 2 == 0: kH = kH - 1

        blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
        return blurred_img

    def createDirIfNotExist(self,d) :
        try :
            if not os.path.isdir(d) :
                try :
                    os.mkdir(d)
                except :
                    os.makedirs(d)
        except :
            pass

    def annomizeImgsInDir(self,facesDir) :
        outDir = self.outDir
        facesDir = os.path.join(os.getcwd(),"faces")
        # personDir = os.path.join(os.getcwd(),"groundtruths")

        self.createDirIfNotExist(outDir)

        # print(getDataFromFile("/home/akunchala/Documents/PhDStuff/PrivacyFramework/faces/000622.json", "k")[0])

        for d in [facesDir] :
            print(F"processing {d}")
            # process faces
            with Pool() as pool:
                pool.map(self.blurBbox,[ os.path.join(d,x) for x in os.listdir(d)])
            
            with Pool() as pool:
                pool.map(self.pixelateBbox,[ os.path.join(d,x) for x in os.listdir(d)])

    def blurPersonsInDir(self,personDir):
        outDir = self.outDir
        # personDir = os.path.join(os.getcwd(),"groundtruths")

        self.createDirIfNotExist(outDir)

        # process faces
        with Pool() as pool:
            pool.map(self.blurBbox,[ os.path.join(personDir,x) for x in os.listdir(personDir)])
        # for x in os.listdir(personDir) :
        #     self.blurBbox(os.path.join(personDir,x))
    

    def pixelatePersonsInDir(self,personDir):
        outDir = self.outDir
        # personDir = os.path.join(os.getcwd(),"groundtruths")

        self.createDirIfNotExist(outDir)
        
        with Pool() as pool:
            pool.map(self.pixelateBbox,[ os.path.join(personDir,x) for x in os.listdir(personDir)])


    def blurBbox(self,imgFile):
        imgName, bboxes,outDir = self.getDataFromFile(imgFile,"blur")
        # print(imgName)
        img = cv2.imread(imgName)
        
        for box in bboxes :
            try :
                x1,y1,x2,y2 = [int(x) for x in box]
                face = img[y1:y2,x1:x2]
                try :
                    face_blurred = self.blur_img(face,factor=2)
                except :
                    # print("factor")
                    try :
                        face_blurred = self.blur_img(face,factor=2)
                    except:
                        # face_blurred = blur_img(face,factor=1)
                        # unable to blur, face is too small
                        face_blurred = face
                img[y1:y2,x1:x2] = face_blurred
            except :
                pass
        
        outPath = os.path.join(outDir,os.path.basename(imgName))
        cv2.imwrite(outPath,img)


    def pixelateBbox(self,imgFile):
        imgName, bboxes,outDir = self.getDataFromFile(imgFile,"pixelate")

        img = cv2.imread(imgName)
        for box in bboxes :
            try :
                x1,y1,x2,y2 = [int(x) for x in box]
                face = img[y1:y2,x1:x2]
                # try :
                face_pixelated = self.anonymize_face_pixelate(face)
                # except :
                    # face_pixelated = anonymize_face_pixelate(face,blocks=6)
                img[y1:y2,x1:x2] = face_pixelated
            except :
                pass

        outPath = os.path.join(outDir,os.path.basename(imgName))
        cv2.imwrite(outPath,img) 


    def anonymize_face_pixelate2(self,image):
        # divide the input image into NxN blocks
        (h, w) = image.shape[:2]
        # print(F"h {h} w {w}")

        if self.pixelateFactor != None :
            factor = self.pixelateFactor
        else :
            factor = 4
        xblocks = int(w/factor)
        yblocks = int(h/factor)

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

    def anonymize_face_pixelate(self,image):
        # ref - https://stackoverflow.com/questions/55508615/how-to-pixelate-image-using-opencv-in-python
        # divide the input image into NxN blocks
        (h, w) = image.shape[:2]
        if self.pixelateFactor == None :
            self.pixelateFactor = 4
        
        pix_size = (self.pixelateFactor , self.pixelateFactor)

        try :

            # Resize input to "pixelated" size
            temp = cv2.resize(image, pix_size, interpolation=cv2.INTER_LINEAR)

            # Initialize output image
            output = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        except :
            return image
            
        return output

    # get bbox information and fileName from given file
    def getDataFromFile(self,fName,k):
        # origImgs = '/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled'

        if fName.split(".")[-1] == "json" :
            outDir = F"{self.outDir}/{k}_faces"
            imgName = os.path.join(self.origImgs,os.path.basename(fName).replace(".json",".png"))
            self.createDirIfNotExist(outDir)

            with open(fName) as fd:
                d = json.load(fd)
                return imgName , d["facesBbox"], outDir
        else :
            outDir =  F"{self.outDir}/{k}_body"  
            self.createDirIfNotExist(outDir)
            imgName = os.path.join(self.origImgs,os.path.basename(fName).replace(".txt",".png"))
            with open(fName) as fd:
                b = []
                for l in fd:
                    b.append([float(x.replace("[","").replace("]","").replace(",", "").replace("}", "")) for x in l.split(" ")[-4:]])
                return imgName, b, outDir   


if __name__ == "__main__":
    personsDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/persons"
    annonImgsPath = "annoImgs"
    srcImagesDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp2/orig_images_scaled"

    aI = AnnonImgs(srcImagesDir,annonImgsPath)
    
    blurRange = range(1,100,5)
    pixelatedRange = range(2,100,5)

    print("Blurring images")
    for b in blurRange :
        aI.outDir = os.path.join(annonImgsPath,"blur",str(b))
        aI.blurFactor = b 
        aI.blurPersonsInDir(personsDir)
    
    print("Pixelating images")
    aI = AnnonImgs(srcImagesDir,annonImgsPath)
    for b in pixelatedRange :
        aI.outDir = os.path.join(annonImgsPath,"pix",str(b))
        aI.pixelateFactor = b 
        aI.pixelatePersonsInDir(personsDir)