# Extract persons from the neural-art images using masks
# ref - https://stackoverflow.com/questions/34691128/how-to-use-mask-to-remove-the-background-in-python

img1 = "/home/akunchala/Documents/PhDStuff/testing/fast_neural_style/out/000000.png"
msk1 = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/masks/000000.png"
bk1 = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/background/000000.png"

from skimage import io
import numpy as np
import cv2
from multiprocessing import Pool
import os

# img = cv2.cvtColor(cv2.imread(img1),
#                    cv2.COLOR_BGR2RGB)
# _, mask = cv2.threshold(cv2.imread(msk1, 0),
#                         0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# bk = cv2.cvtColor(cv2.imread(bk1),
#                    cv2.COLOR_BGR2RGB)


# # get masked foreground
# fg_masked = cv2.bitwise_and(img, img, mask=mask)

# # get masked background, mask must be inverted 
# mask = cv2.bitwise_not(mask)
# bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

# # combine masked foreground and masked background 
# final = cv2.bitwise_or(fg_masked, bk_masked)

# # Using cv2.imshow() method 
# # Displaying the image 
# cv2.imwrite("final.png", final)
# cv2.imwrite("fg_masked.png", fg_masked)
# cv2.imwrite("bk_masked.png", bk_masked)

class ProcessNeuralArt:
    def __init__(self,neuralArtDir,maskDir,backgroundDir,outDir) :
        self.neuralArtDir = neuralArtDir
        self.maskDir = maskDir
        self.backgroundDir = backgroundDir
        self.outDir = outDir
    
    def getOutImg(self,imgFileName):
        img = os.path.join(self.neuralArtDir,imgFileName)
        msk = os.path.join(self.maskDir,imgFileName)
        bk = os.path.join(self.backgroundDir,imgFileName)

        outImgPath = os.path.join(self.outDir,imgFileName)

        img = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
        msk1 = cv2.cvtColor(cv2.imread(msk),cv2.COLOR_BGR2RGB)

        _, mask = cv2.threshold(cv2.imread(msk, 0),0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bk = cv2.cvtColor(cv2.imread(bk),cv2.COLOR_BGR2RGB)

        # get masked foreground
        fg_masked = cv2.bitwise_and(img, img, mask=mask)

        # get masked background, mask must be inverted 
        mask = cv2.bitwise_not(mask)
        bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

        # out_test = cv2.addWeighted(fg_masked, 0.5 ,bk,0.5,0.0)

        # combine masked foreground and masked background 
        final = cv2.bitwise_or(fg_masked, bk_masked)

        cv2.imwrite(outImgPath, final)
    
    def processAllImgs(self):
        allImgs = [os.path.basename(x) for x in sorted(os.listdir(self.neuralArtDir))]
        # print(allImgs)
        with Pool() as pool:
            pool.map(self.getOutImg, allImgs)


if __name__ == "__main__" :
    neuralArtDir = "/home/akunchala/Documents/PhDStuff/testing/fast_neural_style/out"
    maskDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/masks"
    backgroundDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/background"
    outDir = "/home/akunchala/Documents/PhDStuff/testing/fast_neural_style/out_with_background"

    pn = ProcessNeuralArt(neuralArtDir=neuralArtDir,maskDir=maskDir,backgroundDir=backgroundDir,outDir=outDir)
    pn.processAllImgs()