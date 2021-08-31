# Author : Kunchala Anil
# Date : 31 Aug 2021
# Email : d20125529@mytudublin.ie

import os
import random
import glob
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid



class ImagePairGen :

    def __init__(self,root_dir,limit_ids=None,max_frames=None):
        # limit_ids -> number of pedestrains to be processed -> to simplify testing
        # max_frames -> maximum number of frames per track per pedestrains to collect
        self.rootDir = root_dir
        self.limit_ids = limit_ids
        self.max_frames = max_frames
        

    def readAllImgs(self):
        totalIds = os.listdir(self.rootDir)
        print(F" total number of pedestrian ids in dir {self.rootDir} => {len(totalIds)}")
        
        # limit the number of ids to generate pairs -> used for faster testing 
        if self.limit_ids != None :
            # extract the number of random pedestrian ids
            self.selectedIds = random.sample(totalIds,self.limit_ids)
            print(F"limiting the selected ids to {len(self.selectedIds)} => {self.selectedIds}")
        else :
            self.selectedIds = totalIds

        allImgs = {}

        for sId in self.selectedIds :
            _sidDir = os.path.join(self.rootDir,sId)
            imgsInDir = os.listdir(os.path.join(_sidDir))
            tracklets = list(set([ i[6:11] for i in imgsInDir]))
            # print(F" no of tracklets in {sId} are {len(tracklets)}")

            # get list of tracklets and no of images it contains
            tImgs = [len(glob.glob(F"{_sidDir}/*{t}*")) for t in tracklets]
            tImgsMaxIndex = tImgs.index(max(tImgs))
            
            tMaxFrames = tracklets[tImgsMaxIndex]
            print(F"no of tracklets in pId : {sId} are : {len(tracklets)} and tacklet : {tMaxFrames} has Max number of frames : {max(tImgs)}")
            # self.getTrackeletIdwithMaxFrames(_sidDir)
            pImgs= glob.glob(F"{_sidDir}/*{tMaxFrames}*") 
            if self.max_frames == None or max(tImgs) < self.max_frames:
                allImgs[sId] = pImgs
            else : 
                print(F"selecting only : {self.max_frames} images from {max(tImgs)}")
                allImgs[sId] = glob.glob(F"{_sidDir}/*{tMaxFrames}*")[:self.max_frames]
        
        return allImgs

    
    def generateTripletsRandomly(self):
        # This function will generate triplets randomly in following fashion
        # positive pairs 
        allImgs = self.readAllImgs()
        finalTriplets = []

        for pId in allImgs.keys() :
            cImgs = allImgs[pId]
            totalImgs = len(cImgs)
            pairsToGenerate = int(totalImgs/4)
            # print(F"Total images in {pId} are {totalImgs}, selecting {pairsToGenerate} pairs")
            
            # randomly sample anchor images from first half of the images
            anchorImgs = random.sample(cImgs[:int(totalImgs/2)],pairsToGenerate)

            # randomly sample positive images from second half of selected images
            positiveImgs = random.sample(cImgs[int(totalImgs/2):],pairsToGenerate)

            nImgsToCollect = math.ceil(pairsToGenerate/len(allImgs.keys())) + 1
            #randomly select negative image from one of the remaining folders
            negativeImgs = []
            cnt = 0 
            
            _keys = list(allImgs.keys())
            random.shuffle(_keys) # shiffle keys so order wont be repeated

            for pName in _keys:
                if pName == pId :
                    continue # ignore the same pIdx
                else :
                    neg_imgs = random.sample(allImgs[pName],nImgsToCollect)
                    negativeImgs.extend(neg_imgs)
                    cnt = cnt + 1

                    if cnt == pairsToGenerate :
                        break
            # print("anchor",len(anchorImgs))
            # print(len(negativeImgs))
            triplets = [ (anchorImgs[i],positiveImgs[i],negativeImgs[i]) for i in range(len(anchorImgs))]
            finalTriplets.extend(triplets)
        
        print(F"#######\n Total number of triplets selected are {len(finalTriplets)} \n#######")
        return finalTriplets

    def visualizeTriplets(self) :
        triplets = self.generateTripletsRandomly()
        triplets_sel = random.sample(triplets,9)
        # select 3 triplets and show them in grid
        fig = plt.figure()
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(3, 9),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        imgs = []
        for img in triplets_sel :
            imgs.extend([cv2.imread(i)[:,:,::-1] for i in img])

        for ax, im in zip(grid, imgs):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
        plt.show()


    def getTrackeletIdwithMaxFrames(self,pDir) :
        # not using this for now -> for now just take single tracklet images , 
        # in future this can be extended to multiple tracklets
        # get trackletId which is having maximum number of frames for given directory
        # pDir -> dir (abs) path where all images stored        
        allImgs = os.listdir(os.path.join(pDir))
        tracklets = set([ i[6:11] for i in allImgs])
        # print(F"no of tracklets in {pDir} are {len(tracklets)}")
        for t in tracklets :
            imgs = glob.glob(F"{pDir}/*{t}*")
            # print(imgs)
            print(F"no of tracklets in {pDir} are {len(tracklets)} and {pDir} has {len(imgs)} for tracklet {t}")


if __name__ == "__main__":
    root_dir = "/home/akunchala/Downloads/MARS_Dataset/bbox_train"
    p = ImagePairGen(root_dir,limit_ids=20 ,max_frames=None)
    p.visualizeTriplets()