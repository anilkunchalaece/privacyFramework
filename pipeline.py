"""
Author : Kunchala Anil

This script will generate privacy preserving video / dataset / images for given set
of images.

Pipeline consist of following
    Step 1 : If input is video generate images 
    Step 2 : generate masks using getMaskdFromImage.py 
    Step 3 : remove all the people using generated masks and images using STTN (runSttn.py)
    Step 4 : Generate wireframe representations from input images using VIBE (runVibe.py)
    Step 5 : Render generated wireframes to the background images generated using STTN
"""

import argparse
import os,sys,subprocess,shutil

from getMaksFromImage import GetMask
from lib.ppfp.config import getConfig

def main(args):
    config = getConfig()

    # check if tmp dir exist or not - if not exist create one
    createDir(args.tmp_dir,True)
    print(args.tmp_dir)

    # step 1 : check vid_file or images_dir
    print("########## STEP 1 creating images if video file is used #################")
    if args.vid_file == None and args.images_dir == None :
        raise("Please specify either --vid_file or --images_dir as argument")
    elif args.vid_file != None :
        srcImagesDir = createDir(os.path.join(args.tmp_dir,"orig_images"))
        generateImagesFromVideo(args.vid_file, srcImagesDir)
    else :
        print(F"processing images using src {args.images_dir}")
        srcImagesDir = args.images_dir
    
    # step 2 : generate masks for images
    print("########## STEP 2 generating masks for images #################")
    srcImgMasksDir = createDir(os.path.join(args.tmp_dir,"masks"))
    m = GetMask()
    m.generateMasks(srcImagesDir, srcImgMasksDir)

    # step 3 : run STTN to extract background
    print("########## STEP 3 extracting background #################")
    backgroundImgsDir = createDir(os.path.join(args.tmp_dir,"background"))
    extractBackgroundUsingStnn(srcImgMasksDir, srcImagesDir,backgroundImgsDir)

    # step 4 : run VIBE to extract wireframe representations from file
    print("########## STEP 4 generating wireframes #################")
    extractWireframesUsingVibe(srcImagesDir,args.output_dir,backgroundImgsDir)


def extractBackgroundUsingStnn(maskImgDir,srcImageDir,backgroundImgsDir,checkPointFile=os.path.join("data","sttn_data","sttn.pth")):
    print(F"Extracting background using sttn with masks from {maskImgDir} , images from {srcImageDir} and checkpoint from {checkPointFile}")
    cmd = ["python","runSttn.py","--mask",maskImgDir,"--ckpt",checkPointFile,"--image_dir",srcImageDir,"--output_dir",backgroundImgsDir]
    runCmd(cmd)

def extractWireframesUsingVibe(srcImageDir,outputDir,backgroundImgsDir):
    print(F"Extracting wireframe representations using images from {srcImageDir}")
    cmd = ["python","runVibe.py","--images_dir",srcImageDir,"--output_folder",outputDir,"--background",backgroundImgsDir]
    runCmd(cmd)


def generateImagesFromVideo(srcFile,dirToStoreImages):
    # generate images using given video
    print(F"generating images using video {srcFile} and saving in {dirToStoreImages}")
    cmd = ["ffmpeg","-i",srcFile,"-vf","fps=30",F"{dirToStoreImages}/%06d.png"]
    runCmd(cmd)

def createDir(dirName,removeIfExists=False) :
    if os.path.exists(dirName) and removeIfExists == True:
        shutil.rmtree(dirName)
    os.makedirs(dirName,exist_ok=True)
    return dirName

def runCmd(cmd) :
    # cmd -> list of cmd and its arguments
    print(F"running cmd {cmd}")
    os.system(" ".join(cmd))
    # r = subprocess.run(cmd,capture_output=True,text=True,check=True)
    # print(F"output of cmd is {r}")





if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, help='input video path', required=False, default=None)
    parser.add_argument('--images_dir', type=str, help='input images dir', required=False, default=None)
    parser.add_argument('--output_dir', type=str, help="output dir to save processed files", required=True)
    parser.add_argument('--tmp_dir', type=str, help='temp dir to save the intermediate results', default='tmp')
    
    args = parser.parse_args()
    main(args)