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
import os,sys,subprocess,shutil,json

from getMaksFromImage import GetMask
from lib.ppfp.config import getConfig
from lib.ppfp.insightFace import InsightFaceDetector
from lib.ppfp.personDetector import PersonDetector
from lib.ppfp.keyPointDetector import KeyPointDetector
from lib.ppfp.annonImgs import AnnonImgs
from lib.ppfp.extractBbox import BBoxExtractor
import sinet_train as sinet

from multiprocessing import Pool
from moviepy.editor import VideoFileClip, TextClip, clips_array, vfx,CompositeVideoClip

c = {
    "src" : False,
    "pred" : False,
    "background" : False,
    "detections" : False,
}

def main(args):
    config = getConfig()

    generateVideo = False

    frameRate = args.frame_rate

    # check if tmp dir exist or not - if not exist create one
    createDir(args.tmp_dir,False)
    print(args.tmp_dir)

    # step 1 : check vid_file or images_dir
    print("########## STEP 1 creating images if video file is used #################")
    if args.vid_file == None and args.images_dir == None :
        raise("Please specify either --vid_file or --images_dir as argument")
    elif args.vid_file != None :
        srcImagesDirOrg = createDir(os.path.join(args.tmp_dir,"orig_images"))
        if c["src"] == True :
            generateImagesFromVideo(args.vid_file, srcImagesDirOrg)
    else :
        print(F"processing images using src {args.images_dir}")
        srcImagesDirOrg = args.images_dir
    
    # Step 1.5 -> resize the images to w, h = 432, 240 (from sttn)
    srcImagesDir = createDir(os.path.join(args.tmp_dir,"src","orig_images_scaled"))
    if c["src"] == True :
        resizeImages(srcImagesDirOrg,srcImagesDir)

    # step 2 : generate masks for images
    print("########## STEP 2 generating masks for images #################")
    srcImgMasksDir = createDir(os.path.join(args.tmp_dir,"masks"))
    if c["src"] == True :
        m = GetMask()
        m.generateMasks(srcImagesDir, srcImgMasksDir)

    # step 3 : run STTN to extract background
    print("########## STEP 3 extracting background #################")
    backgroundImgsDir = createDir(os.path.join(args.tmp_dir,"background"))
    if c["src"] == True :
        extractBackgroundUsingStnn(srcImgMasksDir, srcImagesDir,backgroundImgsDir)

    if c["pred"] == True :
        # step 4 : run VIBE to extract wireframe representations from file
        print("########## STEP 4 generating wireframes #################")
        extractWireframesUsingVibe(srcImagesDir,args.output_dir,backgroundImgsDir)

    if generateVideo == True :
        generateVideoFromImages(backgroundImgsDir,frameRate=frameRate)
        generateVideoFromImages(srcImagesDir,frameRate=frameRate)
        generateVideoFromImages(F"{srcImagesDir}_output",frameRate=frameRate)
        generateVideoFromImages(F"{srcImagesDir}_output_empty_background",frameRate=frameRate)

        generateFinalVideo(args.tmp_dir)
    
    ## Ananomization pipeline
    # run the face and person detector and generate the detections
    # use the detections to blur and pixelate the face and persons

    # detect faces
    faceDetectionsPath = os.path.join(args.tmp_dir,"faces")
    createDir(faceDetectionsPath,False)
    if c["detections"] == True :
        fd = InsightFaceDetector(srcImagesDir,outDir=faceDetectionsPath)
        fd.detectAllFacesInDir()

    # detect body
    bodyDetectionsPath = os.path.join(args.tmp_dir,"persons")
    createDir(bodyDetectionsPath)
    if c["detections"] == True :
        pd = PersonDetector()
        pd.generateDetections(srcImagesDir,dirToSave=bodyDetectionsPath)

    # generate blurred & pixelated images
    annonImgsPath = os.path.join(args.tmp_dir,"annonImgs")
    # annonTools.annomizeImgsInDir(outDir=annonImgsPath,facesDir=faceDetectionsPath,personDir=bodyDetectionsPath)

    # blur and pixelate faces
    # aI = AnnonImgs(srcImagesDir,annonImgsPath)
    # aI.annomizeImgsInDir(facesDir=faceDetectionsPath)

    ## Run Person Detector and KeyPoint detector for original, privacy enhanced and annon img's
    origDir = os.path.join(args.tmp_dir,"src","orig_images_scaled")
    predDir = os.path.join(args.tmp_dir,"pred","orig_images_scaled_output")

    # personDetectorOutDir = os.path.join(args.tmp_dir,"personDetectorOut")
    # createDir(personDetectorOutDir)

    # keyPointDetectirOutDir = os.path.join(args.tmp_dir,"keyPointDetectorOut")
    # createDir(keyPointDetectirOutDir)

    # kd = KeyPointDetector()

    # run person detector for all dirs
    print("Running detections for original rescaled images")
    # pd.generateDetections2(origDir,personDetectorOutDir,"orig.json")
    # kd.generateDetections(origDir,keyPointDetectirOutDir,"orig.json")
    if c["detections"] == True :
        detectPersons(origDir,os.path.join(args.tmp_dir,"src"),"person_det.json")
        detectKeyPoints(origDir,os.path.join(args.tmp_dir,"src"),"keypoint_det.json")

    print("Running detections for privacy enhanced / pred images")
    # pd.generateDetections2(predDir,personDetectorOutDir,"pred.json")
    # kd.generateDetections(predDir,keyPointDetectirOutDir,"pred.json")
    if c["detections"] == True :
        detectPersons(predDir,os.path.join(args.tmp_dir,"pred"),"person_det.json")
        detectKeyPoints(predDir,os.path.join(args.tmp_dir,"pred"),"keypoint_det.json")

    # Skipping it for now
    # # runnning for face annomizations
    # for d in os.listdir(annonImgsPath) :
    #     print(F"Running detections for {d} images")
    #     createDir(os.path.join(annonImgsPath,d.split("_")[0]))
    #     detectPersons(os.path.join(annonImgsPath,d),os.path.join(annonImgsPath,d.split("_")[0]),F"person_det.json")
    #     detectKeyPoints(os.path.join(annonImgsPath,d),os.path.join(annonImgsPath,d.split("_")[0]),F"keypoint_det.json")

    
    annonBodyImgsPath = os.path.join(args.tmp_dir,"annonBodyImgs")
    aI = AnnonImgs(srcImagesDir,annonBodyImgsPath)
    
    blurRange = range(1,100,2)
    pixelatedRange = range(2,100,2)

    if c["detections"] == True :

        print("Blurring images")
        for b in blurRange :
            aI.outDir = os.path.join(annonBodyImgsPath,"blur",str(b))
            aI.blurFactor = b 
            aI.blurPersonsInDir(bodyDetectionsPath)
        
        print("Running detections for blurred images")
        for d in os.listdir(os.path.join(annonBodyImgsPath,"blur")) :
            detectPersons(os.path.join(annonBodyImgsPath,"blur",d,"blur_body"),os.path.join(annonBodyImgsPath,"blur",d),"person_det.json")
            detectKeyPoints(os.path.join(annonBodyImgsPath,"blur",d,"blur_body"),os.path.join(annonBodyImgsPath,"blur",d),"keypoint_det.json")


        print("Pixelating images")
        aI = AnnonImgs(srcImagesDir,annonBodyImgsPath)
        for b in pixelatedRange :
            aI.outDir = os.path.join(annonBodyImgsPath,"pix",str(b))
            aI.pixelateFactor = b 
            aI.pixelatePersonsInDir(bodyDetectionsPath)

        print("Running detections for pixelated images")
        for d in os.listdir(os.path.join(annonBodyImgsPath,"pix")) :
            detectPersons(os.path.join(annonBodyImgsPath,"pix",d,"pixelate_body"),os.path.join(annonBodyImgsPath,"pix",d),"person_det.json")
            detectKeyPoints(os.path.join(annonBodyImgsPath,"pix",d,"pixelate_body"),os.path.join(annonBodyImgsPath,"pix",d),"keypoint_det.json")

    extractBBoxes(args)
    calculteSIValues(args)


def extractBBoxes(args):
    tmpDir = args.tmp_dir
    gtImgsLocation = os.path.join(tmpDir,"src","orig_images_scaled")
    predImgsLocation = os.path.join(tmpDir,"pred","orig_images_scaled_output")
    gtDetFile = os.path.join(tmpDir,"src","person_det.json")
    predDetFile = os.path.join(tmpDir,"pred","person_det.json")
    annonDir = os.path.join(tmpDir,"annonBodyImgs")

    bExtractor = BBoxExtractor(gtImgsLocation,predImgsLocation,annonDir,gtDetFile,predDetFile)
    bExtractor.extractAllBboxes()

def calculteSIValues(args):
    tmpDir = args.tmp_dir
    annonDir = os.path.join(tmpDir,"annonBodyImgs")

    gtImgsDir = os.path.join(annonDir,"gt","bbox")
    predImgsDir = os.path.join(annonDir,"pred","bbox")

    pred_v = sinet.evalResults(gtImgsDir,predImgsDir)
    
    blur = dict()
    for d in os.listdir(os.path.join(annonDir,"blur")) :
        _c_si = sinet.evalResults(gtImgsDir,os.path.join(annonDir,"blur",d,"bbox"))
        blur[d] = _c_si
    
    pix = dict()
    for d in os.listdir(os.path.join(annonDir,"pix")) :
        _c_si = sinet.evalResults(gtImgsDir,os.path.join(annonDir,"pix",d,"bbox"))
        pix[d] = _c_si

    outFile = os.path.join(annonDir,"sinetValues.json")
    with open(outFile,"w") as fd:
        json.dump({
            "pred" : pred_v,
            "blur" : blur,
            "pix" : pix
        },fd)


def detectFaces() :
    pass

def detectPersons(srcDir, outDir, fileName) :
    pd = PersonDetector()
    pd.generateDetections2(srcDir, outDir, fileName)

def detectKeyPoints(srcDir, outDir, fileName) :
    kd = KeyPointDetector()
    kd.generateDetections(srcDir, outDir, fileName)

def resizeImages(srcDir,desDir):
    print(" resizing the images ")
    allImages = sorted(os.listdir(srcDir))
    imageExtn = os.listdir(srcDir)[0][-3:]
    # for idx in range(len(allImages)) :
    #     cmd = ['ffmpeg','-hide_banner','-loglevel','error', '-i', F'"{srcDir}/{allImages[idx]}"','-s','432x240',F'"{desDir}/{idx:06d}.png"']
    #     runCmd(cmd)
    # # cmd = ['convert', F'"{srcDir}/*.{imageExtn}"','-resize' ,'432x240!' ,'-set', 'filename:base', '"%[basename]"',F'"{desDir}/%[filename:base].{imageExtn}"']
    # cmd = ['for','f','in',F'{srcDir}/*',';','do','ffmpeg','-hide_banner','-loglevel','error','-i','$f','-s','432x240',F'"{desDir}/$(basename $f)"', ';','done']
    # runCmd(cmd)
    with Pool() as pool: 
        pool.map(runCmd, [['ffmpeg','-hide_banner','-loglevel','error', '-i', F'"{srcDir}/{allImages[idx]}"','-s','432x240',F'"{desDir}/{idx:06d}.png"'] for idx in range(len(allImages))])

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

def generateVideoFromImages(dirName,frameRate=20):
    print("generating video from images")
    imageExtn = os.listdir(dirName)[0][-3:]
    cmd = ["ffmpeg","-y","-threads","16","-i",F'"{dirName}/%06d.{imageExtn}"',"-profile:v","baseline","-level","3.0","-c:v","libx264","-pix_fmt","yuv420p","-an","-v","error", F'"{dirName}/result.mp4"',"-framerate",frameRate,"-r","60"]
    runCmd(cmd)

def runCmd(cmd) :
    # cmd -> list of cmd and its arguments
    cmd = " ".join(cmd)
    # print(F"running cmd {cmd}")
    os.system(cmd)
    # r = subprocess.run(cmd,capture_output=True,text=True,check=True)
    # print(F"output of cmd is {r}")



def generateFinalVideo(baseDir):
    margin = 1 
    fontSize = 26
    clip_background =  VideoFileClip(os.path.join(baseDir,"background","result.mp4")).margin(margin)
    txt_clip = TextClip("Background", fontsize = fontSize,stroke_width=2.0,method='caption',stroke_color='red').set_duration(clip_background.duration)
    txt_clip.set_pos(("left","bottom"))
    clip_background = CompositeVideoClip([clip_background, txt_clip]) 

    clip_orig =  VideoFileClip(os.path.join(baseDir,"orig_images_scaled","result.mp4")).margin(margin)
    txt_clip = TextClip("Original", fontsize = fontSize,stroke_width=2.0,method='caption',stroke_color='red').set_duration(clip_background.duration)
    txt_clip.set_pos(("left","bottom"))
    clip_orig = CompositeVideoClip([clip_orig, txt_clip]) 

    clip_wireframe =  VideoFileClip(os.path.join(baseDir,"orig_images_scaled_output_empty_background","result.mp4")).margin(margin)
    txt_clip = TextClip("Wireframes", fontsize = fontSize,stroke_width=2.0,method='caption',stroke_color='red').set_duration(clip_background.duration)
    txt_clip.set_pos(("left","bottom"))
    clip_wireframe = CompositeVideoClip([clip_wireframe, txt_clip]) 


    clip_output =  VideoFileClip(os.path.join(baseDir,"orig_images_scaled_output","result.mp4")).margin(margin)
    txt_clip = TextClip("Output", fontsize = fontSize,stroke_width=2.0,method='caption',stroke_color='red').set_duration(clip_background.duration)
    txt_clip.set_pos(("left","bottom"))
    clip_output = CompositeVideoClip([clip_output, txt_clip]) 


    final_clip = clips_array([[clip_orig, clip_background],
                            [clip_wireframe, clip_output]])
    final_clip.write_videofile(F"{baseDir}/finalVideo.mp4")    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, help='input video path', required=False, default=None)
    parser.add_argument('--images_dir', type=str, help='input images dir', required=False, default=None)
    parser.add_argument('--output_dir', type=str, help="output dir to save processed files", required=True)
    parser.add_argument('--tmp_dir', type=str, help='temp dir to save the intermediate results', default='tmp')
    parser.add_argument('--frame_rate',type=str,help='frame rate of original dataset', default='20')
    parser.add_argument('--process',type=str,help='specify the process : main -> generate privacy enhanced, blurred and pixelated images', default='')
    
    args = parser.parse_args()
    if args.process == "main" :
        main(args)
    elif args.process == "extract" :
        calculteSIValues(args)
    elif args.process == "":
        print("Please specify the process")

