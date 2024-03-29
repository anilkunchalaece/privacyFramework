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
import matplotlib.pyplot as plt

from getMaksFromImage import GetMask
from lib.ppfp.config import getConfig
from lib.ppfp.insightFace import InsightFaceDetector
from lib.ppfp.personDetector import PersonDetector
from lib.ppfp.keyPointDetector import KeyPointDetector
from lib.ppfp.annonImgs import AnnonImgs
from lib.ppfp.extractBbox import BBoxExtractor
import sinet_train as sinet
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lib.ppfp.cocoConverter import COCOConverter 


from multiprocessing import Pool
from moviepy.editor import VideoFileClip, TextClip, clips_array, vfx,CompositeVideoClip

c = {
    "src" : True,
    "pred" : True,
    "background" : True,
    "detections" : True,
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
    
    blurRange = range(2,100,2)
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
    
    runSegmentation(args)
    extractBBoxes(args)
    calculteSIValues(args)
    calculateFinalValues(args)


def extractBBoxes(args):
    tmpDir = args.tmp_dir
    gtImgsLocation = os.path.join(tmpDir,"src","orig_images_scaled")
    predImgsLocation = os.path.join(tmpDir,"pred","orig_images_scaled_output")
    gtDetFile = os.path.join(tmpDir,"src","person_det.json")
    predDetFile = os.path.join(tmpDir,"pred","person_det.json")
    annonDir = os.path.join(tmpDir,"annonBodyImgs")
    segmDir = os.path.join(tmpDir,"segmentation","maskWithBackground")
    neuralArtDir = os.path.join(tmpDir,"neuralArt","outWithBackground")

    bExtractor = BBoxExtractor(gtImgsLocation,predImgsLocation,annonDir,gtDetFile,predDetFile,segmDir,neuralArtDir)
    bExtractor.extractAllBboxes()

def calculteSIValues(args):
    tmpDir = args.tmp_dir
    annonDir = os.path.join(tmpDir,"annonBodyImgs")

    gtImgsDir = os.path.join(annonDir,"gt","bbox")
    predImgsDir = os.path.join(annonDir,"pred","bbox")
    segmentationDir = os.path.join(annonDir,"segm","bbox")
    neuralArtDir = os.path.join(annonDir,"neuralArt","bbox")

    pred_v = sinet.evalResults(gtImgsDir,predImgsDir)
    segm_v = sinet.evalResults(gtImgsDir,segmentationDir)
    na_v = sinet.evalResults(gtImgsDir,neuralArtDir)

    
    blur = dict()
    # for d in os.listdir(os.path.join(annonDir,"blur")) :
    #     _c_si = sinet.evalResults(gtImgsDir,os.path.join(annonDir,"blur",d,"bbox"))
    #     blur[d] = _c_si
    
    pix = dict()
    # for d in os.listdir(os.path.join(annonDir,"pix")) :
    #     _c_si = sinet.evalResults(gtImgsDir,os.path.join(annonDir,"pix",d,"bbox"))
    #     pix[d] = _c_si

    outFile = os.path.join(annonDir,"sinetValues2.json")
    with open(outFile,"w") as fd:
        json.dump({
            "pred" : pred_v,
            "segm" : segm_v,
            "na_v" : na_v,
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

def calculateFinalValues(args):
    src_dir = os.path.join(args.tmp_dir,"src")
    pred_dir = os.path.join(args.tmp_dir,"pred")
    segm_dir = os.path.join(args.tmp_dir,"segmentation")
    na_dir = os.path.join(args.tmp_dir,"neuralArt")

    annonImgsDir = os.path.join(args.tmp_dir,"annonBodyImgs")
    sinetFile = os.path.join(annonImgsDir,"sinetValues.json")
    
    PERSON_DET_FILE = "person_det.json"
    KEYPOINT_DET_FILE = "keypoint_det.json"

    resDict = {}

    srcFile_pd = os.path.join(src_dir,PERSON_DET_FILE)
    sObj_pd = COCOConverter(srcFile_pd)
    srcFile_pd = sObj_pd.convert()
    predFile_pd = os.path.join(pred_dir,PERSON_DET_FILE)

    srcFile_kd = os.path.join(src_dir,KEYPOINT_DET_FILE)
    sObj_kd = COCOConverter(srcFile_kd)
    srcFile_kd = sObj_kd.convert()
    predFile_kd = os.path.join(pred_dir,KEYPOINT_DET_FILE)

    segmFile_pd = os.path.join(segm_dir,PERSON_DET_FILE)
    segmFile_kd = os.path.join(segm_dir,KEYPOINT_DET_FILE)

    naFile_pd = os.path.join(na_dir,PERSON_DET_FILE)
    naFile_kd = os.path.join(na_dir,KEYPOINT_DET_FILE)

    pred_pd = getCocoEval(srcFile_pd,predFile_pd,"bbox")
    pred_kd = getCocoEval(srcFile_kd,predFile_kd,"keypoints")

    segm_pd = getCocoEval(srcFile_pd,segmFile_pd,"bbox")
    segm_kd = getCocoEval(srcFile_kd,segmFile_kd,"keypoints")

    na_pd = getCocoEval(srcFile_pd,naFile_pd,"bbox")
    na_kd = getCocoEval(srcFile_kd,naFile_kd,"keypoints")

    #load sinetValues
    with open(sinetFile) as fd:
        siNetValues = json.load(fd)
        
    # print(pred_pd)
    # print(pred_kd)
    # for d in os.listdir(os.path.join(annonImgsDir,"blur")) :
    #     print(F" ===> print running for {d}")
    #     resDict[d] = {
    #         "blur" : {
    #             "personDetector" : getCocoEval(srcFile_pd, os.path.join(annonImgsDir,"blur",d,PERSON_DET_FILE), "bbox"),
    #             "keypointDetector" : getCocoEval(srcFile_kd, os.path.join(annonImgsDir,"blur",d,KEYPOINT_DET_FILE), "keypoints"),
    #             "similarityIndex" :  siNetValues["blur"][d]
    #         },
    #         "pix" : {
    #                 "personDetector" : getCocoEval(srcFile_pd, os.path.join(annonImgsDir,"pix",d,PERSON_DET_FILE), "bbox"),
    #                 "keypointDetector" : getCocoEval(srcFile_kd, os.path.join(annonImgsDir,"pix",d,KEYPOINT_DET_FILE), "keypoints"),
    #                 "similarityIndex" :  siNetValues["pix"][d]
    #             }
    #         }


    resDict["wireframe"] = {
        "personDetector" : pred_pd,
        "keypointDetector" : pred_kd,
        "similarityIndex" : siNetValues["pred"]
    }

    resDict["segmentation"] = {
        "personDetector" : segm_pd,
        "keypointDetector" : segm_kd,
        "similarityIndex" : siNetValues["segm"]
    }

    resDict["neuralArt"] = {
        "personDetector" : na_pd,
        "keypointDetector" : na_kd,
        "similarityIndex" : siNetValues["na_v"]
    }

    with open(os.path.join(args.tmp_dir,"results2.json"),'w') as fw :
        json.dump(resDict,fw)

def runSegmentation(args):
    m = GetMask()
    createDir(os.path.join(args.tmp_dir,"segmentation","maskWithBackground"))

    m.generateMasks(os.path.join(args.tmp_dir,"src","orig_images_scaled"),
                    os.path.join(args.tmp_dir,"segmentation","maskWithBackground"),
                    os.path.join(args.tmp_dir,"background"))
    
    srcDir = os.path.join(args.tmp_dir,"segmentation","maskWithBackground")
    desDir = os.path.join(args.tmp_dir,"segmentation")

    detectPersons(srcDir,desDir,"person_det.json")
    detectKeyPoints(srcDir,desDir,"keypoint_det.json")

def runNeuralArt(args):
    srcDir = os.path.join(args.tmp_dir,"neuralArt","outWithBackground")
    desDir = os.path.join(args.tmp_dir,"neuralArt")

    detectPersons(srcDir,desDir,"person_det.json")
    detectKeyPoints(srcDir,desDir,"keypoint_det.json")


def visualizeResults(args):
    srcFile = os.path.join(args.tmp_dir,"results.json")
    try :
        with open(srcFile) as fd:
            data = json.load(fd)
    except Exception as E:
        print(F"Unable to load the results file from {srcFile}, exited with Exception {E}")
        raise E
    
    # Plot the blur utility vs privacy
    blur = []
    for i in data.keys() :
        if i != "wireframe" :
            if type(data[i]["blur"]["personDetector"]) == list :
                try :
                    _pd_ap = data[i]["blur"]["personDetector"][5]
                    _pd_ar = data[i]["blur"]["personDetector"][-1]
                    _pd_f1 = (2.0*(_pd_ap*_pd_ar))/(_pd_ap+_pd_ar)
                    _kd_ap = data[i]["blur"]["keypointDetector"][4]
                    _kd_ar = data[i]["blur"]["keypointDetector"][-1]
                    _kd_f1 = 2*(_kd_ap*_kd_ar)/(_kd_ap+_kd_ar)
                    print(F"{i} => _pd_f1 : {_pd_f1}     _kd_f1 : {_kd_f1}")
                    blur.append({
                        "idx" : int(i),
                        "utility" : 0.5*(_pd_f1 + _kd_f1),
                        "privacy" : 1 - data[i]["blur"]["similarityIndex"]
                    })
                except :
                    print(F"unable to capture for {i}")
        else :
            _wp_ap = data[i]["personDetector"][5]
            _wp_ar = data[i]["personDetector"][-1]
            _wp_f1 = (2.0*(_wp_ap*_wp_ar))/(_wp_ap+_wp_ar)
            _wk_ap = data[i]["keypointDetector"][4]
            _wk_ar = data[i]["keypointDetector"][-1]
            _wk_f1 = (2.0*(_wk_ap*_wk_ar))/(_wk_ap+_wk_ar)
            _w_pr = data[i]["similarityIndex"]
            print(F"{i} => _wp_f1 : {_wp_f1}     _wk_f1 : {_wk_f1}    si: {_w_pr}")
    blur = sorted(blur,key=lambda x:x["idx"], reverse=False)
    sizeToPlot = 11
    idx = [x["idx"] for x in blur ][:sizeToPlot]
    utility = [x["utility"] for x in blur][:sizeToPlot]
    privacy = [x["privacy"] for x in blur][:sizeToPlot]

    fig, ax = plt.subplots()
    plt.xlabel("Privacy Metric")
    plt.ylabel("Utility Metric")
    ax.plot(privacy,utility,"-*",label="blur")
    # ax.plot(privacy,utility)
    for i, txt in enumerate(idx):
        ax.annotate(txt, (privacy[i]+0.002, utility[i]+0.002))
    plt.show()


def getCocoEval(gtFile, resFile, key) :
    try :
        cocoGt=COCO(gtFile)
        cocoDt=cocoGt.loadRes(resFile)
        cocoEval = COCOeval(cocoGt,cocoDt,key)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()    
        return cocoEval.stats.tolist()
    except Exception as e :
        print("################## EXCEPTION ############")
        print(str(e))
        # raise e
        return 0


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
    cmd = ["ffmpeg","-i",srcFile,"-vf","fps=10",F"{dirToStoreImages}/%06d.png"]
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
        extractBBoxes(args)
        calculteSIValues(args)
    elif args.process == "final" :
        calculateFinalValues(args)
    elif args.process == "plot" :
        visualizeResults(args)
    elif args.process == "segmentation":
        runSegmentation(args)
    elif args.process == "neuralArt":
        runNeuralArt(args)
    elif args.process == "":
        print("Please specify the process")

