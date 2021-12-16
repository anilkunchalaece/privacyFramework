# Towards A Framework for Privacy-Preserving Pedestrian Analysis 

## Things TODO
For given video / set of images
1. Run VIBE to extract the all the 3D wireframes of people present in the images
2. Use semantic segmentation / MASK-RNN to extract the masks for all the people in the images / video
3. Using Original images and masks run STTN (Digital Inpainiting) to remove people/ extract background
4. Render both extracted 3D wireframes of people and background
5. Test both of the datasets ( Original and Privacy preserving ) for action detection / object detection ?

## Deadends
- AttrNet is not working. Unable to train beyond 30% accuracy irrespective of epochs and batchsize. Seems to be issue with image size
- Face detection is not working with standard face detection algorithms ( like MTCNN from facenet_pytorch) seems like wireframes does not have usual face charactiristics ( Most of the face detectors works by first detecting eyes, which are not very distinct in SMPL/Wireframe model). I think size is not an issue with face detector

## Improvements 
DeepSort for Tracking
- Look into [deepsort](https://github.com/nwojke/deep_sort) to track people/pedestrians across frames 
- Integrate DeepSort and Mask-RCNN for video semantic segmentation

VIBE with Openpose
- Currently I'm unable to run VIBE using openpose. Currently VIBE is using multiperson tracker ( with SORT to track people ).

#### Object detection evaluation
ref - https://eavise.gitlab.io/brambox/notes/02-getting_started.html

### Face detection
ref - https://github.com/deepinsight/insightface

### Extra tool needed
- ffmpeg - to convert video to images
- convert / image magick - used for batch image resizing
    - Why two ? I'm lazy :)
