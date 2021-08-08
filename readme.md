# Towards A Framework for Privacy-Preserving Pedestrian Analysis 

## Things TODO
For given video / set of images
1. Run VIBE to extract the all the 3D wireframes of people present in the images
2. Use semantic segmentation / MASK-RNN to extract the masks for all the people in the images / video
3. Using Original images and masks run STTN (Digital Inpainiting) to remove people/ extract background
4. Render both extracted 3D wireframes of people and background
5. Test both of the datasets ( Original and Privacy preserving ) for action detection / object detection ?

## Improvements 
DeepSort for Tracking
- Look into [deepsort](https://github.com/nwojke/deep_sort) to track people/pedestrians across frames 
- Integrate DeepSort and Mask-RCNN for video semantic segmentation

VIBE with Openpose
- Currently I'm unable to run VIBE using openpose. Currently VIBE is using multiperson tracker ( with SORT to track people ).