# Last modified : 17 Oct 2021
# This script uses MTCNN to detect the faces in the given images.
# able to detect faces in the original image , but unable to detect faces in modified / wireframe images
# Ref - 
# https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
# https://medium.com/@iselagradilla94/how-to-build-a-face-detection-application-using-pytorch-and-opencv-d46b0866e4d6

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from retinaface import RetinaFace

class FaceDetector:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU or GPU
        self.mtcnn = MTCNN(device=self.device,keep_all=True,thresholds=[0.2,0.2,0.2])

    def detectFaces(self,imgs):
        # this function will detect the faces in the given image and return bboxes
        bboxes, probs, landmarks = self.mtcnn.detect(imgs,landmarks=True)
        print(bboxes)
        print(probs)
        return bboxes, probs, landmarks

    def _draw(self, frame, boxes, probs, landmarks=None):
        """
        Draw landmarks and boxes for each face detected
        """
        # for box in boxes :
        #     cv2.rectangle(frame,
        #                     (int(box[0]), int(box[1])),
        #                     (int(box[2]), int(box[3])),
        #                     (0, 0, 255),
        #                     thickness=2)            
        # return frame

        try:
            for box, prob in zip(boxes, probs):
                # Draw rectangle on frame
                if prob < 0.75 :
                    continue
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)

                # Show probability
                # cv2.putText(frame, str(
                    # prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except Exception as e:
            print(e)
            raise
            # pass

        return frame

    



if __name__ == "__main__" :
    # from deepface import DeepFace
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
    testImgPath = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_mot_16_08/orig_images_scaled/000001.png"
    # bbox = DeepFace.detectFace(img_path = testImgPath, detector_backend = backends[-1])
    # print(bbox.shape)
    # fd = FaceDetector()
    # img = cv2.imread(testImgPath)
    # print(img.shape)
    # color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(color_coverted.shape)
    # pil_image=Image.fromarray(color_coverted)
    # print(pil_image.mode)
    # print(pil_image.size)
    # b, p, l = fd.detectFaces(img)
    # o_img = fd._draw(cv2.imread(testImgPath),b,p,l)
    # cv2.imwrite("out.png",o_img)

    import cv2
    import numpy as np
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image

    app = FaceAnalysis(allowed_modules=["detection"])
    app.prepare(ctx_id=0,det_thresh=0.75)
    # img = ins_get_image(testImgPath)
    img = cv2.imread(testImgPath)
    faces = app.get(img)
    
    # print(faces[0]["embedding"].shape)
    # o_img = fd._draw(img,[faces[0]["bbox"]],None)

    # cv2.imwrite("out.png",o_img)
    
    # rimg = app.draw_on(img, faces)
    # cv2.imwrite("./t1_output.jpg", rimg)
