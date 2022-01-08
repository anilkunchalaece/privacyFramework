# Convert detections to COCO format
# Author : Kunchala Anil
import json

class COCOConverter:
    def __init__(self,inpFile):
        self.inpFile = inpFile
        self.outFile = inpFile.split(".")[0]+"_coco"+".json"
        self. annId = 0
    
    def getImageFormat(self,image_id):
        fName = F"{image_id}.png"
        return {
            "id" :image_id,
            "file_name" : fName,
            "width" : 432,
            "height" : 240
        }

    def imgs(self,obj) :
        return obj["image_id"]
    
    def getAnnotationFormat(self,obj):
        self.annId = self.annId + 1
        if obj.get("keypoints",0) != 0  : # if obj has keypoints
            return {
                # "keypoints" : obj["keypoints"],
                "keypoints" : [2 if x==1 else x for x in obj["keypoints"]],
                "num_keypoints" : len(obj["keypoints"])/3,
                "bbox" : obj["bbox"],
                "iscrowd" : 0,
                "category_id" : 1,
                "id" : self.annId,
                "image_id" : obj["image_id"],
                "area" : obj["area"]
            }
        else : 
            return {
                # "keypoints" : obj["keypoints"],
                # "keypoints" : [2 if x==1 else x for x in obj["keypoints"]],
                # "num_keypoints" : len(obj["keypoints"])/3,
                "bbox" : obj["bbox"],
                "iscrowd" : 0,
                "category_id" : 1,
                "id" : self.annId,
                "image_id" : obj["image_id"],
                "area" : obj["area"]
            }            

    def getCategoryFormat(self) :
        return [{
            "id": 1,
            "name": "person",
            "supercategory": "pedestrian"
        }
        # ,{
        #     "id": 1,
        #     "name": "other",
        #     "supercategory": "background"
        # }
        ]

    def convert(self):
        out = {}
        with open(self.inpFile) as fd:
            data = json.load(fd)
            print(F"total values are {len(data)}")

            anns = []
            imgs = []
            for v in data :
                # if v["score"] >= 0.75 :
                anns.append(self.getAnnotationFormat(v))
                imgs.append(self.imgs(v))
            
            imgs = list(set(imgs))
            imgsF = [self.getImageFormat(im) for im in imgs]
            print(F"total imgs are {len(imgsF)}")

            out["images"] = imgsF
            out["annotations"] = anns
            out["categories"] = self.getCategoryFormat()

            with open(self.outFile,"w") as fout :
                json.dump(out, fout)
                print(F"converted file is saved {self.outFile}")
            return self.outFile


if __name__ == "__main__":
    # inpFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/keyPoints/orig_keypoints.json"
    inpFile = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/annoImgs/gt/keypoint_det.json"
    cObj = COCOConverter(inpFile)
    cObj.convert()