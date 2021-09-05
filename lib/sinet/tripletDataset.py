from torch.utils.data import Dataset
import torch
import PIL
import os
from torchvision.transforms import transforms

class TripletDataset(Dataset):
    def __init__(self,triplets,transform):
        self.triplets = triplets
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        anchor,positive,negative = self.triplets[idx]

        anchorImg = PIL.Image.open(anchor).convert('RGB')
        positiveImg = PIL.Image.open(positive).convert('RGB')
        negativeImg = PIL.Image.open(negative).convert('RGB')

        return {
            "anchorImg" : self.transform(anchorImg),
            "positiveImg" : self.transform(positiveImg),
            "negativeImg" : self.transform(negativeImg)
        }




if __name__ == "__main__" :
    from image_pairs import ImagePairGen
    from similarityNet import SimilarityNet
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt

    root_dir = "/home/akunchala/Downloads/MARS_Dataset/bbox_train"
    p = ImagePairGen(root_dir,limit_ids=20 ,max_frames=None)
    triplets = p.generateTripletsRandomly()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(degrees=45)])

    dataset = TripletDataset(triplets, transform)

    net = SimilarityNet()

    out = net(dataset[0]["anchorImg"],dataset[0]["positiveImg"],dataset[0]["negativeImg"])
    print(out[1].shape)

    # visualize first triplet in dataset
    # _,ax = plt.subplots(1,3)
    # ax[0].imshow(dataset[0]["anchorImg"])
    # ax[0].text(0,0,"Anchor Img",color='r')
    # ax[1].imshow(dataset[0]["positiveImg"])
    # ax[1].text(0,0,"Positive Img",color='r')
    # ax[2].imshow(dataset[0]["negativeImg"])
    # ax[2].text(0,0,"Negative Img",color='r')
    # plt.show()

