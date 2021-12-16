# saimese network with denseNet backbone
import torch
from torchvision import models
import torch.nn as nn

class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()

        self.net = models.resnet50(pretrained=True)

        # freeze layers of pre-trained model
        for param in self.net.parameters() :
            param.requires_grad = False

        fc_inp = self.net.fc.in_features

        self.net.fc = nn.Sequential(
            nn.Linear(fc_inp, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),            
            nn.Linear(512, 300),
        )

    def forward_one(self,x):
        if list(x.shape) == 3 : # f arg is single image, add extra dimension
            x = x.unsqueeze(0)
        x = self.net(x)
        return x

    def forward(self,anchor,positive,negative=None):
        a_out = self.forward_one(anchor)
        p_out = self.forward_one(positive)
        if negative == None :
            return a_out,p_out
        else :
            n_out = self.forward_one(negative)
            return a_out,p_out,n_out



if __name__ == "__main__":
    net = SimilarityNet()
    print(net)
