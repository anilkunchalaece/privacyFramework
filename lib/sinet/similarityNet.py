# saimese network with denseNet backbone
import torch
from torchvision import models
import torch.nn as nn

class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()

        self.net = models.resnet18(pretrained=True)

        # freeze layers of pre-trained model
        for param in self.net.parameters() :
            param.requires_grad = False

        fc_inp = self.net.fc.in_features

        self.net.fc = nn.Sequential(
            nn.Linear(fc_inp, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 200),
        )

    def forward_one(self,x):
        if list(x.shape) == 3 : # f arg is single image, add extra dimension
            x = x.unsqueeze(0)
        x = self.net(x)
        return x

    def forward(self,anchor,positive,negative):
        a_out = self.forward_one(anchor)
        p_out = self.forward_one(positive)
        n_out = self.forward_one(negative)
        return a_out,p_out,n_out



if __name__ == "__main__":
    net = SimilarityNet()
    print(net)