# Pedestrian Attribute Measurement Net
import torch
from torchvision import models
import torch.nn as nn 


class AttributeNet(nn.Module):
    def __init__(self):
        super(AttributeNet,self).__init__()
        
        self.net = models.resnet152(pretrained=True)

        #freeze layers of pre-trained model
        for param in self.net.parameters():
            param.requires_grad = False
        
        fc_inp = self.net.fc.in_features

        self.net.fc = nn.Sequential(
            nn.Linear(fc_inp, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 37),
        )

    def forward(self,x):
        return self.net(x)


if __name__ == "__main__" :
    net = AttributeNet()
    print(net)