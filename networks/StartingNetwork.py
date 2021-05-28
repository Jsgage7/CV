import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

#600x800

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x

 

class CNN(nn.Module): #changed to resnet
    """
    Basic CNN to pass the data through
    """
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1])) #//cut off last layer
        self.resnet.eval() #//set on eval mode to freeze weights

        # #filter is 5, output channels is 6 (both can be changed)
        # self.conv1 = nn.Conv2d(input_channels, 6, 5)

        # #the filter dimmensions of the pooling layer (here 2x2, can be changed)
        # self.pool = nn.MaxPool2d(3, 3)


        # #16 output channels, filter still at 5
        # self.conv2 = nn.Conv2d (6, 12, 5)

        # self.pool2 = nn.MaxPool2d(3, 3)

        #         #16 output channels, filter still at 5
        # self.conv3 = nn.Conv2d (12, 24, 5)

        # self.pool3 = nn.MaxPool2d(2, 2)

        # self.conv4 = nn.Conv2d(24, 48, 5)
        # self.pool4 = nn.MaxPool2d(2,2)

        # #16 channels, not sure about 4x4
        self.fc = StartingNetwork(512, output_dim=5)

    def forward(self, x):

        with torch.no_grad(): 
            x = self.resnet(x)#freeze the weight 
           #//use like normal but use no_grad

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        # x = self.pool4(F.relu(self.conv4(x)))

        #keep the fc layers unchanged 
        x = self.fc.forward(x)
        return x