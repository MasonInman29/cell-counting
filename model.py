'''
File: model.py
Author: Abdurahman Mohammed
Date: 2024-09-05
Description: A Python script that defines a PyTorch model for the cell counting task.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# Creating the model.
class CellCounter(nn.Module):
    '''
    A simple CNN model for cell counting. It uses 8 convolutional layers and 2 fully connected layers. In between the convolutional layers, there are max pooling layers to reduce the spatial dimensions of the feature maps. This model assumes that the input images are of size 256x256. If you are working with images of a different size, you may need to adjust the model architecture accordingly (Fully connected layers will be different or you will have to add an adaptive pooling layer).
    '''

    def __init__(self):
        super(CellCounter, self).__init__()
        # Convolutional layers.
        #'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] -> Adapted from VGG-16. Check configuration A in the VGG paper. Note that this is not a pretrained model.
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)

        self.counter = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        
        # Feature extraction part
        x = F.relu(self.conv1(x))
        x = self.pooling(x)
        x = F.relu(self.conv2(x))
        x = self.pooling(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pooling(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pooling(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pooling(x)

        # At this point, we have a tensor of shape (batch_size, 512, 8, 8).

        # Flatten the tensor.
        x = x.view(x.size(0), -1)

        # Fully connected layers to get the count as a scalar value.
        x = self.counter(x)

        return x
