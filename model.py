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


class _Column(nn.Module):
    """A single column that is resolution-agnostic and outputs (B, C) counts."""
    def __init__(self, k1: int, k2: int, k3: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=k1, padding=k1 // 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=k2, padding=k2 // 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=k3, padding=k3 // 2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # <- makes it work for any HxW
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)  # (B, C)


class _Router(nn.Module):
    """Tiny router with a learnable temperature."""
    def __init__(self, init_tau: float = 1.5, K: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, K)
        )
        # learnable temperature in log-space to keep it > 0
        self.log_tau = nn.Parameter(torch.log(torch.tensor([init_tau], dtype=torch.float32)))

    def forward(self, x):
        logits = self.net(x)  # (B, K)
        tau = torch.clamp(self.log_tau.exp(), 0.5, 5.0)
        probs = torch.softmax(logits / tau, dim=1)  # (B, K)
        return logits, probs, tau


class SwitchCNN(nn.Module):
    """
    Image-level Switch-CNN with three columns and a temperature-controlled router.
    Returns:
      counts: (B, C) non-negative per-class counts
      probs:  (B, 3) router probabilities
      tau:    scalar temperature tensor
      logits: (B, 3) router logits (for CE)
      cols:   (B, 3, C) raw per-column counts (pre-mix, pre-Softplus)
    """
    def __init__(self, num_classes: int = 1, K: int = 3, init_tau: float = 1.5):
        super().__init__()
        assert K == 3
        self.num_classes = num_classes
        self.col1 = _Column(3, 3, 3, num_classes)
        self.col2 = _Column(5, 3, 3, num_classes)
        self.col3 = _Column(7, 5, 3, num_classes)
        self.router = _Router(init_tau=init_tau, K=K)
        self.out_act = nn.Softplus(beta=1.0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        o1 = self.col1(x)               # (B, C)
        o2 = self.col2(x)
        o3 = self.col3(x)
        cols = torch.stack([o1, o2, o3], dim=1)  # (B, 3, C)

        logits, probs, tau = self.router(x)      # (B,3),(B,3),scalar
        mixed = (probs[:, 0:1] * o1 +
                 probs[:, 1:2] * o2 +
                 probs[:, 2:3] * o3)             # (B, C)

        counts = self.out_act(mixed)             # enforce ≥0
        return counts, probs, tau, logits, cols
