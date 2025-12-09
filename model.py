'''
File: model.py
Author: Abdurahman Mohammed
Date: 2024-09-05
Description: A Python script that defines a PyTorch model for the cell counting task.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Creating the model.
class CellCounter(nn.Module):
    '''
    A simple CNN model for cell counting. It uses 8 convolutional layers and 2 fully connected layers. In between the convolutional layers, there are max pooling layers to reduce the spatial dimensions of the feature maps. 
    This model assumes that the input images are of size 256x256. 
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

# -----------------------------------------------------------------------------
# Residual block (BN-ReLU-Conv style, like paper's RB) :contentReference[oaicite:1]{index=1}
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Simple residual block:
      x -> BN -> ReLU -> Conv(k) -> BN -> ReLU -> Conv(k) + skip
    Channels are preserved (no projection).
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding=padding, bias=False)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity

# -----------------------------------------------------------------------------
# Multi-column, multi-resolution front-end (approx. Table 1 in paper) :contentReference[oaicite:2]{index=2}
# -----------------------------------------------------------------------------
class MultiColumnEncoder(nn.Module):
    """
    Multi-column & multi-resolution front-end inspired by Jiang & Yu:
      - Branch A: bigger receptive field with 5x5 then 3x3 convs.
      - Branch B: starts with pooling and uses 3x3 convs.
      - Branch C: deeper pooled features with 1x1 convs.

    Returns:
      feat_l2: features at 1/2 resolution (H/2, W/2)   [B, C, H/2, W/2]
      feat_l3: features at 1/4 resolution (H/4, W/4)   [B, C, H/4, W/4]
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.base_channels = base_channels
        self.pool = nn.MaxPool2d(2, 2)

        # Branch A (roughly following Table 1: 5x5 -> 5x5 -> 5x5 -> pool -> 3x3 -> 3x3 -> pool)
        self.a_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2)
        self.a_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=5, padding=2)
        self.a_conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=5, padding=2)
        self.a_conv4 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.a_conv5 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Branch B (pool -> 3x3 x4 -> pool)
        self.b_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.b_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.b_conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.b_conv4 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Branch C (pooled twice -> 1x1 x2)
        self.c_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.c_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        # Merging convs (reduce concatenated channels back to base_channels)
        self.merge_l2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.merge_l3 = nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # -------- Branch A --------
        a = F.relu(self.a_conv1(x))
        a = F.relu(self.a_conv2(a))
        a = F.relu(self.a_conv3(a))        # H x W
        a_half = self.pool(a)              # H/2 x W/2
        a_half = F.relu(self.a_conv4(a_half))
        a_half = F.relu(self.a_conv5(a_half))
        a_quarter = self.pool(a_half)      # H/4 x W/4

        # -------- Branch B --------
        b = self.pool(x)                   # H/2 x W/2
        b = F.relu(self.b_conv1(b))
        b = F.relu(self.b_conv2(b))
        b = F.relu(self.b_conv3(b))
        b_half = F.relu(self.b_conv4(b))   # H/2 x W/2
        b_quarter = self.pool(b_half)      # H/4 x W/4

        # -------- Branch C --------
        c_half = self.pool(x)              # H/2 x W/2
        c_quarter = self.pool(c_half)      # H/4 x W/4
        c_quarter = F.relu(self.c_conv1(c_quarter))
        c_quarter = F.relu(self.c_conv2(c_quarter))

        # -------- Merge multi-column features at L2 (H/2) and L3 (H/4) --------
        feat_l2 = torch.cat([a_half, b_half], dim=1)           # [B, 2C, H/2, W/2]
        feat_l2 = F.relu(self.merge_l2(feat_l2))               # [B, C, H/2, W/2]

        feat_l3 = torch.cat([a_quarter, b_quarter, c_quarter], dim=1)  # [B, 3C, H/4, W/4]
        feat_l3 = F.relu(self.merge_l3(feat_l3))                        # [B, C, H/4, W/4]

        return feat_l2, feat_l3


# -----------------------------------------------------------------------------
# MM-Net style U-Net for cell counting (density-regression backbone)
# -----------------------------------------------------------------------------
class MMNetCellCounter(nn.Module):
    """
    MM-Net-style cell counter based on:
      - U-Net-like encoder-decoder
      - Residual blocks in encoder and decoder
      - Multi-column multi-resolution encoder front-end (MultiColumnEncoder)

    Forward returns:
      counts: (B, num_classes) non-negative image-level counts
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Multi-column front-end
        self.multi_column = MultiColumnEncoder(in_channels=in_channels,
                                               base_channels=base_channels)

        # Encoder path (single backbone, residual blocks)
        # Level 1: use larger kernel (5x5) for the first residual block, as in paper.
        self.enc_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2)
        self.enc_rb1 = ResidualBlock(base_channels, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2)

        self.enc_rb2 = ResidualBlock(base_channels, kernel_size=3)
        self.enc_rb3 = ResidualBlock(base_channels, kernel_size=3)
        self.enc_rb4 = ResidualBlock(base_channels, kernel_size=3)

        # Decoder path (transpose conv upsampling + residual blocks)
        self.up3 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec3_conv = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.dec3_rb = ResidualBlock(base_channels, kernel_size=3)

        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec2_conv = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.dec2_rb = ResidualBlock(base_channels, kernel_size=3)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec1_conv = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.dec1_rb = ResidualBlock(base_channels, kernel_size=3)

        # Final 1x1 conv to density maps (pixel-wise regression)
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        # Non-negative density
        self.out_act = nn.Softplus(beta=1.0)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init similar to your SwitchCNN
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 3, H, W)  [H,W multiple of 8 recommended, e.g. 256x256]

        Returns:
          counts: (B, num_classes)
        """
        # Multi-column features at L2 (H/2) and L3 (H/4)
        feat_l2, feat_l3 = self.multi_column(x)

        # -------- Encoder --------
        # Level 1
        x1 = F.relu(self.enc_conv1(x))
        x1 = self.enc_rb1(x1)                     # [B, C, H, W]

        # Level 2 (H/2)
        x2 = self.pool(x1)
        x2 = x2 + feat_l2                         # inject multi-column features
        x2 = self.enc_rb2(x2)                     # [B, C, H/2, W/2]

        # Level 3 (H/4)
        x3 = self.pool(x2)
        x3 = x3 + feat_l3                         # inject deeper multi-column features
        x3 = self.enc_rb3(x3)                     # [B, C, H/4, W/4]

        # Bottleneck (H/8)
        x4 = self.pool(x3)
        x4 = self.enc_rb4(x4)                     # [B, C, H/8, W/8]

        # -------- Decoder --------
        # Up from bottleneck to level 3
        d3 = self.up3(x4)                         # [B, C, H/4, W/4]
        d3 = torch.cat([d3, x3], dim=1)           # skip connection
        d3 = F.relu(self.dec3_conv(d3))
        d3 = self.dec3_rb(d3)                     # [B, C, H/4, W/4]

        # Up to level 2
        d2 = self.up2(d3)                         # [B, C, H/2, W/2]
        d2 = torch.cat([d2, x2], dim=1)
        d2 = F.relu(self.dec2_conv(d2))
        d2 = self.dec2_rb(d2)                     # [B, C, H/2, W/2]

        # Up to level 1 (full resolution)
        d1 = self.up1(d2)                         # [B, C, H, W]
        d1 = torch.cat([d1, x1], dim=1)
        d1 = F.relu(self.dec1_conv(d1))
        d1 = self.dec1_rb(d1)                     # [B, C, H, W]

        # -------- Density & count --------
        density = self.out_conv(d1)               # [B, num_classes, H, W]
        density = self.out_act(density)           # non-negative

        # Integrate density map to get counts
        counts = density.sum(dim=(2, 3))          # [B, num_classes]

        return counts