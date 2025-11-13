""""
This is the U-Net architecture for satellite trail detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers with batch normalization and ReLU.
    It goes Input -> Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> Output
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """ Downscaling block """

    """ 
    This takes 2x2 blocks, keeps only the max value and then cuts size in half.
    It then applies DoubleConv to extract features
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    This is the upscaling block
    """

    """
    ConvTranspose2d: Upsamples - makes image bigger
    torch.cat: Combines the upsampled features with the skip connection
    Skip connection (x2): Brings back details from the encoder
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1 is the input from previous layer (lower resolution)
        x2 is the skip connection from encoder (same resolution as output)
        """
        x1 = self.up(x1)

        # Case where sizes don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    The actual U-Net architecture
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            features: List[int] = [64, 128, 256, 512]
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])

        # Encoder (downsampling path)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck (deepest point)
        self.down4 = Down(features[3], features[3] * 2)

        # Decoder (upsampling path)
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        # Output convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """ A forward pass through U-Net """

        # Encoder
        x1 = self.inc(x)  # 64 channels
        x2 = self.down1(x1)  # 128 channels
        x3 = self.down2(x2)  # 256 channels
        x4 = self.down3(x3)  # 512 channels

        # Bottleneck
        x5 = self.down4(x4)  # 1024 channels

        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512 channels
        x = self.up2(x, x3)  # 256 channels
        x = self.up3(x, x2)  # 128 channels
        x = self.up4(x, x1)  # 64 channels

        # Output
        logits = self.outc(x)

        # Apply sigmoid to get probabilities [0, 1]
        return torch.sigmoid(logits)

    def count_parameters(self):
        """ Count the number of trainable parameters. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks

    Dice loss measures the overlap between prediction and target.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        # This flattens so we can compute intersection and union
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice

class CombinedLoss(nn.Module):
    """ Combined binary cross-entropy and dice loss """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def test_unet():
    """ Test the U-Net architecture """
    import torch

    model = UNet(in_channels=3, out_channels=1)

    print(f"Model parameters: {model.count_parameters()}")

    # Test with random input
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test loss
    target = torch.randint(0, 2, (batch_size, 1, 512, 512)).float()
    criterion = CombinedLoss()
    loss = criterion(output, target)
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_unet()