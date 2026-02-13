import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            depth = min(x.shape[-3], skip.shape[-3])
            height = min(x.shape[-2], skip.shape[-2])
            width = min(x.shape[-1], skip.shape[-1])
            x = x[:, :, :depth, :height, :width]
            skip = skip[:, :, :depth, :height, :width]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PhysicsAwareUNet3D(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock3D(in_channels, c1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.enc2 = ConvBlock3D(c1, c2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = ConvBlock3D(c3, c4)

        self.up3 = UpBlock3D(c4, c3, c3)
        self.up2 = UpBlock3D(c3, c2, c2)
        self.up1 = UpBlock3D(c2, c1, c1)

        self.out_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out_conv(d1)
