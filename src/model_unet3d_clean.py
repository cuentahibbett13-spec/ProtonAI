"""
3D UNet simple y eficiente para denoising de dosis en protonterapia.
Input: (2, Z, Y, X) - [noisy_dose, density_map]
Output: (1, Z, Y, X) - predicted_dose
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Conv3D → BatchNorm → ReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownBlock3D(nn.Module):
    """Downsample: Conv → Conv → MaxPool"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x_skip = x.clone()
        x = self.pool(x)
        return x, x_skip


class UpBlock3D(nn.Module):
    """Upsample: ConvTranspose → Concat skip → Conv → Conv"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock3D(out_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet3D(nn.Module):
    """
    3D UNet con skip connections.
    Input: (B, 2, Z, Y, X)
    Output: (B, 1, Z, Y, X)
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        
        f = base_filters
        
        # Encoder (downsampling)
        self.enc1 = DownBlock3D(in_channels, f)
        self.enc2 = DownBlock3D(f, f*2)
        self.enc3 = DownBlock3D(f*2, f*4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock3D(f*4, f*8),
            ConvBlock3D(f*8, f*8),
        )
        
        # Decoder (upsampling)
        self.dec3 = UpBlock3D(f*8, f*4, f*4)
        self.dec2 = UpBlock3D(f*4, f*2, f*2)
        self.dec1 = UpBlock3D(f*2, f, f)
        
        # Output
        self.out = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        
        # Bottleneck
        x_bot = self.bottleneck(x3)
        
        # Decoder
        x_up3 = self.dec3(x_bot, skip3)
        x_up2 = self.dec2(x_up3, skip2)
        x_up1 = self.dec1(x_up2, skip1)
        
        # Output
        out = self.out(x_up1)
        
        return out


if __name__ == "__main__":
    # Test forward pass
    model = UNet3D(in_channels=2, out_channels=1, base_filters=32)
    x = torch.randn(1, 2, 64, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
