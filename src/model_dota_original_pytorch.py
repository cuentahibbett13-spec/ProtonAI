"""
Reimplementación de DoTA original (Pastor-Serrano 2022) en PyTorch.
Arquitectura exacta: Conv encoder (2D) → Transformer 1D causal → Conv decoder
Input: (150, 24, 24) geometry + energy
Output: (150, 24, 24) dose
Loss: MSE puro (paper original)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding para tokens de profundidad."""
    def __init__(self, d_model: int, max_len: int = 151):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.tensor(10000.0).log() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1), :]


class ConvEncoder2D(nn.Module):
    """
    Encodea cada slice 2D (24x24) em um token de dimensión d_model.
    Aplica convoluciones 2D en cada slice.
    """
    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(32, d_model, kernel_size=kernel_size, padding=padding)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1, H, W) = (batch, slices, channels, height, width)
        B, L, C, H, W = x.shape
        
        # Procesa cada slice independientemente
        x_flat = x.view(B * L, C, H, W)
        x = F.relu(self.conv1(x_flat))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pool(x)  # (B*L, d_model, 1, 1)
        x = x.view(B, L, -1)  # (B, L, d_model)
        
        return x


class TransformerCausal(nn.Module):
    """Transformer causal 1D para procesamiento de secuencia de slices."""
    def __init__(self, d_model: int, num_heads: int = 16, num_layers: int = 1, ff_dim: int = 2048):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model, max_len=151)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = self.pos_enc(x)
        
        # Máscara causal (no atiende a posiciones futuras)
        L = x.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        
        x = self.transformer(x, src_mask=causal_mask)
        return x


class ConvDecoder2D(nn.Module):
    """
    Decodifica token de dimensión d_model a slice 2D (24x24).
    """
    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        
        self.linear = nn.Linear(d_model, 32 * 24 * 24)
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.ConvTranspose2d(16, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, D = x.shape
        
        x = self.linear(x)  # (B, L, 32*24*24)
        x = x.view(B * L, 32, 24, 24)
        
        x = F.relu(self.conv1(x))
        x = self.conv2(x)  # (B*L, 1, 24, 24)
        
        x = x.view(B, L, 1, 24, 24)
        return x


class DotaOriginalPyTorch(nn.Module):
    """
    DoTA original (Pastor-Serrano 2022) reimplementado en PyTorch.
    Input: geometry (B, 150, 24, 24) + energy (B, 1)
    Output: dose (B, 150, 24, 24)
    """
    def __init__(self, d_model: int = 36, num_heads: int = 16, num_layers: int = 1, 
                 kernel_size: int = 5, ff_dim: int = 2048):
        super().__init__()
        
        self.encoder = ConvEncoder2D(d_model, kernel_size=kernel_size)
        self.transformer = TransformerCausal(d_model, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)
        self.decoder = ConvDecoder2D(d_model, kernel_size=kernel_size)
        
        # Energy embedding
        self.energy_embedding = nn.Linear(1, d_model)

    def forward(self, geometry: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """
        geometry: (B, 150, 24, 24)
        energy: (B, 1)
        Returns: dose (B, 150, 24, 24)
        """
        B = geometry.size(0)
        
        # Agregar canal
        geometry = geometry.unsqueeze(2)  # (B, 150, 1, 24, 24)
        
        # Encodea slices
        tokens = self.encoder(geometry)  # (B, 150, d_model)
        
        # Aplica Transformer
        tokens = self.transformer(tokens)  # (B, 150, d_model)
        
        # Decodea slices
        dose = self.decoder(tokens)  # (B, 150, 1, 24, 24)
        
        # Remove channel dimension
        dose = dose.squeeze(2)  # (B, 150, 24, 24)
        
        return dose


if __name__ == "__main__":
    # Test
    model = DotaOriginalPyTorch(d_model=36, num_heads=16, num_layers=1)
    
    geometry = torch.randn(2, 150, 24, 24)
    energy = torch.randn(2, 1)
    
    output = model(geometry, energy)
    print(f"Input geometry: {geometry.shape}")
    print(f"Output dose: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
