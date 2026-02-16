import math

import torch
import torch.nn as nn


class SliceEncoder2D(nn.Module):
    def __init__(self, in_channels: int, feat_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        feat_map = self.net(x)
        token = self.pool(feat_map).flatten(1)
        return feat_map, token


class SliceDecoder2D(nn.Module):
    def __init__(self, feat_channels: int, d_model: int):
        super().__init__()
        self.token_proj = nn.Linear(d_model, feat_channels)
        self.head = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_channels, 1, kernel_size=1),
        )

    def forward(self, feat_map: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat_map.shape
        cond = self.token_proj(token).view(b, c, 1, 1).expand(b, c, h, w)
        return self.head(torch.cat([feat_map, cond], dim=1))


class DoTAModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 32,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.slice_encoder = SliceEncoder2D(in_channels=in_channels, feat_channels=feat_channels)
        self.slice_token_proj = nn.Linear(feat_channels, d_model)
        self.energy_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.slice_decoder = SliceDecoder2D(feat_channels=feat_channels, d_model=d_model)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def _depth_positional_encoding(self, length: int, d_model: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, ct_volume: torch.Tensor, energy_mev: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = ct_volume.shape

        x2d = ct_volume.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        feat_map_2d, slice_token_2d = self.slice_encoder(x2d)

        slice_tokens = self.slice_token_proj(slice_token_2d).view(b, d, -1)
        energy_token = self.energy_proj(energy_mev).unsqueeze(1)

        seq = torch.cat([energy_token, slice_tokens], dim=1)
        pos = self._depth_positional_encoding(d + 1, seq.shape[-1], seq.device)
        seq = seq + self.pos_scale * pos.unsqueeze(0)

        mask = self._causal_mask(d + 1, seq.device)
        seq_out = self.transformer(seq, mask=mask)

        slice_ctx = seq_out[:, 1:, :].reshape(b * d, -1)
        pred_2d = self.slice_decoder(feat_map_2d, slice_ctx)

        pred = pred_2d.view(b, d, 1, h, w).permute(0, 2, 1, 3, 4)
        return torch.clamp(pred, min=0.0)
