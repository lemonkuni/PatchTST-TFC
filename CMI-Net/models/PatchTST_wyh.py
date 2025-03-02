import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange


def get_activation_fn(activation):
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class PatchTSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256,
                 dropout=0.1, activation="gelu",
                 n_layers=3):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        return x


class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x2 = self.norm1(x)
        x = x + self.dropout(self.attention(x2, x2, x2)[0])
        # Feed-forward
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class PatchTSTNet(nn.Module):
    def __init__(self, num_classes=5, patch_size=16, d_model=128, n_heads=8,
                 n_layers=3, d_ff=256, dropout=0.1, activation='gelu'):
        super().__init__()

        # 只保留加速度的patch embedding
        self.patch_size = patch_size
        self.d_model = d_model

        # 只需要一个patch embedding层
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=(1, patch_size), stride=(1, patch_size))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000 // patch_size, d_model))  # Adjust size as needed

        # 只需要一个transformer encoder
        self.encoder = PatchTSTEncoder(d_model, n_heads, d_ff, dropout, activation, n_layers)

        # 直接使用encoder输出进行分类，移除fusion层
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 只处理加速度数据
        x = x[:, :, :, 0:3]  # 只取加速度数据

        # Reshape and process data
        x = x.permute(0, 1, 3, 2)  # [batch, channel, features, time]
        B, C, F, T = x.shape
        x = x.reshape(B * C, F, T)  # Combine batch and channel dimensions
        x = x.unsqueeze(1)  # Add channel dimension for conv2d

        # Patch embedding
        x = self.patch_embed(x)  # [B*C, d_model, F, T//patch_size]

        # Reshape for transformer
        x = rearrange(x, '(b c) d f t -> (b c f) t d', b=B, c=C)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]

        # Apply transformer encoder
        x = self.encoder(x)

        # Global average pooling
        x = x.mean(dim=1)  # [B*C*F, d_model]

        # Reshape back
        x = x.view(B, -1, self.d_model).mean(dim=1)  # [B, d_model]

        # Classification
        output = self.classifier(x)  # [B, num_classes]

        return output


def PatchTST_wyh():
    """Returns a PatchTST model for time series classification.
    
    The model processes accelerometer and gyroscope data separately using patch embedding
    and transformer encoders, then fuses the features for classification.
    
    Returns:
        PatchTSTNet: A model instance with default parameters
    """
    return PatchTSTNet()


# Keep the old function name as an alias for backward compatibility
patchtst_wyh = PatchTST_wyh
