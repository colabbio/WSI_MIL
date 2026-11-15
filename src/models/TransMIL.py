"""
TransMIL: Correlated Multiple Instance Learning with Transformer.

Reference:
  Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, et al.
  "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification."
  NeurIPS 2021.

Features:
  1. Input feature projection and position encoding (or PPEG: Pyramidal Position Encoding Generator).
  2. [CLS] token prepended to the sequence of patch tokens.
  3. Multi-Head Self-Attention layers modeling correlations among all patches.
  4. Classification / regression / survival output head from the final [CLS] token representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator (PPEG).
    Uses multi-scale depthwise/grouped convolutions to capture 2D spatial context
    from a pseudo-2D feature grid.
    """
    def __init__(self, dim: int = 512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C) where N = H * W
        """
        B, N, C = x.shape
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat_out = self.proj(feat) + self.proj1(feat) + self.proj2(feat)
        feat_out = feat_out.flatten(2).transpose(1, 2)
        return x + feat_out


class TransMILBlock(nn.Module):
    """
    Standard Transformer encoder block with Multihead Attention and MLP.
    """
    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransMILBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransMIL(nn.Module):
    """
    TransMIL architecture for Whole Slide Image representation.
    
    Args:
        input_dim (int): Input patch embedding dimension (default: 2048 for ResNet, 768 for ViT).
        hidden_dim (int): Transformer latent feature dimension (default: 512).
        n_classes (int): Number of output target classes (e.g. 5 for ISUP, 2 for binary, 1 for risk).
        depth (int): Number of Transformer layers (default: 2).
        num_heads (int): Number of attention heads (default: 8).
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        n_classes: int = 2,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(TransMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Linear projection from patch embedding to transformer hidden_dim
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable [CLS] token representing whole-slide context
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=1e-2)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransMILBlock(dim=hidden_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])

        # PPEG positional generator
        self.ppeg = PPEG(dim=hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        # Classifier / Predictor Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, h: torch.Tensor, return_features: bool = False):
        """
        Args:
            h (torch.Tensor): Tensor of shape (N, input_dim) or (B, N, input_dim).
        Returns:
            dict with 'logits' (or 'risk_score' if n_classes==1), and optionally 'slide_embedding'.
        """
        # Ensure (B, N, input_dim)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        B, N, C = h.shape

        # Project patch features
        h = self.fc_in(h.float())  # (B, N, hidden_dim)

        # PPEG positional context: Pad to nearest square H x W >= N
        H = int(math.ceil(math.sqrt(N)))
        W = H
        pad_size = H * W - N
        if pad_size > 0:
            pad = torch.zeros((B, pad_size, self.hidden_dim), device=h.device, dtype=h.dtype)
            h_padded = torch.cat([h, pad], dim=1)
        else:
            h_padded = h

        # Apply PPEG
        h_pos = self.ppeg(h_padded, H, W)
        # Unpad
        h = h_pos[:, :N, :]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)  # (B, 1 + N, hidden_dim)

        # Pass through Transformer encoder layers
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # Extract [CLS] token embedding for whole-slide prediction
        slide_feat = h[:, 0, :]  # (B, hidden_dim)

        logits = self.classifier(slide_feat)

        out = {
            "logits": logits,
        }
        if return_features:
            out["slide_embedding"] = slide_feat

        return out
