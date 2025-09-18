from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res, inplace=True)


class TCN(nn.Module):
    def __init__(self, in_ch: int, channels: Tuple[int, ...], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        c_prev = in_ch
        for i, c in enumerate(channels):
            d = 2 ** i
            layers.append(TemporalBlock(c_prev, c, kernel_size, d, dropout))
            c_prev = c
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AttentiveMILPooling(nn.Module):
    """Attention pooling over time with mask.

    Given H: [B, L, D], mask: [B, L] (True for valid), returns pooled [B, D].
    """

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # H: [B, L, D], mask: [B, L]
        scores = self.att(H).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))
        A = torch.softmax(scores, dim=1)  # [B, L]
        A = A.masked_fill(~mask, 0.0)
        pooled = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [B, D]
        return pooled


class MILTCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hid_channels: Tuple[int, ...] = (64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        mil_hidden: int = 64,
        use_noisy_or: bool = True,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for symmetric receptive field"
        self.tcn = TCN(in_ch=in_features, channels=hid_channels, kernel_size=kernel_size, dropout=dropout)
        D = hid_channels[-1]
        self.instance_head = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(D, 1, kernel_size=1),
        )
        self.pool = AttentiveMILPooling(D, hidden=mil_hidden)
        self.bag_head = nn.Linear(D, 1)
        self.use_noisy_or = use_noisy_or

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, L, F]
            mask: [B, L] (bool) True where valid

        Returns:
            bag_logits: [B, 1]
            inst_logits: [B, L]
        """
        B, L, F = x.shape
        # TCN expects [B, C, T]
        h = self.tcn(x.transpose(1, 2))  # [B, D, L]
        inst_logits = self.instance_head(h).squeeze(1)  # [B, L]

        # MIL pooling branch
        H = h.transpose(1, 2)  # [B, L, D]
        pooled = self.pool(H, mask)  # [B, D]
        bag_logits = self.bag_head(pooled)  # [B, 1]

        if self.use_noisy_or:
            # Blend with Noisy-OR for sharper 'any-of' semantics
            inst_probs = torch.sigmoid(inst_logits).masked_fill(~mask, 0.0)
            # p_bag = 1 - prod(1 - p_i)
            log_1mp = torch.log(torch.clamp(1.0 - inst_probs + 1e-7, min=1e-7))
            sum_log = torch.sum(log_1mp, dim=1, keepdim=True)
            p_no = 1.0 - torch.exp(sum_log)
            bag_logits = bag_logits + torch.logit(torch.clamp(p_no, 1e-6, 1 - 1e-6))

        return bag_logits, inst_logits
