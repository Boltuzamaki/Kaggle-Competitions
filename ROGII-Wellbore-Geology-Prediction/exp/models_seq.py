"""Sequence-to-sequence per-position regressors (predict dTVT at every point).
Families: Conv1D+BiGRU (well-log SOTA), Transformer encoder, TCN, TCN-Transformer."""
import math
import torch
import torch.nn as nn


class ConvBiGRU(nn.Module):
    def __init__(self, c_in, d=96, layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, d, 7, padding=3), nn.GELU(),
            nn.Conv1d(d, d, 7, padding=3), nn.GELU())
        self.gru = nn.GRU(d, d, layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask=None):          # x: (B,L,C)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=8192):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class TransformerSeq(nn.Module):
    def __init__(self, c_in, d=128, nhead=4, layers=4, ff=256):
        super().__init__()
        self.proj = nn.Linear(c_in, d)
        self.pe = PositionalEncoding(d)
        enc = nn.TransformerEncoderLayer(d, nhead, ff, dropout=0.1,
                                         batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask=None):
        h = self.pe(self.proj(x))
        kpm = (~mask) if mask is not None else None   # True = pad -> ignore
        h = self.enc(h, src_key_padding_mask=kpm)
        return self.head(h).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, c, k, dil):
        super().__init__()
        pad = (k - 1) * dil // 2
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding=pad, dilation=dil), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(c, c, k, padding=pad, dilation=dil), nn.GELU())

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    def __init__(self, c_in, d=96, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(c_in, d, 1)
        self.blocks = nn.ModuleList([_TCNBlock(d, 5, dl) for dl in dils])
        self.head = nn.Sequential(nn.Conv1d(d, d, 1), nn.GELU(), nn.Conv1d(d, 1, 1))

    def forward(self, x, mask=None):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)


class TCNTransformer(nn.Module):
    """TCN local encoder -> Transformer global mixer (2025 SOTA-flavoured)."""
    def __init__(self, c_in, d=128, nhead=4, layers=3, ff=256, dils=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Conv1d(c_in, d, 1)
        self.tcn = nn.ModuleList([_TCNBlock(d, 5, dl) for dl in dils])
        self.pe = PositionalEncoding(d)
        enc = nn.TransformerEncoderLayer(d, nhead, ff, dropout=0.1,
                                         batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask=None):
        h = self.inp(x.transpose(1, 2))
        for b in self.tcn:
            h = b(h)
        h = h.transpose(1, 2)
        h = self.pe(h)
        kpm = (~mask) if mask is not None else None
        h = self.enc(h, src_key_padding_mask=kpm)
        return self.head(h).squeeze(-1)


def build(name, c_in):
    return {"convgru": ConvBiGRU, "transformer": TransformerSeq,
            "tcn": TCN, "tcntransformer": TCNTransformer}[name](c_in)
