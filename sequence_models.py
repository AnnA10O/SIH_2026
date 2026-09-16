"""
sequence_models.py — PS-26077 Cloudburst Nowcaster
Two architectures, both fed the SAME real (batch, seq_len>1, core_feats)
tensor from build_sequences.py, plus a (batch, static_feats) context vector
concatenated after temporal pooling. Neither model sees a length-1 fake
sequence — this is what the earlier CNN+BiLSTM and SpatiotemporalTransformer
were both missing (confirmed by the codebase's own docstring: "the BiLSTM
hidden state does NOT persist across rows in the batch").

Sizing rationale (why these are deliberately small):
  Your dataset has few distinct storm EVENTS (event_grouped_split groups
  by event, not by row) — high model capacity on a small number of
  independent events overfits fast, which is the most likely reason the
  earlier full-size SpatiotemporalTransformer (d_model=64, 2 layers,
  cross-attention, but fed a length-1 sequence) failed. Keeping d_model
  small and dropout/weight-decay high here is a deliberate choice, not
  a placeholder — raise capacity only after confirming these underfit on
  real validation curves, not before.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal PE — needed now because seq_len > 1 for real,
    so token order actually carries information (it didn't before)."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def _static_head(static_feats: int, d_model: int) -> nn.Module:
    if static_feats == 0:
        return None
    return nn.Sequential(nn.Linear(static_feats, d_model), nn.SiLU())


class SequenceBiLSTM(nn.Module):
    """Real BiLSTM over real hourly timesteps. Max-pool over time, concat static context."""

    def __init__(self, core_feats: int, static_feats: int, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(core_feats)
        self.lstm = nn.LSTM(
            input_size=core_feats, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_head = _static_head(static_feats, hidden_dim * 2)
        head_in = hidden_dim * 2 + (hidden_dim * 2 if static_feats > 0 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        # x_seq: (batch, seq_len, core_feats) — a REAL sequence this time
        x_seq = self.input_norm(torch.nan_to_num(x_seq, nan=0.0))
        lstm_out, _ = self.lstm(x_seq)             # (batch, seq_len, 2*hidden)
        pooled, _ = torch.max(lstm_out, dim=1)      # (batch, 2*hidden)
        feats = [pooled]
        if self.static_head is not None and x_static is not None:
            feats.append(self.static_head(torch.nan_to_num(x_static, nan=0.0)))
        concat = torch.cat(feats, dim=-1)
        return self.head(concat).squeeze(-1)

    def predict_proba(self, x_seq: torch.Tensor, x_static: torch.Tensor = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_seq, x_static))


class TinySeqTransformer(nn.Module):
    """Small Transformer encoder over real hourly timesteps. Mean-pool over time,
    concat static context. Deliberately small — see module docstring."""

    def __init__(self, core_feats: int, static_feats: int, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 1, dropout: float = 0.3, max_len: int = 32):
        super().__init__()
        self.embed = nn.Linear(core_feats, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.static_head = _static_head(static_feats, d_model)
        head_in = d_model + (d_model if static_feats > 0 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x_seq: (batch, seq_len, core_feats), seq_len > 1 — attention has
        # something real to attend across this time, unlike the old length-1 version.
        x_seq = torch.nan_to_num(x_seq, nan=0.0)
        emb = self.pos_enc(self.embed(x_seq))
        enc = self.encoder(emb, src_key_padding_mask=key_padding_mask)  # (batch, seq_len, d_model)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).float()
            pooled = (enc * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = enc.mean(dim=1)
        feats = [pooled]
        if self.static_head is not None and x_static is not None:
            feats.append(self.static_head(torch.nan_to_num(x_static, nan=0.0)))
        concat = torch.cat(feats, dim=-1)
        return self.head(concat).squeeze(-1)

    def predict_proba(self, x_seq: torch.Tensor, x_static: torch.Tensor = None,
                       key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_seq, x_static, key_padding_mask))
