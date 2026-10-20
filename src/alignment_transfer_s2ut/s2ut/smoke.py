"""Tiny offline test model. This backend is NOT a fairseq S2UT reproduction."""
import math
import torch
from torch import nn
from ..guidance.attention import attention
from .model import NativeModel


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.q, self.k, self.v, self.out = [nn.Linear(dim, dim) for _ in range(4)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, mask, prior, beta, heads, prior_smoothing=0.0):
        b, t, d = x.shape
        def split(v):
            return v.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(memory)), split(self.v(memory))
        raw, guided = attention(q @ k.transpose(-2, -1) / math.sqrt(d // self.heads), prior, beta, mask,
                                prior_smoothing=prior_smoothing)
        context = (self.dropout(guided) @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(context), (raw if heads is None else raw[:, heads]).mean(1), \
            (guided if heads is None else guided[:, heads]).mean(1)


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.decoder_embed_dim
        self.self_attn = nn.MultiheadAttention(d, c.decoder_attention_heads,
                                              dropout=c.attention_dropout, batch_first=True)
        self.cross = CrossAttention(d, c.decoder_attention_heads, c.attention_dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(d, c.decoder_ffn_embed_dim), nn.ReLU(),
                                 nn.Dropout(c.activation_dropout), nn.Linear(c.decoder_ffn_embed_dim, d))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, memory, mask, prior, beta, heads, prior_smoothing=0.0):
        causal = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), 1)
        n = self.norms[0](x)
        x = x + self.drop(self.self_attn(n, n, n, attn_mask=causal, need_weights=False)[0])
        context, raw, guided = self.cross(self.norms[1](x), memory, mask, prior, beta, heads, prior_smoothing)
        x = x + self.drop(context)
        return x + self.drop(self.ffn(self.norms[2](x))), raw, guided


def positions(length, dim, device):
    inv = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    angles = torch.arange(length, device=device)[:, None] * inv[None]
    return torch.stack([angles.sin(), angles.cos()], -1).reshape(length, -1)[:, :dim]


class SmokeModel(nn.Module):
    def __init__(self, c, modality, source_vocabulary, target_vocabulary):
        super().__init__()
        if c.encoder_embed_dim != c.decoder_embed_dim:
            raise ValueError("smoke backend requires equal dimensions")
        self.config, self.modality = c, modality
        self.frontend = nn.Embedding(len(source_vocabulary), c.encoder_embed_dim, padding_idx=1) \
            if modality == "text" else nn.Conv1d(c.input_feat_per_channel, c.encoder_embed_dim, 5, stride=4, padding=2)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(c.encoder_embed_dim,
                    c.encoder_attention_heads, c.encoder_ffn_embed_dim, c.dropout, batch_first=True,
                    norm_first=True), c.encoder_layers, norm=nn.LayerNorm(c.encoder_embed_dim),
                    enable_nested_tensor=False)
        self.embedding = nn.Embedding(len(target_vocabulary), c.decoder_embed_dim, padding_idx=1)
        self.decoder = nn.ModuleList(DecoderLayer(c) for _ in range(c.decoder_layers))
        self.final_norm = nn.LayerNorm(c.decoder_embed_dim)
        self.projection = nn.Linear(c.decoder_embed_dim, len(target_vocabulary), bias=False)
        if c.share_decoder_input_output_embed:
            self.projection.weight = self.embedding.weight

    def encode(self, source, lengths):
        if self.modality == "text":
            x = self.frontend(source)
        else:
            x = self.frontend(source.transpose(1, 2)).transpose(1, 2)
            lengths = (lengths + 3) // 4
        mask = torch.arange(x.size(1), device=x.device)[None] >= lengths[:, None]
        x = x * math.sqrt(self.config.encoder_embed_dim) + positions(x.size(1), x.size(2), x.device)
        return self.encoder(x, src_key_padding_mask=mask), mask, lengths

    @staticmethod
    def encoder_lengths(encoded):
        return encoded[2]

    def output_lengths(self, lengths):
        return lengths if self.modality == "text" else (lengths + 3) // 4

    def decode(self, previous, encoded, prior=None, beta=0.0, layer=-1, heads=None, prior_smoothing=0.0):
        memory, mask, _ = encoded
        x = self.embedding(previous) * math.sqrt(self.config.decoder_embed_dim)
        x = x + positions(x.size(1), x.size(2), x.device)
        selected = layer % len(self.decoder)
        for index, block in enumerate(self.decoder):
            x, current_raw, current_guided = block(x, memory, mask, prior if index == selected else None,
                                                   beta if index == selected else 0, heads, prior_smoothing)
            if index == selected:
                raw, guided = current_raw, current_guided
        return self.projection(self.final_norm(x)), raw, guided

    forward = NativeModel.forward
