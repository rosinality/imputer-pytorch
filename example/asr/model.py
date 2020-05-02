import math

import torch
from torch import nn
from torch.nn import functional as F

from audio import DeltaFeature


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, n_head, dropout=0.1):
        super().__init__()

        self.dim_head = dim // n_head
        self.n_head = n_head

        self.weight = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, dim)

    def forward(self, input, mask=None):
        query = input

        batch_size = query.shape[0]

        def reshape(input):
            return input.view(batch_size, -1, self.n_head, self.dim_head).transpose(
                1, 2
            )

        query, key, value = self.weight(query).chunk(3, dim=2)

        query = reshape(query)
        key = reshape(key).transpose(2, 3)
        value = reshape(value)

        attn = torch.matmul(query, key) / math.sqrt(self.dim_head)

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, 3)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.dim_head * self.n_head)
        )
        out = self.linear(out)

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.dropout = dropout

    def forward(self, input):
        out = self.feedforward(input)

        return out


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_head, dim_ff, dropout=0.1):
        super().__init__()

        self.norm_attention = nn.LayerNorm(dim)
        self.attention = MultiHeadedAttention(dim, n_head, dropout)
        self.dropout_attention = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = PositionwiseFeedForward(dim, dim_ff)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, input):
        skip = input
        out = self.dropout_attention(self.attention(self.norm_attention(input)))
        out = out + skip

        skip = out
        out = self.dropout_ff(self.ff(self.norm_ff(out)))
        out = out + skip

        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, batch_size=None):
        sinusoid_in = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1).unsqueeze(0)

        if batch_size is not None:
            return pos_emb.expand(batch_size, -1, -1)

        else:
            return pos_emb


class Conv2dFeature(nn.Module):
    def __init__(self, in_channel, feat_dim, channel, out_dim):
        super().__init__()

        self.in_channel = in_channel
        self.feat_dim = feat_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, channel, (11, 3), stride=(2, 1), padding=(5, 1)),
            nn.ReLU(),
            nn.Conv2d(channel, channel, (11, 3), stride=(2, 1), padding=(5, 1)),
            nn.ReLU(),
        )

        self.linear = nn.Linear(feat_dim * channel, out_dim)

    def forward(self, input):
        # input: B, T, D
        batch, time, dim = input.shape
        out = input.view(batch, time, self.in_channel, self.feat_dim).transpose(1, 2)
        out = self.conv(out)
        out = out.transpose(1, 2).reshape(
            batch, out.shape[2], -1
        )  # B, C, T / s, D -> B, C, D, T / s -> B, CD, T / s -> B, T / s, CD
        out = self.linear(out)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        n_label,
        delta,
        feature_dim,
        feature_channel,
        dim,
        dim_ff,
        n_layer,
        n_head,
        dropout,
        normalize=True,
    ):
        super().__init__()

        self.dim = dim

        self.feature = Conv2dFeature(delta + 1, feature_dim, feature_channel, dim)

        self.embed = nn.Embedding(n_label + 1, dim)
        self.pos_enc = PositionalEmbedding(dim)

        self.dropout_embed = nn.Dropout(dropout)

        layers = []
        for i in range(n_layer):
            layers.append(TransformerLayer(dim, n_head, dim_ff, dropout))

        self.layers = nn.Sequential(*layers)

        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_label))

        self.reset_parameters()

    def copy_embed(self, index):
        target = self.embed.weight.data[index]
        target_repeat = target.repeat(self.embed.weight.data.shape[0], 1)
        self.embed.weight.data.copy_(target_repeat)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input, text):
        # input: B, T, D
        out = input

        out = self.feature(out)
        pos_seq = torch.arange(out.shape[1], device=out.device, dtype=out.dtype)
        pos_enc = self.pos_enc(pos_seq)
        embed = self.embed(text)  # B L D

        out = self.dropout_embed((out + embed) * math.sqrt(self.dim) + pos_enc)

        out = self.layers(out)

        out = self.out(out)

        return out
