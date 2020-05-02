import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.compliance import kaldi


class SpecNormalize(nn.Module):
    def __init__(self, dim=2, eps=1e-8):
        super().__init__()

        self.dim = dim
        self.eps = eps

    def forward(self, input):
        mean = input.mean(self.dim, keepdim=True)
        std = input.std(self.dim, keepdim=True)

        out = (input - mean) / (std + self.eps)

        return out


class FilterbankFeature(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()

        self.n_mels = n_mels

    def forward(self, input, sample_rate):
        feat = kaldi.fbank(
            input, channel=-1, sample_frequency=sample_rate, num_mel_bins=self.n_mels
        )

        return feat


class DeltaFeature(nn.Module):
    def __init__(self, order=1, window_size=2):
        super().__init__()

        self.order = order
        self.window_size = window_size

        filters = torch.from_numpy(self.make_filter(self.order, self.window_size)).to(
            torch.float32
        )
        self.register_buffer("filters", filters.unsqueeze(1))
        self.padding = (filters.shape[-1] - 1) // 2

    def make_filter(self, order, window_size):
        denom = window_size * (window_size + 1) * (2 * window_size + 1) / 3
        delta = np.arange(-self.window_size, self.window_size + 1)
        delta_filters = [np.array([1]), delta]
        delta_next = delta

        for i in range(order - 1):
            delta_next = np.convolve(delta, delta_next)
            delta_filters.append(delta_next)

        max_len = delta_filters[-1].shape[0]

        delta_filters_pad = []

        for delta in delta_filters:
            pad = (max_len - delta.shape[0]) // 2
            delta = np.pad(delta, (pad, pad))

            delta_filters_pad.append(delta)

        delta_filters = np.stack(delta_filters_pad, 0)
        delta_filters = delta_filters / np.expand_dims(
            denom ** np.arange(delta_filters.shape[0]), -1
        )

        return delta_filters

    def forward(self, input):
        filter_exp = self.filters.repeat(1, input.shape[1], 1).view(
            -1, 1, self.filters.shape[-1]
        )

        return F.conv1d(input, filter_exp, padding=self.padding, groups=input.shape[1])
