import math
import pickle

import torch
from torch.utils import data
from tensorfn import load_config

from torch_imputer import best_alignment

from config import CTCASR
from dataset import ASRDataset, collate_data
from model import Transformer
from evaluate import ctc_decode

conf = load_config(CTCASR, "aihub.conf")

device = "cuda"

dset = ASRDataset(conf.dataset.path)

with open("trainval_indices.pkl", "rb") as f:
    split_indices = pickle.load(f)

train_set = data.Subset(dset, split_indices["train"])


model = Transformer(
    conf.dataset.n_vocab,
    conf.model.delta,
    conf.dataset.n_mels,
    conf.model.feature_channel,
    conf.model.dim,
    conf.model.dim_ff,
    conf.model.n_layer,
    conf.model.n_head,
    conf.model.dropout,
).to(device)


ckpt = torch.load("checkpoint/ctc_005000.pt", map_location=lambda storage, loc: storage)
model.load_state_dict(ckpt["model"])
model.eval()

batch0 = train_set[10]
batch1 = train_set[100]
mel_b, token_b, mel_len, token_len, texts, files = collate_data([batch0, batch1])
mel_b = mel_b.to(device)
align_in = torch.ones(
    1, math.ceil(mel_b.shape[1] / conf.model.reduction), dtype=torch.int64
).to(device)
logit = model(mel_b, align_in)

mel_len_reduce = torch.ceil(mel_len.to(torch.float32) / conf.model.reduction).to(
    torch.int64
)

best_align = best_alignment(
    torch.log_softmax(logit, 2).transpose(0, 1), token_b, mel_len_reduce, token_len
)

print(best_align)
