import math
import pickle
from typing import NamedTuple, Optional, Any

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from tqdm import tqdm
from tensorfn import load_arg_config, distributed as dist
import editdistance

from config import CTCASR
from dataset import ASRDataset, collate_data
from model import Transformer


def char_distance(ref, hyp):
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")

    dist = editdistance.eval(hyp, ref)
    length = len(ref)

    return dist, length


def ctc_decode(seq, blank=0):
    result = []

    prev = -1
    for s in seq:
        if s == blank:
            prev = s

            continue

        if prev == -1:
            result.append(s)

        else:
            if s != prev:
                result.append(s)

        prev = s

    return result


class ModelValid(NamedTuple):
    model: nn.Module
    dataset: ASRDataset
    valid_loader: data.DataLoader
    device: str
    wandb: Optional[Any]


'''@torch.no_grad()
def decode_argmax(log_prob, block_size, n_candid, mask_token=1):
    batch, length, n_vocab = log_prob.shape
    len_block = math.ceil(length / block_size) * block_size
    pad_log_prob = F.pad(log_prob, (0, 0, 0, len_block - length)).view(
        batch, block_size, -1, n_vocab
    )
    pad_max_prob, pad_max_token = pad_log_prob.max(3)
    threshold = pad_max_prob.topk(n_candid + 1, dim=1).values.min(1).values.unsqueeze(1)
    mask = (pad_max_prob > threshold).to(torch.int64)

    mask = mask.transpose(1, 2).reshape(batch, -1)[:, :length]
    pad_max_token = pad_max_token.view(batch, -1)[:, :length]
    masked_token = pad_max_token * mask + (1 - mask) * mask_token

    return masked_token, mask'''


@torch.no_grad()
def decode_argmax(logit, block_size, n_candid, prev_mask=None, mask_token=1):
    neginf = float('-inf')
    batch, length, n_vocab = logit.shape
    len_block = math.ceil(length / block_size) * block_size

    pad_logit = logit
    if prev_mask is not None:
        pad_logit = pad_logit.masked_fill((prev_mask == 1).unsqueeze(-1), neginf)

    pad_log_prob = (
        F.pad(pad_logit, (0, 0, 0, len_block - length), value=neginf)
        .view(batch, -1, block_size, n_vocab)
        .transpose(1, 2)
    )
    pad_max_prob = pad_log_prob.max(3).values
    threshold = pad_max_prob.topk(n_candid + 1, dim=1).values.min(1).values.unsqueeze(1)
    mask = (pad_max_prob > threshold).to(torch.int64)

    mask = mask.transpose(1, 2).reshape(batch, -1)[:, :length]

    if prev_mask is not None:
        mask = (mask + prev_mask).clamp(max=1)

    masked_token = logit.argmax(2) * mask + (1 - mask) * mask_token

    return masked_token, mask


@torch.no_grad()
def valid(conf, model_training, step, block_size=1, max_decode_iter=1):
    criterion = nn.CTCLoss(reduction="mean", zero_infinity=True)
    pbar = model_training.valid_loader

    if dist.is_primary():
        pbar = tqdm(pbar)

    device = model_training.device
    model = model_training.model
    decoder = model_training.dataset.decode

    was_training = model.training
    model.eval()

    dist.synchronize()

    total_dist = 0
    total_length = 0
    show_text = 0
    text_table = []

    for mels, tokens, mel_lengths, token_lengths, texts, _ in pbar:
        mels = mels.to(device)
        tokens = tokens.to(device)
        mel_len_reduce = torch.ceil(
            mel_lengths.to(torch.float32) / conf.model.reduction
        ).to(torch.int64)

        pred_token = None

        for decode_candid in range(0, block_size, block_size // max_decode_iter):
            if decode_candid == 0:
                align_in = tokens.new_ones(
                    mels.shape[0], math.ceil(mels.shape[1] / conf.model.reduction)
                )
                out = None
                mask = None

            else:
                align_in, mask = decode_argmax(
                    out, block_size, block_size // max_decode_iter, mask
                )

            out = torch.log_softmax(model(mels, align_in), 2)

            """if pred_token is None:
                pred_token = out.argmax(2)

            else:
                pred_token = (1 - mask) * out.argmax(2) + mask * pred_token"""
            pred_token = out.argmax(2)

        loss = criterion(
            out.transpose(0, 1).contiguous(), tokens, mel_len_reduce, token_lengths
        )

        pred_token = pred_token.to("cpu").tolist()

        for mel_len, pred_tok, gt in zip(mel_len_reduce.tolist(), pred_token, texts):
            pred = "".join(decoder(ctc_decode(pred_tok[:mel_len])))
            editdist, reflen = char_distance(gt, pred)
            total_dist += editdist
            total_length += reflen

            if dist.is_primary() and show_text < 8:
                pbar.write(f"gt: {gt}\t\ttranscription: {pred}")
                show_text += 1
                text_table.append([gt, pred, str(editdist), str(editdist / reflen)])

        dist.synchronize()

        comm = {
            "loss": loss.item(),
            "total_dist": total_dist,
            "total_length": total_length,
        }
        comm = dist.all_gather(comm)

        part_dist = 0
        part_len = 0
        part_loss = 0
        for eval_parts in comm:
            part_dist += eval_parts["total_dist"]
            part_len += eval_parts["total_length"]
            part_loss += loss

        if dist.is_primary():
            n_part = len(comm)
            cer = part_dist / part_len * 100
            pbar.set_description(f"loss: {part_loss / n_part:.4f}; cer: {cer:.2f}")

        dist.synchronize()

    if dist.is_primary():
        n_part = len(comm)
        cer = part_dist / part_len * 100
        pbar.write(f"loss: {part_loss / n_part:.4f}; cer: {cer:.2f}")

        if conf.evaluate.wandb and model_training.wandb is not None:
            model_training.wandb.log(
                {
                    f"valid/iter-{max_decode_iter}/loss": part_loss / n_part,
                    f"valid/iter-{max_decode_iter}/cer": cer,
                    f"valid/iter-{max_decode_iter}/text": model_training.wandb.Table(
                        data=text_table,
                        columns=["Reference", "Transcription", "Edit Distance", "CER"],
                    ),
                },
                step=step,
            )

    if was_training:
        model.train()


def main(conf):
    conf.distributed = False

    device = "cpu"

    if dist.is_primary():
        from pprint import pprint

        pprint(conf.dict())

    with open("trainval_indices.pkl", "rb") as f:
        split_indices = pickle.load(f)

    valid_set = ASRDataset(conf.dataset.path, indices=split_indices["val"])

    valid_sampler = dist.data_sampler(
        valid_set, shuffle=False, distributed=conf.distributed
    )

    valid_loader = conf.training.dataloader.make(
        valid_set, sampler=valid_sampler, collate_fn=collate_data
    )

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

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        model_p = model

        if conf.distributed:
            model_p = model.module

        model_p.load_state_dict(ckpt["model"])

    model_valid = ModelValid(model, valid_set, valid_loader, device, None)

    valid(conf, model_valid, 0, block_size=8, max_decode_iter=2)


if __name__ == "__main__":
    conf = load_arg_config(CTCASR, show=False)

    main(conf)
