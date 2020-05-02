import math
import os
import pickle
from typing import NamedTuple, Optional, Any, Dict

import torch
from torch import nn, optim
from torch.utils import data
import numpy as np
from tqdm import tqdm
from tensorfn import load_wandb, load_arg_config, distributed as dist
from tensorfn.data import create_groups, GroupedBatchSampler

from config import CTCASR
from dataset import ASRDataset, collate_data
from model import Transformer
from evaluate import valid


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


class ModelTraining(NamedTuple):
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[Any]
    dataset: ASRDataset
    train_loader: data.DataLoader
    valid_loader: data.DataLoader
    device: str
    wandb: Optional[Any]


def train(conf, model_training):
    criterion = nn.CTCLoss(reduction="mean", zero_infinity=True)

    loader = sample_data(model_training.train_loader)
    pbar = range(conf.training.n_iter + 1)
    if dist.is_primary():
        pbar = tqdm(pbar)

    device = model_training.device
    model = model_training.model
    optimizer = model_training.optimizer
    scheduler = model_training.scheduler

    for i in pbar:
        epoch, (mels, tokens, mel_lengths, token_lengths, texts, files) = next(loader)
        mels = mels.to(device)
        tokens = tokens.to(device)

        mel_len_reduce = torch.ceil(
            mel_lengths.to(torch.float32) / conf.model.reduction
        ).to(torch.int64)

        align_in = tokens.new_ones(
            mels.shape[0], math.ceil(mels.shape[1] / conf.model.reduction)
        )

        out = torch.log_softmax(model(mels, align_in), 2)

        loss = criterion(
            out.transpose(0, 1).contiguous(), tokens, mel_len_reduce, token_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        scheduler.step()
        optimizer.step()

        if dist.is_primary() and conf.training.scheduler.type == "lr_find":
            scheduler.record_loss(loss)

        if i % conf.evaluate.log_every == 0:
            loss_dict = {"loss": loss}
            loss_reduced = dist.reduce_dict(loss_dict)
            loss_ctc = loss_reduced["loss"].mean().item()

            lr = optimizer.param_groups[0]["lr"]

        if i > 0 and i % conf.evaluate.valid_every == 0:
            valid(conf, model_training, i)

        if dist.is_primary():
            if i % conf.evaluate.log_every == 0:
                pbar.set_description(
                    f"epoch: {epoch}; loss: {loss_ctc:.4f}; lr: {lr:.5f}"
                )

                if conf.evaluate.wandb and model_training.wandb is not None:
                    model_training.wandb.log(
                        {
                            "training/epoch": epoch,
                            "training/loss": loss_ctc,
                            "training/lr": lr,
                        },
                        step=i,
                    )

            if i > 0 and i % conf.evaluate.save_every == 0:
                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "conf": conf,
                    },
                    f"checkpoint/ctc_{str(i).zfill(6)}.pt",
                )

    if dist.is_primary() and conf.training.scheduler.type == "lr_find":
        scheduler.write_log("loss.log")


def main(conf):
    conf.distributed = dist.get_world_size() > 1

    device = "cuda"

    if dist.is_primary():
        from pprint import pprint

        pprint(conf.dict())

    if dist.is_primary() and conf.evaluate.wandb:
        wandb = load_wandb()
        wandb.init(project="asr")

    else:
        wandb = None

    with open("trainval_indices.pkl", "rb") as f:
        split_indices = pickle.load(f)

    train_set = ASRDataset(conf.dataset.path, indices=split_indices["train"])
    valid_set = ASRDataset(conf.dataset.path, indices=split_indices["val"])

    train_sampler = dist.data_sampler(
        train_set, shuffle=True, distributed=conf.distributed
    )
    valid_sampler = dist.data_sampler(
        valid_set, shuffle=False, distributed=conf.distributed
    )

    if conf.training.batch_sampler is not None:
        train_lens = []

        for i in split_indices["train"]:
            train_lens.append(train_set.mel_lengths[i])

        opts = conf.training.batch_sampler

        bins = (
            (opts.base ** np.linspace(opts.start, 1, 2 * opts.k + 1)) * 1000
        ).tolist()
        groups, bins, n_samples = create_groups(train_lens, bins)
        batch_sampler = GroupedBatchSampler(
            train_sampler, groups, conf.training.dataloader.batch_size
        )

        conf.training.dataloader.batch_size = 1
        train_loader = conf.training.dataloader.make(
            train_set, batch_sampler=batch_sampler, collate_fn=collate_data
        )

    else:
        train_loader = conf.training.dataloader.make(train_set, collate_fn=collate_data)

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

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        model_p = model

        if conf.distributed:
            model_p = model.module

        model_p.load_state_dict(ckpt["model"])
        scheduler.load_state_dict(ckpt["scheduler"])

    model_training = ModelTraining(
        model,
        optimizer,
        scheduler,
        train_set,
        train_loader,
        valid_loader,
        device,
        wandb,
    )

    train(conf, model_training)


if __name__ == "__main__":
    conf = load_arg_config(CTCASR, show=False)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
