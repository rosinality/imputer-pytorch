from typing import Optional
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool

from tensorfn.config import Config, Optimizer, Scheduler, DataLoader


class Dataset(Config):
    name: StrictStr
    path: StrictStr
    alignment: StrictStr
    n_mels: StrictInt
    n_vocab: StrictInt


class Model(Config):
    delta: StrictInt
    feature_channel: StrictInt
    dim: StrictInt
    dim_ff: StrictInt
    n_layer: StrictInt
    n_head: StrictInt
    dropout: StrictFloat
    reduction: StrictInt


class BatchSampler(Config):
    base: StrictInt
    start: StrictInt
    k: StrictInt
    length_multiplier: StrictInt


class Training(Config):
    n_iter: StrictInt
    optimizer: Optimizer
    scheduler: Optional[Scheduler]
    dataloader: DataLoader
    batch_sampler: Optional[BatchSampler]


class Eval(Config):
    wandb: StrictBool
    save_every: StrictInt
    valid_every: StrictInt
    log_every: StrictInt


class CTCASR(Config):
    n_gpu: Optional[StrictInt]
    n_machine: Optional[StrictInt]
    machine_rank: Optional[StrictInt]
    dist_url: Optional[StrictStr]
    distributed: Optional[StrictBool]
    ckpt: Optional[StrictStr]
    evaluate: Eval

    dataset: Dataset
    model: Model
    training: Training
