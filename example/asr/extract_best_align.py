import math
import pickle

import torch
from torch.utils import data
from tensorfn import load_arg_config
from tqdm import tqdm
import lmdb

from torch_imputer import best_alignment

from config import CTCASR
from dataset import ASRDataset, collate_data
from model import Transformer
from evaluate import ctc_decode


def get_symbol(state, targets_list):
    if state % 2 == 0:
        symbol = 0

    else:
        symbol = targets_list[state // 2]

    return symbol

    return state


if __name__ == "__main__":
    device = "cuda"

    conf = load_arg_config(CTCASR)

    with open("trainval_indices.pkl", "rb") as f:
        split_indices = pickle.load(f)

    train_set = ASRDataset(conf.dataset.path, indices=split_indices["train"])

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

    ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model"])

    model.eval()

    train_loader = conf.training.dataloader.make(train_set, collate_fn=collate_data)

    pbar = tqdm(train_loader)

    show_sample = 0
    db_i = 0

    with torch.no_grad() as no_grad, lmdb.open(
        conf.dataset.alignment, map_size=1024 ** 4, readahead=False
    ) as env:
        for mels, tokens, mel_lengths, token_lengths, texts, files in pbar:
            mels = mels.to(device)
            tokens = tokens.to(device).to("cpu")

            mel_len_reduce = torch.ceil(
                mel_lengths.to(torch.float32) / conf.model.reduction
            ).to(torch.int64)

            align_in = tokens.new_ones(
                mels.shape[0], math.ceil(mels.shape[1] / conf.model.reduction)
            ).to(device)

            log_p = torch.log_softmax(model(mels, align_in), 2)

            # log_p = log_p.to("cpu")
            tokens = tokens.to("cpu")

            best_align = best_alignment(
                log_p.transpose(0, 1),
                tokens,
                mel_len_reduce,
                token_lengths,
                zero_infinity=True,
            )

            align_probs = []

            for l_p, best_a, toks in zip(log_p.to("cpu"), best_align, tokens.tolist()):
                align_p = []

                for p, a in zip(l_p, best_a):
                    align_p.append(p[get_symbol(a, toks)].item())

                align_probs.append(align_p)

            for model_align, mel_l, b_align, b_p, file, toks in zip(
                log_p, mel_len_reduce, best_align, align_probs, files, tokens.tolist()
            ):
                model_p, model_align = model_align.max(1)
                model_p = model_p[:mel_l].sum().item()
                model_align = model_align[:mel_l].tolist()
                b_p = sum(b_p)

                with env.begin(write=True) as txn:
                    txn.put(
                        str(db_i).encode("utf-8"),
                        pickle.dumps((b_align, b_p, model_align, model_p, file)),
                    )
                    db_i += 1

                if show_sample < 8:
                    model_align = train_set.decode(ctc_decode(model_align))
                    b_align = train_set.decode(
                        ctc_decode([get_symbol(a, toks) for a in b_align])
                    )

                    print(
                        f"model: {model_align} ({model_p:.3f})\nbest: {b_align} ({b_p:.3f})"
                    )

                    show_sample += 1

        with env.begin(write=True) as txn:
            txn.put(b"length", str(db_i).encode("utf-8"))
            txn.put(b"meta", pickle.dumps({"conf": conf}))
