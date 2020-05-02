import math

import torch
from torch.utils.data import Dataset
from tensorfn.data import LMDBReader

from audio import SpecNormalize, DeltaFeature

EPS_TOKEN = 0
MASK_TOKEN = 1
UNK_TOKEN = 2


class ASRDataset:
    def __init__(self, path, indices, delta=2, normalize=True, alignment=None):
        self.db = LMDBReader(path)

        meta = self.db.get(b"meta", reader="pickle")

        if alignment is not None:
            self.align_db = LMDBReader(alignment, reader="pickle")

        else:
            self.align_db = None

        self.vocab = meta["vocab"]
        self.vocab["[E]"] = EPS_TOKEN
        self.vocab["[M]"] = MASK_TOKEN
        self.vocab["[U]"] = UNK_TOKEN

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.indices = indices

        self.mel_lengths = meta["mel_lengths"]
        self.text_lengths = meta["text_lengths"]

        self.normalize = normalize
        self.delta = delta

        if self.normalize:
            self.spec_normalize = SpecNormalize(0)

        if self.delta > 0:
            self.delta_feature = DeltaFeature(self.delta)

    def decode(self, code):
        text = []

        for c in code:
            text.append(self.inv_vocab[c])

        return "".join(text)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index_split = self.indices[index]

        mel, text, files = self.db[index_split]

        if self.delta > 0:
            mel = mel.transpose(0, 1)
            mel = self.delta_feature(mel.unsqueeze(0))
            mel = mel.squeeze(0).transpose(0, 1)

        if self.normalize:
            mel = self.spec_normalize(mel)

        token = []
        for ch in text:
            try:
                token.append(self.vocab[ch])

            except KeyError:
                token.append(UNK_TOKEN)

        if self.align_db is not None:
            align = self.align_db[index][0]

            return mel, token, text, files, align

        else:
            return mel, token, text, files, None


def collate_data(batch):
    max_mel_len = max(b[0].shape[0] for b in batch)
    max_token_len = max(len(b[1]) for b in batch)

    batch_size = len(batch)
    n_mels = batch[0][0].shape[1]

    mels = torch.zeros(batch_size, max_mel_len, n_mels, dtype=torch.float32)
    tokens = torch.zeros(batch_size, max_token_len, dtype=torch.int64)

    mel_lengths = torch.zeros(batch_size, dtype=torch.int64)
    token_lengths = torch.zeros(batch_size, dtype=torch.int64)

    texts = []
    files = []

    for i, b in enumerate(batch):
        mel, token, text, file, _ = b

        mel_len = mel.shape[0]
        token_len = len(token)

        mels[i, :mel_len] = mel
        tokens[i, :token_len] = torch.tensor(token)
        texts.append(text)
        files.append(file)

        mel_lengths[i] = mel_len
        token_lengths[i] = token_len

    return mels, tokens, mel_lengths, token_lengths, texts, files


def make_emit_targets(token, align, mask, blank=0):
    token_align = []
    token_i = 0

    if len(token) < 1:
        return [], []

    for i, a in enumerate(align):
        if a == token[token_i]:
            token_align.append(i)
            token_i += 1

            if token_i == len(token):
                break

    new_targets = []
    force_emits = []
    emit_i = 0

    for i, (a, m) in enumerate(zip(align, mask)):
        if m == 1 or i in token_align:
            new_targets.append(a)

            if m == 1:
                force_emits.append(emit_i)
                emit_i += 1

            else:
                force_emits.append(-1)
                emit_i += 1

        else:
            force_emits.append(-1)

    return new_targets, force_emits


def make_block_mask(batch_size, n_block, block_size):
    r = torch.randperm(batch_size * n_block * block_size).view(
        batch_size, n_block, block_size
    )
    r = r.argsort(2).view(batch_size, n_block * block_size)
    mask_threshold = torch.multinomial(
        torch.ones(block_size), batch_size, replacement=True
    )

    return (r > mask_threshold.unsqueeze(1)).to(torch.int64)


def get_symbol(state, targets_list):
    if state % 2 == 0:
        symbol = 0

    else:
        symbol = targets_list[state // 2]

    return symbol

    return state


def collate_data_imputer(batch):
    max_mel_len = max(b[0].shape[0] for b in batch)
    max_token_len = max(len(b[1]) for b in batch)

    batch_size = len(batch)
    n_mels = batch[0][0].shape[1]

    mels = torch.zeros(batch_size, max_mel_len, n_mels, dtype=torch.float32)
    tokens = torch.zeros(batch_size, max_token_len, dtype=torch.int64)

    max_mel_len_reduce = math.ceil(max_mel_len / 4)
    force_emits = torch.full((batch_size, max_mel_len_reduce), -1, dtype=torch.int64)
    token_in = torch.zeros_like(force_emits)

    mel_lengths = torch.zeros(batch_size, dtype=torch.int64)
    token_lengths = torch.zeros(batch_size, dtype=torch.int64)

    texts = []
    files = []

    mask = make_block_mask(batch_size, math.ceil(max_mel_len / 4 / 8), 8)[
        :, :max_mel_len_reduce
    ]

    for i, (b, m) in enumerate(zip(batch, mask)):
        mel, token, text, file, align = b

        mel_len = mel.shape[0]

        mels[i, :mel_len] = mel

        token_len = len(token)
        tokens[i, :token_len] = torch.tensor(token)

        texts.append(text)
        files.append(file)

        mel_lengths[i] = mel_len
        token_lengths[i] = token_len

        align_t = torch.tensor(align, dtype=torch.int64)
        mel_len_reduce = math.ceil(mel_len / 4)
        m = m[:mel_len_reduce]
        force_emits[i, : len(align)] = (1 - m) * -1 + m * align_t
        symbols = [get_symbol(s, token) for s in align]
        token_in[i, : len(align)] = (1 - m) * 1 + m * torch.tensor(
            symbols, dtype=torch.int64
        )

    return (
        mels,
        token_in,
        tokens,
        force_emits,
        mel_lengths,
        token_lengths,
        texts,
        files,
    )


"""def collate_data_imputer(batch):
    max_mel_len = max(b[0].shape[0] for b in batch)

    batch_size = len(batch)
    n_mels = batch[0][0].shape[1]

    max_mel_len_reduce = math.ceil(max_mel_len / 4)

    mels = torch.zeros(batch_size, max_mel_len, n_mels, dtype=torch.float32)

    mel_lengths = torch.zeros(batch_size, dtype=torch.int64)
    targets_ctc_lengths = torch.zeros(batch_size, dtype=torch.int64)

    texts = []
    files = []

    mask_block = make_block_mask(batch_size, math.ceil(max_mel_len / 4 / 8), 8)[
        :, :max_mel_len_reduce
    ]
    mask = torch.zeros_like(mask_block)
    new_tokens = []

    targets_ce = torch.zeros((batch_size, max_mel_len_reduce), dtype=torch.int64)

    # max_text_len = max(len(b[1]) for b in batch)
    # targets_full = torch.zeros((batch_size, max_text_len))

    for i, (b, m) in enumerate(zip(batch, mask_block.tolist())):
        mel, token, text, file, align, j = b

        mel_len = mel.shape[0]

        mels[i, :mel_len] = mel

        texts.append(text)
        files.append(file)

        mel_lengths[i] = mel_len

        mel_len_reduce = math.ceil(mel_len / 4)

        mask[:, :mel_len_reduce] = torch.tensor(m[:mel_len_reduce])
        new_tok = ctc_decode(make_ctc_targets(align, m[:mel_len_reduce]))
        new_tokens.append(new_tok)
        targets_ce[i, : len(align)] = torch.tensor(align)

        # targets_full[i, : len(token)] = torch.tensor(token)

    token_in = (1 - mask) * 1 + mask * targets_ce

    max_token_len = max(len(l) for l in new_tokens)
    targets_ctc = torch.zeros(batch_size, max_token_len, dtype=torch.int64)

    for i, token in enumerate(new_tokens):
        token_len = len(token)
        targets_ctc[i, :token_len] = torch.tensor(token)
        targets_ctc_lengths[i] = token_len

    return (
        mels,
        token_in,
        targets_ctc,
        targets_ce,
        mask,
        mel_lengths,
        targets_ctc_lengths,
        texts,
        files,
        # targets_full,
    )"""
