# imputer-pytorch
Implementation of Imputer: Sequence Modelling via Imputation and Dynamic Programming (https://arxiv.org/abs/2002.08926) in PyTorch

# Usage

Imputer loss is basically ctc loss with force emit symbols. (force emit ctc states) So you need to get ctc states. (alignments) You can get it by training ctc models on your data.

Then, you can extract best alignments given input log probabilities and target sequences using `torch_imputer.best_alignment`

```python
def best_alignment(
    log_prob, targets, input_lengths, target_lengths, blank=0, zero_infinity=False
):
    """Get best alignment (maximum probability sequence of ctc states)
       conditioned on log probabilities and target sequences

    Input:
        log_prob (T, N, C): C = number of characters in alphabet including blank
                            T = input length
                            N = batch size
                            log probability of the outputs (e.g. torch.log_softmax of logits)
        targets (N, S): S = maximum number of characters in target sequences
        input_lengths (N): lengths of log_prob
        target_lengths (N): lengths of targets
        blank (int): index of blank tokens (default 0)
        zero_infinity (bool): if true imputer loss will zero out infinities.
                            infinities mostly occur when it is impossible to generate
                            target sequences using input sequences
                            (e.g. input sequences are shorter than target sequences)

    Output:
        best_aligns (List[List[int]]): sequence of ctc states that have maximum probabilties
                                       given log probabilties, and compatible with target sequences"""
```

You can refer to `example/asr/extract_best_align.py`

Then you can train imputer model using `torch_imputer.ImputerLoss` or `torch_imputer.imputer_loss`

```python
def imputer_loss(
    log_prob,
    targets,
    force_emits,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """The Imputer loss

    Parameters:
        log_prob (T, N, C): C = number of characters in alphabet including blank
                            T = input length
                            N = batch size
                            log probability of the outputs (e.g. torch.log_softmax of logits)
        targets (N, S): S = maximum number of characters in target sequences
        force_emits (N, T): sequence of ctc states that should be occur given times
                            that is, if force_emits is state s at time t, only ctc paths
                            that pass state s at time t will be enabled, and will be zero out the rest
                            this will be same as using cross entropy loss at time t
                            value should be in range [-1, 2 * S + 1), valid ctc states
                            -1 will means that it could be any states at time t (normal ctc paths)
        input_lengths (N): lengths of log_prob
        target_lengths (N): lengths of targets
        blank (int): index of blank tokens (default 0)
        reduction (str): reduction methods applied to the output. 'none' | 'mean' | 'sum'
        zero_infinity (bool): if true imputer loss will zero out infinities.
                              infinities mostly occur when it is impossible to generate
                              target sequences using input sequences
                              (e.g. input sequences are shorter than target sequences)
    """
```

You need to appropriately mask best alignment sequences and pass it `force_emits`. You also need to convert best alignment sequences (that is, sequence of ctc states) into sequence of target tokens to use it as an input to the model. You can do it using function like this:

```python
def get_symbol(state, targets_list):
    """Convert sequence of ctc states into sequence of target tokens

    Input:
        state (List[int]): list of ctc states (e.g. from torch_imputer.best_alignment)
        targets_list (List[int]): token indices of targets
                                  (e.g. targets that you will pass to ctc_loss or imputer_loss)
    """

    if state % 2 == 0:
        symbol = 0

    else:
        symbol = targets_list[state // 2]

    return symbol
```

May you can refer to `collate_data_imputer` in `example/asr/dataset.py` to how you can construct data for imputer loss.