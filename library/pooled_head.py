"""Attention-pooled classification head, as an alternative to flattening the encoder output.

Why there is a choice at all
---------------------------
The original head flattens every encoder position and feeds the whole thing to one Linear layer, so
each (slot, embedding dimension) pair gets its own weight. That parameterisation is only meaningful
when slot i means the same thing in every genome. Measured on 300 K. pneumoniae genomes:

    channel        width   most common token's share of the slot   entropy per slot
    k-mer           1000                     1.000                    0.00 bits
    graph (snv)       76                     0.222                    3.81 bits
    PC              1558                     0.013                    6.87 bits

The k-mer channel is fixed-slot by construction (slot i is the i-th selected k-mer), so the flatten
head fits it exactly. The other two are packed -- slot i holds whichever gene came i-th in that
genome -- so a position-specific weight has nothing stable to learn, while the PC channel still
takes 59% of the head's parameters (6.4 M of 10.8 M at embed_size 64). The result on that dataset was
a model whose decisions were reproduced 98.7% of the time by a single binary k-mer feature, with the
PC channel collapsed to a constant offset and the graph channel to another.

Pooling removes the dependence on slot position: each channel is summarised by a weighted mean of its
token vectors, the weights coming from a learned query. That makes the head permutation-invariant and
length-independent, drops it from ~10.8 M parameters to ~12 k, and lets a packed channel contribute.
What it gives up is order: the flatten head can in principle use the relative position of two genes,
and pooling cannot.

Both heads are available so the two can be compared on the same database; `--head flatten` stays the
default and reproduces the original model exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """One learned query per channel; returns a fixed embed_size vector per channel.

    Padding positions are excluded from the weighting. A genome can legitimately have no tokens at
    all in a channel (a genome with no resistance-gene hit has an empty graph sentence), in which
        case the pooled vector is zero rather than NaN -- masking every position and then taking a
    softmax is what would produce NaN, so the scores are filled with a finite floor instead and the
    whole vector is zeroed afterwards.
    """

    def __init__(self, n_channels: int, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.query = nn.Parameter(torch.empty(n_channels, embed_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # named like the torch convention so the model-randomisation sanity check can re-initialise
        # the pooling query along with the rest of the head
        with torch.no_grad():
            self.query.normal_(0.0, self.embed_size ** -0.5)

    def forward(self, enc: torch.Tensor, keep: torch.Tensor, channel: int) -> torch.Tensor:
        """enc: (N, L, E) encoder output; keep: (N, L) True where the position is a real token."""
        scores = enc @ self.query[channel] / math.sqrt(self.embed_size)      # (N, L)
        scores = scores.masked_fill(~keep, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)                 # (N, L, 1)
        pooled = (enc * weights).sum(dim=1)                                  # (N, E)
        return pooled * keep.any(dim=1, keepdim=True).to(pooled.dtype)


def flat_mask(src_mask: torch.Tensor) -> torch.Tensor:
    """(N, 1, 1, L) mask as make_src_mask builds it -> (N, L) boolean keep-mask."""
    return src_mask.reshape(src_mask.shape[0], -1).bool()


def head_of_state_dict(state: dict, embed_size: int, n_channels: int) -> str:
    """Which head a checkpoint was trained with, read off fc1's shape.

    Self-describing beats a side file: a checkpoint and a recorded flag can drift apart, the weight
    shape cannot.
    """
    for key in ("fc1.weight", "module.fc1.weight"):
        w = state.get(key)
        if w is not None:
            return "pooled" if int(w.shape[1]) == n_channels * embed_size else "flatten"
    return "flatten"
