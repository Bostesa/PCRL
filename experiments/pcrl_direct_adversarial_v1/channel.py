"""The trainable auxiliary channel: the frozen `A0` map, refined.

Two widths. **Width 16** *is* the frozen `A0` mapper, parameter for parameter, so its
starting point is the released 2018 channel bit-exactly. **Width 8** is that same
mapper followed by one deterministic **training-only** PCA projection of the `A0`
output, expressed as a linear layer so the whole path stays differentiable. The PCA
sees the representation-fitting rows and no reserved label.

The source heads read the released channel `Z` **alone**. They never see `H`, so
copying the service output cannot satisfy the auxiliary-capability objective.

`transform(t, hA, arm)` is the same contract the transport pipeline uses for every
other interface, so a fitted channel can be carried to another year without refitting.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .inputs import A0_WIDTH, PCA_WIDTH, SOURCE_TASKS, standardize

WIDTHS = (16, 8)


def pca_projection(channel: np.ndarray, width: int):
    """Deterministic top-`width` PCA of the frozen channel on representation-fit rows."""
    x = np.asarray(channel, np.float64)
    mean = x.mean(0)
    centred = x - mean
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    basis = np.ascontiguousarray(vt[:width].T)                       # (A0_WIDTH, width)
    total = float((singular ** 2).sum())
    kept = float((singular[:width] ** 2).sum())
    return {'mean': mean, 'basis': basis, 'singular_values': singular.tolist(),
            'explained_variance_ratio': kept / total if total > 0 else None}


class AdversarialChannel(nn.Module):
    """`x -> mapper -> (optional PCA compressor) -> Z`, with Z-only source heads."""

    def __init__(self, width: int, mapper: nn.Module, heads: dict, compressor=None):
        super().__init__()
        if width not in WIDTHS:
            raise ValueError(f'width {width} is not one of {WIDTHS}')
        self.width = width
        self.mapper = copy.deepcopy(mapper)
        self.compress = compressor
        self.heads = nn.ModuleDict({name: copy.deepcopy(head) for name, head in heads.items()})
        if tuple(self.heads) != SOURCE_TASKS:
            raise ValueError('source heads must be exactly the two authorised tasks')

    def encode(self, x):
        z = self.mapper(x)
        return z if self.compress is None else self.compress(z)

    def forward(self, x):
        z = self.encode(x)
        return z, {name: head(z) for name, head in self.heads.items()}

    @torch.no_grad()
    def release(self, x) -> np.ndarray:
        return self.encode(x).numpy().astype(np.float64)


def build_channel(width: int, a0: dict, reference: np.ndarray) -> dict:
    """Common starting point for every policy at this width.

    Width 16 starts **at** `A0`. Width 8 starts at the PCA compression of the same
    `A0` output; its heads are the exact algebraic image of the `A0` heads under that
    projection, so both widths start from one object rather than two unrelated fits.
    """
    if width == A0_WIDTH:
        model = AdversarialChannel(width, a0['mapper'], a0['heads'], None)
        return {'model': model, 'projection': None,
                'initialisation': 'identical to the frozen A0 mapper and heads'}

    pca = pca_projection(reference, width)
    basis = torch.tensor(pca['basis'], dtype=torch.float32)          # (16, width)
    mean = torch.tensor(pca['mean'], dtype=torch.float32)            # (16,)
    compressor = nn.Linear(A0_WIDTH, width)
    with torch.no_grad():
        compressor.weight.copy_(basis.T)
        compressor.bias.copy_(-(mean @ basis))
    heads = {}
    for name, head in a0['heads'].items():
        new = nn.Linear(width, 1)
        with torch.no_grad():
            new.weight.copy_(head.weight @ basis)
            new.bias.copy_(head.bias + head.weight @ mean)
        heads[name] = new
    model = AdversarialChannel(width, a0['mapper'], heads, compressor)
    return {'model': model, 'projection': pca,
            'initialisation': ('A0 mapper followed by one deterministic training-only PCA '
                               'projection of the A0 output; heads are the algebraic image of '
                               'the A0 heads under that projection')}


class TransportableChannel:
    """`transform(T, hA, arm)` over a dict of fitted channels, for cross-year transport."""

    def __init__(self, input_mean, input_scale, models: dict):
        self.input_mean = np.asarray(input_mean, np.float64)
        self.input_scale = np.asarray(input_scale, np.float64)
        self.models = models
        self.maps = {name: None for name in models}

    def features(self, t, ha):                      # signature parity with SpectralModel
        return np.asarray(t, np.float64)

    def transform(self, t, ha, arm) -> np.ndarray:
        x = standardize(t, self.input_mean, self.input_scale)
        return self.models[arm].release(x)

    @property
    def arms(self):
        return tuple(sorted(self.models))


def state_of(model: AdversarialChannel) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_state(model: AdversarialChannel, state: dict) -> AdversarialChannel:
    model.load_state_dict(state)
    return model


__all__ = ['WIDTHS', 'AdversarialChannel', 'build_channel', 'pca_projection',
           'TransportableChannel', 'state_of', 'load_state', 'PCA_WIDTH']
