"""SparseGPT layer reconstruction, adapted from Frantar & Alistarh (2023).

Reference: https://arxiv.org/abs/2301.00774v3, Algorithm 1; author code
IST-DASLab/sparsegpt/sparsegpt.py (Apache-2.0, included in licenses/).
Changes: CPU-only pure tensor API, caller-supplied Hessian or fixed mask,
exact-count stable block selection instead of author's <= threshold selection,
explicit diagnostics, no quantization, N:M, convolution or model traversal.
This is a declared adaptation, not a complete full-model reproduction.
"""
import math
import torch


def _matrix(value, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.ndim != 2 or not value.is_floating_point()
            or not value.numel() or not torch.isfinite(value).all()):
        raise ValueError(f'{name}: nonempty finite CPU floating matrix required')


@torch.no_grad()
def input_hessian(inputs):
    """2 X.T X / n for valid input-token rows X[n, input_features].

    The author normalizes by sequences instead of tokens. With relative damping
    either positive global normalization yields the same reconstruction. Padding
    must already be removed. Do not center: this is an uncentered second moment.
    """
    _matrix(inputs, 'inputs')
    x = inputs.double()
    return (2.0 / x.shape[0]) * (x.T @ x)


@torch.no_grad()
def sparsegpt_reconstruct(weight, hessian, *, sparsity=None, prune_mask=None,
                          blocksize=128, relative_damping=0.01):
    """Return FP32 weights, bool removal mask and diagnostics without mutation.

    Select exactly one of sparsity in [0,1] or fixed prune_mask (True=remove).
    Fixed-mask mode isolates compensation from mask choice. Adaptive mode chooses
    a mask at each block after earlier compensation; floor(sparsity*block_size)
    entries are selected with stable flattened-order tie breaking.
    """
    _matrix(weight, 'weight'); _matrix(hessian, 'hessian')
    columns = weight.shape[1]
    if hessian.shape != (columns, columns):
        raise ValueError('Hessian must use weight input coordinates')
    if not torch.allclose(hessian, hessian.T, rtol=1e-6, atol=1e-10):
        raise ValueError('Symmetric Hessian required')
    if type(blocksize) is not int or blocksize < 1:
        raise ValueError('Positive integer blocksize required')
    if not math.isfinite(relative_damping) or relative_damping < 0:
        raise ValueError('Nonnegative finite relative damping required')
    if (sparsity is None) == (prune_mask is None):
        raise ValueError('Choose sparsity or a fixed removal mask')
    if sparsity is not None and (not math.isfinite(sparsity) or not 0 <= sparsity <= 1):
        raise ValueError('Sparsity must be in [0,1]')
    if prune_mask is not None and (not isinstance(prune_mask, torch.Tensor)
            or prune_mask.device.type != 'cpu' or prune_mask.dtype != torch.bool
            or prune_mask.shape != weight.shape):
        raise ValueError('Fixed mask must be CPU bool with weight shape')
    w = weight.double().clone(); h = hessian.double().clone()
    if torch.diagonal(h).lt(0).any():
        raise ValueError('Nonnegative Hessian diagonal required')
    dead = torch.diagonal(h).eq(0)
    if dead.any() and h[dead].abs().max() != 0:
        raise ValueError('Zero-energy features must have zero cross moments')
    h[dead, dead] = 1; w[:, dead] = 0
    damping = float(relative_damping * h.diagonal().mean())
    h.diagonal().add_(damping)
    try:
        factor = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    except torch.linalg.LinAlgError as error:
        raise ValueError('Hessian not positive definite after declared damping') from error
    removal = torch.zeros_like(w, dtype=torch.bool)
    predicted = 0.0
    for start in range(0, columns, blocksize):
        stop = min(start + blocksize, columns)
        block = w[:, start:stop].clone(); upper = factor[start:stop, start:stop]
        errors = torch.zeros_like(block); output = torch.empty_like(block)
        if prune_mask is None:
            scores = (block / upper.diagonal()[None, :]).square()
            selected = torch.zeros_like(scores, dtype=torch.bool)
            count = math.floor(scores.numel() * sparsity)
            selected.view(-1)[torch.argsort(scores.flatten(), stable=True)[:count]] = True
        else:
            selected = prune_mask[:, start:stop].clone()
        removal[:, start:stop] = selected
        for j in range(stop-start):
            current = block[:, j].clone()
            chosen = current.masked_fill(selected[:, j], 0)
            error = (current - chosen) / upper[j, j]
            output[:, j] = chosen; errors[:, j] = error
            predicted += float(error.square().sum() / 2)
            block[:, j:] -= error[:, None] * upper[j, j:][None, :]
        w[:, start:stop] = output
        w[:, stop:] -= errors @ factor[start:stop, stop:]
    if not torch.isfinite(w).all() or not torch.isfinite(w.float()).all():
        raise ValueError('Reconstruction overflow')
    return w.float(), removal, {
        'method': 'sparsegpt_declared_adaptation_v1', 'blocksize': blocksize,
        'relative_damping': relative_damping, 'absolute_damping': damping,
        'selection': 'fixed_mask' if prune_mask is not None else 'adaptive_exact_count_per_block',
        'removed_entries': int(removal.sum()), 'realized_zero_fraction': float(w.eq(0).double().mean()),
        'dead_input_features': int(dead.sum()), 'predicted_damped_layer_loss': predicted,
        'scope': 'Local quadratic reconstruction; neither task accuracy nor sparse-kernel speedup',
    }
