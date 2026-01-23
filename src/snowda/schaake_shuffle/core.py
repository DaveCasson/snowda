from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _flatten_for_shuffle(
    arr: Sequence, dtype: np.dtype | None = None
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return flattened view of array values (members × flattened sites)."""
    data = np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    if data.ndim < 2:
        raise ValueError("Input array must have at least 2 dimensions (members, sites).")
    return data.reshape(data.shape[0], -1), data.shape


def _reshape_from_flat(flat: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore flattened array into its original multi-dimensional layout."""
    return flat.reshape(original_shape)


def _jitter_for_ranking(shape: tuple[int, ...]) -> np.ndarray:
    """Compute the jitter matrix used to break ties deterministically."""
    members, sites = shape
    member_idx = np.arange(members, dtype="float64").reshape(-1, 1)
    site_idx = np.arange(sites, dtype="float64").reshape(1, -1)
    eps = np.finfo("float64").eps
    return eps * (member_idx + 1e-3 * site_idx)


def compute_shuffle_permutation(Y: Sequence) -> np.ndarray:
    """
    Return the permutation indices derived from the dependence template ``Y``.

    ``Y`` is expected to be shaped (n_members, n_sites, ...). The returned array
    always has the same layout as ``Y`` and contains integer indices describing
    the order of members sorted by ``Y`` (ascending).
    """
    flat, shape = _flatten_for_shuffle(Y, dtype=np.float64)
    if flat.size == 0:
        raise ValueError("Y must contain at least one member/site.")
    scale = float(np.nanstd(flat))
    scale = max(1.0, 0.0 if np.isnan(scale) else scale)
    jitter = _jitter_for_ranking(flat.shape) * scale
    ranked = flat + jitter
    permutation = np.argsort(ranked, axis=0, kind="stable")
    return _reshape_from_flat(permutation, shape)


def apply_shuffle(X: Sequence, B: Sequence) -> np.ndarray:
    """
    Apply the Schaake Shuffle permutation to ``X`` (per-site sorting).

    The output has the same marginal distributions as ``X`` but the sorted order
    matches the dependence encoded by ``B``.
    """
    X_flat, original_shape = _flatten_for_shuffle(X, dtype=np.float64)
    B_flat, _ = _flatten_for_shuffle(B, dtype=np.int64)
    if X_flat.shape != B_flat.shape:
        raise ValueError("X and B must share the same shape.")
    shuffled = np.empty_like(X_flat)
    for col in range(X_flat.shape[1]):
        sorted_vals = np.sort(X_flat[:, col], kind="mergesort")
        shuffled[B_flat[:, col], col] = sorted_vals
    return _reshape_from_flat(shuffled, original_shape)


def apply_permutation(X: Sequence, B: Sequence) -> np.ndarray:
    """
    Reorder ``X`` using the permutation ``B`` without re-sorting each column.
    """
    X_flat, original_shape = _flatten_for_shuffle(X, dtype=np.float64)
    B_flat, _ = _flatten_for_shuffle(B, dtype=np.int64)
    if X_flat.shape != B_flat.shape:
        raise ValueError("X and B must share the same shape.")
    permuted = np.empty_like(X_flat)
    for col in range(X_flat.shape[1]):
        permuted[B_flat[:, col], col] = X_flat[:, col]
    return _reshape_from_flat(permuted, original_shape)


def schaake_shuffle(X: Sequence, Y: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper that builds ``B`` from ``Y`` and applies it to ``X``.
    """
    permutation = compute_shuffle_permutation(Y)
    shuffled = apply_shuffle(X, permutation)
    return shuffled, permutation
