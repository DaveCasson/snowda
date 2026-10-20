"""Thin wrappers around the ``scores`` verification package.

This is the only snowda module that imports ``scores``. Argument order is
``fcst, obs`` as in ``scores`` (forecast/simulation first).

Kling–Gupta Efficiency uses Gupta et al. (2009): variability is the ratio of
standard deviations (``alpha``), not the Kling et al. (2012) coefficient-of-
variation form (KGE').
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import numpy as np
import scores
import xarray as xr
from scores.categorical import BinaryContingencyManager
from scores.categorical import probability_of_detection as _probability_of_detection
from scores.continuous import additive_bias as _additive_bias
from scores.continuous import kge as _kge
from scores.continuous import mae as _mae
from scores.continuous import nse as _nse
from scores.continuous import pbias as _pbias
from scores.continuous import rmse as _rmse
from scores.continuous.correlation import pearsonr as _pearsonr
from scores.probability import brier_score as _brier_score
from scores.probability import brier_score_for_ensemble as _brier_score_for_ensemble
from scores.probability import crps_for_ensemble as _crps_for_ensemble
from scores.probability import rank_histogram as _rank_histogram
from scores.probability import roc_curve_data as _roc_curve_data

DEFAULT_REDUCE_DIMS = "time"
DEFAULT_ENSEMBLE_DIM = "member"
DEFAULT_CRPS_METHOD = "fair"
KGE_METHOD = "2009"

HYDROLOGY_DETERMINISTIC_METRICS = (
    "nse",
    "kge",
    "rmse",
    "mae",
    "pbias",
    "additive_bias",
    "pearsonr",
)

_KGE_PARAMS = inspect.signature(_kge).parameters


def _as_dataarray(data: xr.DataArray | xr.Dataset, name: str) -> xr.DataArray:
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError(
                f"{name} Dataset must contain exactly one data variable "
                f"(got {list(data.data_vars)})."
            )
        data = next(iter(data.data_vars.values()))
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{name} must be an xarray DataArray (got {type(data)!r}).")
    return data


def nse(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
    **kwargs: Any,
) -> xr.DataArray:
    """Nash–Sutcliffe Efficiency (Nash and Sutcliffe, 1970)."""
    return _nse(fcst, obs, reduce_dims=reduce_dims, **kwargs)


def kge(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
    include_components: bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Kling–Gupta Efficiency, Gupta et al. (2009) std-ratio form.

    When ``include_components`` is True (the default), returns a Dataset with
    ``kge``, ``rho``, ``alpha``, and ``beta``.
    """
    if "method" in _KGE_PARAMS and "method" not in kwargs:
        kwargs["method"] = KGE_METHOD
    return _kge(
        fcst,
        obs,
        reduce_dims=reduce_dims,
        include_components=include_components,
        **kwargs,
    )


def rmse(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
    **kwargs: Any,
) -> xr.DataArray:
    """Root mean squared error."""
    return _rmse(fcst, obs, reduce_dims=reduce_dims, **kwargs)


def mae(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
    **kwargs: Any,
) -> xr.DataArray:
    """Mean absolute error."""
    return _mae(fcst, obs, reduce_dims=reduce_dims, **kwargs)


def pbias(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
    **kwargs: Any,
) -> xr.DataArray:
    """Percent bias: ``100 * sum(fcst - obs) / sum(obs)``."""
    return _pbias(fcst, obs, reduce_dims=reduce_dims, **kwargs)


def additive_bias(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
    **kwargs: Any,
) -> xr.DataArray:
    """Mean error (additive bias): ``mean(fcst - obs)``."""
    return _additive_bias(fcst, obs, reduce_dims=reduce_dims, **kwargs)


def pearsonr(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
    **kwargs: Any,
) -> xr.DataArray:
    """Pearson correlation coefficient."""
    return _pearsonr(fcst, obs, reduce_dims=reduce_dims, **kwargs)


def compute_deterministic_metrics(
    fcst: xr.DataArray | xr.Dataset,
    obs: xr.DataArray | xr.Dataset,
    *,
    reduce_dims: str | tuple[str, ...] = DEFAULT_REDUCE_DIMS,
) -> xr.Dataset:
    """Compute the default hydrology deterministic metric suite.

    Pairwise-complete samples only contribute to the scores. The returned
    Dataset includes scalar metrics plus KGE components, with ``n_pairs`` and
    the evaluation period in ``attrs``.
    """
    fcst = _as_dataarray(fcst, "fcst")
    obs = _as_dataarray(obs, "obs")
    reduce_name = reduce_dims if isinstance(reduce_dims, str) else reduce_dims[0]
    if reduce_name not in fcst.dims:
        raise ValueError(
            f"Forecast is missing reduce dimension {reduce_name!r} "
            f"(dims={list(fcst.dims)})."
        )
    if reduce_name not in obs.dims:
        raise ValueError(
            f"Observations are missing reduce dimension {reduce_name!r} "
            f"(dims={list(obs.dims)})."
        )

    aligned_fcst, aligned_obs = xr.align(fcst, obs, join="inner")
    valid = np.isfinite(aligned_fcst.values) & np.isfinite(aligned_obs.values)
    n_pairs = int(np.sum(valid))
    if n_pairs < 2:
        raise ValueError(
            f"Need at least 2 pairwise-complete samples to score, got {n_pairs}."
        )

    data_vars: dict[str, xr.DataArray] = {
        "nse": nse(aligned_fcst, aligned_obs, reduce_dims=reduce_dims),
        "rmse": rmse(aligned_fcst, aligned_obs, reduce_dims=reduce_dims),
        "mae": mae(aligned_fcst, aligned_obs, reduce_dims=reduce_dims),
        "pbias": pbias(aligned_fcst, aligned_obs, reduce_dims=reduce_dims),
        "additive_bias": additive_bias(
            aligned_fcst, aligned_obs, reduce_dims=reduce_dims
        ),
        "pearsonr": pearsonr(aligned_fcst, aligned_obs, reduce_dims=reduce_dims),
    }
    kge_result = kge(
        aligned_fcst,
        aligned_obs,
        reduce_dims=reduce_dims,
        include_components=True,
    )
    if isinstance(kge_result, xr.Dataset):
        for name, da in kge_result.data_vars.items():
            data_vars[name] = da
    else:
        data_vars["kge"] = kge_result

    ds = xr.Dataset(data_vars)
    ds.attrs["n_pairs"] = n_pairs
    ds.attrs["kge_method"] = KGE_METHOD
    ds.attrs["kge_reference"] = (
        "Gupta et al. (2009); scores uses std-ratio alpha, not Kling (2012) KGE'"
    )
    ds.attrs["scores_version"] = getattr(scores, "__version__", "unknown")
    ds.attrs["reduce_dims"] = str(reduce_dims)
    if "time" in aligned_fcst.dims:
        ds.attrs["time_start"] = str(aligned_fcst["time"].min().values)
        ds.attrs["time_end"] = str(aligned_fcst["time"].max().values)
    return ds


def skill_score(
    score: xr.DataArray,
    score_ref: xr.DataArray,
) -> xr.DataArray:
    """Skill score ``1 - score / score_ref``; NaN where the reference is 0 or non-finite."""
    aligned_score, aligned_ref = xr.align(score, score_ref, join="inner")
    ref = aligned_ref.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1.0 - aligned_score / ref
    return result.where(np.isfinite(ref) & (ref != 0))


def crps_for_ensemble(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    ensemble_member_dim: str = DEFAULT_ENSEMBLE_DIM,
    method: str = DEFAULT_CRPS_METHOD,
    reduce_dims: str | Sequence[str] | None = None,
    preserve_dims: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """CRPS for an ensemble; default method is fair CRPS (Ferro 2014)."""
    return _crps_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim,
        method=method,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        **kwargs,
    )


def brier_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | Sequence[str] | None = None,
    preserve_dims: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Brier score for probability forecasts in ``[0, 1]`` vs binary observations."""
    return _brier_score(
        fcst,
        obs,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        **kwargs,
    )


def brier_score_for_ensemble(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    event_thresholds: float | Sequence[float],
    *,
    ensemble_member_dim: str = DEFAULT_ENSEMBLE_DIM,
    fair_correction: bool = True,
    reduce_dims: str | Sequence[str] | None = None,
    preserve_dims: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Brier score from real-valued ensemble members and a flow threshold."""
    return _brier_score_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim,
        event_thresholds,
        fair_correction=fair_correction,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        **kwargs,
    )


def _as_binary_events(data: xr.DataArray, name: str) -> xr.DataArray:
    data = _as_dataarray(data, name)
    numeric = data.astype(float)
    if np.any((numeric != 0) & (numeric != 1) & np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only binary values {{0, 1, NaN}}.")
    return numeric


def probability_of_detection(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | Sequence[str] | None = None,
    preserve_dims: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Hit rate: hits / (hits + misses). Forecast and obs must be binary."""
    return _probability_of_detection(
        fcst,
        obs,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        **kwargs,
    )


def false_alarm_ratio(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: str | Sequence[str] | None = None,
    preserve_dims: str | Sequence[str] | None = None,
) -> xr.DataArray:
    """False alarm ratio: false alarms / (hits + false alarms)."""
    manager = BinaryContingencyManager(
        _as_binary_events(fcst, "fcst"),
        _as_binary_events(obs, "obs"),
    )
    table = manager.transform(reduce_dims=reduce_dims, preserve_dims=preserve_dims)
    return table.false_alarm_ratio()


DEFAULT_RELIABILITY_BIN_EDGES = np.linspace(0.0, 1.0, 11)


def reliability_diagram(
    probability: xr.DataArray,
    observed: xr.DataArray,
    *,
    sample_dim: str = "init_time",
    bin_edges: Sequence[float] | None = None,
) -> xr.Dataset:
    """Reliability diagram for probability forecasts of a binary event.

    Bins are ``[0, 0.1), ..., [0.9, 1.0]`` unless ``bin_edges`` is supplied.
    A forecast of exactly 1 falls in the last bin. Each non-empty bin stores
    the mean forecast probability, the observed event frequency, and the
    count. Pairs with a missing forecast or observation are omitted. Every
    dimension other than ``sample_dim`` is preserved.
    """
    probability = _as_dataarray(probability, "probability")
    observed = _as_dataarray(observed, "observed")
    probability, observed = xr.align(probability, observed, join="inner")
    if sample_dim not in probability.dims:
        raise ValueError(
            f"Probability forecasts are missing sample dimension {sample_dim!r} "
            f"(dims={list(probability.dims)})."
        )
    edges = np.asarray(
        DEFAULT_RELIABILITY_BIN_EDGES if bin_edges is None else bin_edges,
        dtype=float,
    )
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")
    n_bins = int(edges.size - 1)
    preserved = [dim for dim in probability.dims if dim != sample_dim]
    p_values = np.asarray(
        probability.transpose(*preserved, sample_dim).values, dtype=float
    )
    o_values = np.asarray(
        observed.transpose(*preserved, sample_dim).values, dtype=float
    )
    group_shape = p_values.shape[:-1]
    n_groups = int(np.prod(group_shape)) if group_shape else 1
    p_flat = p_values.reshape(n_groups, -1)
    o_flat = o_values.reshape(n_groups, -1)
    mean_forecast = np.full((n_groups, n_bins), np.nan)
    observed_frequency = np.full((n_groups, n_bins), np.nan)
    count = np.zeros((n_groups, n_bins), dtype=np.int64)
    event_frequency = np.full(n_groups, np.nan)
    for i in range(n_groups):
        valid = np.isfinite(p_flat[i]) & np.isfinite(o_flat[i])
        p_sample = p_flat[i][valid]
        o_sample = o_flat[i][valid]
        if p_sample.size == 0:
            continue
        event_frequency[i] = float(np.mean(o_sample))
        bin_index = np.digitize(p_sample, edges, right=False) - 1
        bin_index = np.where(p_sample >= edges[-1], n_bins - 1, bin_index)
        in_range = (bin_index >= 0) & (bin_index < n_bins)
        p_sample = p_sample[in_range]
        o_sample = o_sample[in_range]
        bin_index = bin_index[in_range]
        for bin_id in range(n_bins):
            selected = bin_index == bin_id
            n_selected = int(np.sum(selected))
            count[i, bin_id] = n_selected
            if n_selected == 0:
                continue
            mean_forecast[i, bin_id] = float(np.mean(p_sample[selected]))
            observed_frequency[i, bin_id] = float(np.mean(o_sample[selected]))
    bin_dims = tuple(preserved) + ("probability_bin",)
    out_shape = group_shape + (n_bins,)
    diagram = xr.Dataset(
        {
            "mean_forecast": (bin_dims, mean_forecast.reshape(out_shape)),
            "observed_frequency": (bin_dims, observed_frequency.reshape(out_shape)),
            "count": (bin_dims, count.reshape(out_shape)),
            "event_frequency": (tuple(preserved), event_frequency.reshape(group_shape)),
        },
        coords={
            **{dim: probability[dim] for dim in preserved},
            "probability_bin": np.arange(n_bins),
            "bin_left": ("probability_bin", edges[:-1]),
            "bin_right": ("probability_bin", edges[1:]),
        },
    )
    diagram["mean_forecast"].attrs["long_name"] = "mean forecast probability in bin"
    diagram["observed_frequency"].attrs["long_name"] = "observed event frequency in bin"
    diagram["count"].attrs["long_name"] = "number of forecast-observation pairs in bin"
    diagram["event_frequency"].attrs["long_name"] = "observed event frequency"
    return diagram


def roc_diagram(
    probability: xr.DataArray,
    observed: xr.DataArray,
    *,
    reduce_dims: str | Sequence[str] = "init_time",
    thresholds: str | Sequence[float] = "auto",
) -> xr.Dataset:
    """ROC curve for a probability forecast of a binary event.

    ``POD`` is the hit rate and ``POFD`` is the false-alarm rate as the
    decision threshold varies. The decision-threshold coordinate is
    ``probability_threshold``. Months or thresholds with no observed events
    keep a NaN AUC.
    """
    reduce = (reduce_dims,) if isinstance(reduce_dims, str) else tuple(reduce_dims)
    diagram = _roc_curve_data(
        probability,
        observed,
        thresholds,
        reduce_dims=reduce,
    )
    return diagram.rename({"threshold": "probability_threshold"})


def rank_histogram(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    ensemble_member_dim: str = DEFAULT_ENSEMBLE_DIM,
    reduce_dims: Sequence[str] | None = None,
    preserve_dims: Sequence[str] | None = None,
) -> xr.DataArray:
    """Relative frequency of the observation's rank in an ensemble.

    Rank 1 means the observation is below every member. Rank ``n + 1`` means
    it is above every member of an ``n``-member ensemble. Ties share their
    weight equally across the tied ranks, so a calibrated ensemble is flat.
    """
    return _rank_histogram(
        fcst,
        obs,
        ensemble_member_dim,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
