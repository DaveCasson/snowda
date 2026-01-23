from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from snowda.core.fsca_models import FscaState, initialize_model_arrays
from snowda.core.particle_filter import ParticleFilterCore
from snowda.schaake_shuffle.core import (
    apply_permutation,
    compute_shuffle_permutation,
    schaake_shuffle,
)
from snowda.schaake_shuffle.config import SchaakeShuffleConfig
from snowda.workflows.run_snow5_da import Snow5DAEngine, Snow5Particle


def test_marginal_preservation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4))
    Y = rng.standard_normal((5, 4))
    Xss, _ = schaake_shuffle(X, Y)
    assert np.allclose(np.sort(Xss, axis=0), np.sort(X, axis=0))


def test_permutation_validity():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((6, 3))
    B = compute_shuffle_permutation(Y)
    assert B.shape == Y.shape
    for col in range(B.shape[1]):
        assert set(B[:, col]) == set(range(B.shape[0]))


def test_reproducibility_with_ties():
    Y = np.ones((4, 2))
    first = compute_shuffle_permutation(Y)
    second = compute_shuffle_permutation(Y)
    assert np.array_equal(first, second)


def test_integration_schaake_shuffle(tmp_path: Path):
    time = datetime(2000, 1, 15)
    hru_ids = np.array([10, 20])
    n_members = 3
    swe_values = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]])
    snow_depth_values = swe_values * 10.0

    forcing_ds = xr.Dataset(
        coords={
            "time": [np.datetime64("2000-01-01")],
            "hru": hru_ids,
        }
    )
    forcing_ds = forcing_ds.assign_coords(hruId=("hru", hru_ids))

    particles: list[Snow5Particle] = []
    for idx in range(n_members):
        ds = xr.Dataset(
            {
                "swe": (("time", "hru"), swe_values[idx : idx + 1]),
                "snow_depth": (("time", "hru"), snow_depth_values[idx : idx + 1]),
            },
            coords={
                "time": [np.datetime64(time)],
                "hru": hru_ids,
            },
        )
        model_arrays = initialize_model_arrays(np.zeros(len(hru_ids)))
        model_arrays["swe"] = swe_values[idx].astype("float64").copy()
        model_arrays["snow_depth"] = snow_depth_values[idx].astype("float64").copy()
        particles.append(
            Snow5Particle(
                label=f"p{idx}",
                forcing_path=tmp_path / f"forcing_{idx}.nc",
                forcing_ds=forcing_ds,
                model_arrays=model_arrays,
                fsca_states=[FscaState() for _ in hru_ids],
                outputs=[ds],
            )
        )

    archive_times = pd.date_range(time - timedelta(days=30), periods=60, freq="D")
    archive_data = np.vstack(
        [np.linspace(low, high, len(hru_ids)) for low, high in zip(range(60), range(60, 120))]
    )
    archive_ds = xr.Dataset(
        {"swe": (("time", "hru"), archive_data.astype("float64"))},
        coords={"time": archive_times, "hru": hru_ids},
    )
    archive_path = tmp_path / "archive.nc"
    archive_ds.to_netcdf(archive_path)

    shuffle_config = SchaakeShuffleConfig.from_config(
        {
            "use_schaake_shuffle": True,
            "schaake_window_days": 15,
            "schaake_seed": 123,
            "schaake_archive_source": str(archive_path),
            "schaake_target_variables": ["swe"],
            "schaake_apply_same_permutation_to": ["snow_depth"],
        },
        tmp_path,
        "swe",
    )

    engine = Snow5DAEngine(
        particles=particles,
        ens_labels=[p.label for p in particles],
        particle_filter=ParticleFilterCore(),
        assimilation_times=[time],
        assimilation_var="swe",
        modeled_var="swe",
        observation_da=xr.DataArray(
            np.zeros((1, len(hru_ids))),
            coords={"time": [np.datetime64(time)], "hru": hru_ids},
            dims=("time", "hru"),
            name="swe",
        ),
        z_var=1.0,
        n_eff_threshold=0.5,
        fsca_model="ssnowd",
        helbig_params=None,
        helbig_settings=None,
        snow_density=300.0,
        pf_resample=False,
        state_mean_name="mean_state",
        state_var_name="var_state",
        obs_var_name="swe",
        schaake_shuffle_config=shuffle_config,
    )

    seed = shuffle_config.seed or 0
    engine._schaake_rng = np.random.default_rng(seed)

    captured: dict[str, np.ndarray] = {}
    original_sample = engine._schaake_archive.sample_Y

    def _spy_sample_Y(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        captured["reference"] = np.asarray(result, dtype="float64").copy()
        return result

    engine._schaake_archive.sample_Y = _spy_sample_Y

    pre_swe, _ = engine._collect_variable_matrix("swe", time)
    pre_snow_depth, _ = engine._collect_variable_matrix("snow_depth", time)
    assert engine._apply_schaake_shuffle(time)
    post_swe, _ = engine._collect_variable_matrix("swe", time)
    post_snow_depth, _ = engine._collect_variable_matrix("snow_depth", time)

    reference = captured.get("reference")
    assert reference is not None
    expected_swe, permutation = schaake_shuffle(pre_swe, reference)
    expected_snow_depth = apply_permutation(pre_snow_depth, permutation)

    assert np.allclose(np.sort(post_swe, axis=0), np.sort(pre_swe, axis=0))
    assert np.allclose(post_swe, expected_swe)
    assert np.allclose(post_snow_depth, expected_snow_depth)
