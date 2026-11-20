from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from snowda.config.pysummada import load_pysummada_config
from snowda.config.run_modes import (
    check_run_mode,
    esp_da_mode,
    historic_da_mode,
    run_mode_from_provenance,
)
from snowda.core.particle_filter import weighted_ensemble_summary
from snowda.evaluation import evaluate_run, evaluate_run_safely
from snowda.evaluation.ensemble_scores import (
    pit_histogram,
    weighted_crps,
    weighted_pit,
)
from snowda.evaluation.historic import (
    climatology_reference,
    evaluate_historic,
    in_season,
    water_year_of,
)
from snowda.evaluation.weights import (
    QUANTILES,
    effective_weights,
    filter_weight_timeline,
    weighted_quantiles,
)
from snowda.workflows import run_case as run_case_module
from snowda.workflows.run_case import resolve_case_config, run_case
from tests.test_pysummada_config import _build_config

MEMBERS = ["001", "002", "003", "004", "005"]
SCALES = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
START = "2000-10-01"
END = "2004-09-30"


def _brute_crps(x: np.ndarray, w: np.ndarray, y: float) -> float:
    accuracy = float(np.sum(w * np.abs(x - y)))
    spread = 0.5 * float(np.sum(w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :])))
    return accuracy - spread


# ---- scores -----------------------------------------------------------------


def test_crps_of_one_member_is_absolute_error():
    values = np.array([[3.0], [5.0], [1.0]])
    obs = np.array([4.0, 5.0, -1.0])
    crps = weighted_crps(values, effective_weights(values, None), obs)
    np.testing.assert_allclose(crps, [1.0, 0.0, 2.0])


def test_weighted_crps_matches_the_double_sum():
    rng = np.random.default_rng(0)
    values = rng.gamma(2.0, 10.0, size=(40, 7))
    raw = rng.random(size=(40, 7))
    obs = rng.gamma(2.0, 10.0, size=40)
    weights = effective_weights(values, raw)
    crps = weighted_crps(values, weights, obs)
    expected = [_brute_crps(values[t], weights[t], obs[t]) for t in range(40)]
    np.testing.assert_allclose(crps, expected, rtol=1e-10)
    equal = weighted_crps(values, effective_weights(values, None), obs)
    uniform = np.full(7, 1 / 7)
    np.testing.assert_allclose(
        equal, [_brute_crps(values[t], uniform, obs[t]) for t in range(40)], rtol=1e-10
    )


def test_crps_ignores_missing_members_and_flags_missing_obs():
    values = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]])
    obs = np.array([2.0, 2.0, np.nan])
    weights = effective_weights(values, None)
    np.testing.assert_allclose(weights[0], [0.5, 0.0, 0.5])
    crps = weighted_crps(values, weights, obs)
    assert crps[0] == pytest.approx(_brute_crps(np.array([1.0, 3.0]), np.array([0.5, 0.5]), 2.0))
    assert np.isnan(crps[1]) and np.isnan(crps[2])


def test_pit_is_flat_for_a_calibrated_ensemble():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(20000, 20))
    obs = rng.normal(size=20000)
    weights = effective_weights(values, None)
    histogram = pit_histogram(*weighted_pit(values, weights, obs), 10)
    assert histogram.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(histogram, 0.1, atol=0.01)
    # An ensemble that is too narrow puts the observation in the outer bins.
    narrow = pit_histogram(*weighted_pit(0.3 * values, weights, obs), 10)
    assert narrow[0] > 0.25 and narrow[-1] > 0.25 and narrow[5] < 0.05


def test_weighted_quantiles_follow_the_particle_filter_rule():
    rng = np.random.default_rng(2)
    values = rng.random(size=(30, 9))
    raw = rng.random(9)
    weights = effective_weights(values, raw)
    bands = weighted_quantiles(values, weights, (0.05, 0.5, 0.95))
    summary = weighted_ensemble_summary(pd.DataFrame(values), raw)
    np.testing.assert_allclose(bands, summary[["q05", "q50", "q95"]].to_numpy())
    all_on_one = effective_weights(values, np.eye(9)[4])
    np.testing.assert_allclose(
        weighted_quantiles(values, all_on_one, QUANTILES),
        np.repeat(values[:, [4]], len(QUANTILES), axis=1),
    )


# ---- weight timeline --------------------------------------------------------


def _updates() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]],
        index=pd.to_datetime(["2001-03-01", "2001-05-01", "2002-04-01"]),
        columns=["a", "b", "c"],
    )


def test_weights_apply_from_the_update_day_and_reset_each_season():
    times = pd.date_range("2000-10-01", "2002-09-30", freq="D")
    timeline = filter_weight_timeline(_updates(), times, season_start_month=10)
    equal = [1 / 3] * 3
    np.testing.assert_allclose(timeline.loc["2001-02-28"], equal)
    np.testing.assert_allclose(timeline.loc["2001-03-01"], [0.7, 0.2, 0.1])
    np.testing.assert_allclose(timeline.loc["2001-04-30"], [0.7, 0.2, 0.1])
    np.testing.assert_allclose(timeline.loc["2001-09-30"], [0.1, 0.1, 0.8])
    # The May weights stop at the season boundary, not at the next update.
    np.testing.assert_allclose(timeline.loc["2001-10-01"], equal)
    np.testing.assert_allclose(timeline.loc["2002-03-31"], equal)
    np.testing.assert_allclose(timeline.loc["2002-04-01"], [0.2, 0.6, 0.2])
    np.testing.assert_allclose(timeline.sum(axis=1), 1.0)


def test_weights_carry_across_seasons_without_a_season_start():
    times = pd.date_range("2001-09-01", "2001-12-31", freq="D")
    timeline = filter_weight_timeline(_updates(), times, season_start_month=None)
    np.testing.assert_allclose(timeline.loc["2001-12-31"], [0.1, 0.1, 0.8])


def test_resampled_updates_leave_equal_weights():
    times = pd.date_range("2001-02-01", "2001-08-01", freq="D")
    timeline = filter_weight_timeline(
        _updates(), times, resampled=[True, False, False], season_start_month=10
    )
    np.testing.assert_allclose(timeline.loc["2001-03-15"], [1 / 3] * 3)
    np.testing.assert_allclose(timeline.loc["2001-05-15"], [0.1, 0.1, 0.8])


# ---- water years ------------------------------------------------------------


def test_water_year_and_season_windows():
    times = pd.DatetimeIndex(["2013-09-30", "2013-10-01", "2014-04-01", "2014-09-30"])
    np.testing.assert_array_equal(water_year_of(times, 10), [2013, 2014, 2014, 2014])
    np.testing.assert_array_equal(water_year_of(times, 1), [2013, 2013, 2014, 2014])
    np.testing.assert_array_equal(in_season(times, (4, 9)), [True, False, True, True])
    np.testing.assert_array_equal(in_season(times, (11, 2)), [False, False, False, False])


def test_climatology_reference_leaves_out_the_verifying_year():
    times = pd.date_range("2000-10-01", "2003-09-30", freq="D")
    obs = pd.Series(np.asarray(water_year_of(times, 10), dtype=float), index=times)
    reference = climatology_reference(obs, times, water_year_of(times, 10))
    row = reference[times.get_loc(pd.Timestamp("2002-06-01"))]
    assert sorted(row[np.isfinite(row)]) == [2001.0, 2003.0]


# ---- synthetic run directories ----------------------------------------------


def _hydrograph(times: pd.DatetimeIndex) -> np.ndarray:
    """Low winter flow and a June melt peak whose size changes each water year."""
    day = np.asarray(times.dayofyear, dtype=float)
    year_factor = 1.0 + 0.25 * np.sin(np.asarray(water_year_of(times, 10), dtype=float))
    return 20.0 + 400.0 * year_factor * np.exp(-0.5 * ((day - 165.0) / 25.0) ** 2)


def _write_mizuroute(path: Path, times: pd.DatetimeIndex, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "IRFroutedRunoff": xr.DataArray(
                np.asarray(values, dtype=float).reshape(len(times), 1),
                dims=("time", "seg"),
                coords={"time": times, "seg": [0]},
                attrs={"units": "m3 s-1"},
            )
        }
    ).to_netcdf(path)


def _write_obs(tmp_path: Path) -> Path:
    times = pd.date_range("1995-10-01", "2005-09-30", freq="D")
    path = tmp_path / "obs.csv"
    pd.DataFrame({"date": times, "discharge_cms": _hydrograph(times)}).to_csv(
        path, index=False
    )
    return path


def _write_historic_ensemble(run_dir: Path) -> None:
    """Five members, each a fixed multiple of the truth, in two segments."""
    first = pd.date_range(START, "2002-02-28", freq="D")
    second = pd.date_range("2002-03-01", END, freq="D")
    for member, scale in zip(MEMBERS, SCALES):
        for times, stamp in ((first, "2002_03_01"), (second, "2004_10_01")):
            _write_mizuroute(
                run_dir / "mizuroute" / f"{member}_{stamp}_mizuroute.nc",
                times,
                scale * _hydrograph(times),
            )


def _write_assimilation(run_dir: Path, *, resampled: bool = False) -> list[pd.Timestamp]:
    """One update each 1 March that puts nearly all weight on the true member."""
    folder = run_dir / "assimilation"
    folder.mkdir(parents=True, exist_ok=True)
    dates = [pd.Timestamp(f"{year}-03-01") for year in (2001, 2002, 2003, 2004)]
    rows = []
    for when in dates:
        stamp = when.strftime("%Y_%m_%d")
        pd.DataFrame([[0.01, 0.02, 0.94, 0.02, 0.01]], index=[when], columns=MEMBERS).to_csv(
            folder / f"{stamp}_swe_weights.csv"
        )
        index = pd.MultiIndex.from_tuples([(1, when)], names=["hru", "time"])
        pd.DataFrame([100.0 * SCALES], index=index, columns=MEMBERS).to_csv(
            folder / f"{stamp}_swe_particles.csv"
        )
        rows.append(
            {
                "date": when.isoformat(),
                "lumped_swe": 100.0,
                "n_samples": 1,
                "n_eff": 1.13,
                "tempering_beta": 1.0,
                "season_reset": when.year > 2001,
                "resampled": resampled,
            }
        )
    pd.DataFrame(rows).to_csv(folder / "updates.csv", index=False)
    return dates


def _write_esp(run_dir: Path, issues: list[pd.Timestamp]) -> None:
    for issue in issues:
        times = pd.date_range(issue, f"{issue.year}-07-31", freq="D")
        for i, scale in enumerate((0.9, 1.0, 1.1)):
            _write_mizuroute(
                run_dir
                / "esp"
                / f"m{i + 1:03d}"
                / issue.strftime("%Y%m%d%H")
                / str(1990 + i)
                / "mizuroute"
                / f"mizuroute_{1990 + i}.nc",
                times,
                scale * _hydrograph(times),
            )


def _config(tmp_path: Path, obs_csv: Path, **extra):
    evaluation = {"obs_csv": str(obs_csv), "figures_swe": False}
    evaluation.update(extra.pop("evaluation", {}))
    return load_pysummada_config(
        _build_config(
            tmp_path,
            extra={"num_ens": len(MEMBERS), "evaluation": evaluation, **extra},
        )
    )


def _provenance(run_dir: Path, run_mode: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps({"run_mode": run_mode}))


def _season(table: pd.DataFrame, estimate: str) -> pd.DataFrame:
    rows = table[(table["window"] == "season") & (table["estimate"] == estimate)]
    return rows.set_index("water_year")


def test_deterministic_run_scores_perfectly_and_drops_partial_years(tmp_path: Path):
    run_dir = tmp_path / "run"
    times = pd.date_range("2001-02-01", END, freq="D")
    _write_mizuroute(run_dir / "mizuroute" / "mizuroute_deterministic.nc", times, _hydrograph(times))
    _provenance(run_dir, "deterministic")
    cfg = _config(tmp_path, _write_obs(tmp_path))

    record = evaluate_run(run_dir, cfg)

    assert record["run_mode"] == "deterministic"
    assert record["primary_estimate"] == "deterministic"
    table = pd.read_csv(run_dir / "evaluation" / "historic" / "scores_by_water_year.csv")
    assert set(table["estimate"]) == {"deterministic"}
    full = table[table["window"] == "water_year"]
    # Water year 2001 starts in February here, so only its season is complete.
    assert sorted(full["water_year"]) == [2002, 2003, 2004]
    assert sorted(_season(table, "deterministic").index) == [2001, 2002, 2003, 2004]
    np.testing.assert_allclose(table["kge"], 1.0)
    np.testing.assert_allclose(table["crps"], 0.0, atol=1e-9)
    np.testing.assert_allclose(table["crpss"], 1.0)
    np.testing.assert_allclose(table["peak_timing_error_days"], 0.0)
    assert table["spread_skill"].isna().all()
    summary = json.loads((run_dir / "evaluation" / "historic" / "summary.json").read_text())
    pooled = summary["estimates"]["deterministic"]["season"]
    assert pooled["n_water_years"] == 4
    assert pooled["nse"] == pytest.approx(1.0)
    assert "pit_histogram" not in pooled
    assert (run_dir / "evaluation" / "historic" / "figures" / "wy_2003.png").is_file()


def test_ensemble_crps_equals_mae_scale_and_is_unchanged_by_weights(tmp_path: Path):
    obs_csv = _write_obs(tmp_path)
    open_loop = tmp_path / "open_loop"
    _write_historic_ensemble(open_loop)
    _provenance(open_loop, "historic_ensemble")
    sis = tmp_path / "sis"
    _write_historic_ensemble(sis)
    _write_assimilation(sis)
    _provenance(sis, "historic_sis")
    cfg = _config(tmp_path, obs_csv, season_start_month=10, evaluation={"figures": False})

    evaluate_run(open_loop, cfg)
    record = evaluate_run(sis, cfg)

    plain = pd.read_csv(open_loop / "evaluation" / "historic" / "scores_by_water_year.csv")
    weighted = pd.read_csv(sis / "evaluation" / "historic" / "scores_by_water_year.csv")
    assert set(plain["estimate"]) == {"ensemble"}
    assert set(weighted["estimate"]) == {"ensemble", "weighted"}
    assert record["primary_estimate"] == "weighted"
    # SIS leaves the members alone: its equally weighted estimate is the open loop.
    pd.testing.assert_frame_equal(
        plain.reset_index(drop=True),
        weighted[weighted["estimate"] == "ensemble"].reset_index(drop=True),
    )
    ensemble = _season(weighted, "ensemble")
    assimilated = _season(weighted, "weighted")
    assert (assimilated["crps"] < 0.5 * ensemble["crps"]).all()
    assert (assimilated["crpss"] > ensemble["crpss"]).all()
    assert (assimilated["coverage_80"] >= 0.0).all()
    # The mean of these members is the truth, so only the spread differs.
    np.testing.assert_allclose(ensemble["kge"], 1.0, atol=1e-6)
    assert (ensemble["spread_skill"].isna() | (ensemble["spread_skill"] > 1)).all()


def test_weighted_series_use_equal_weights_before_the_first_update(tmp_path: Path):
    run_dir = tmp_path / "sis"
    _write_historic_ensemble(run_dir)
    dates = _write_assimilation(run_dir)
    _provenance(run_dir, "historic_sis")
    cfg = _config(tmp_path, _write_obs(tmp_path), season_start_month=10, evaluation={"figures": False})

    evaluate_run(run_dir, cfg)

    series = xr.open_dataset(run_dir / "evaluation" / "historic" / "series.nc")
    winter = series.sel(time=slice("2002-10-01", "2003-02-28"))
    np.testing.assert_allclose(winter["weights"].values, 0.2)
    np.testing.assert_allclose(winter["weighted_mean"].values, winter["ensemble_mean"].values)
    melt = series.sel(time="2003-06-15")
    assert float(melt["weights"].sel(member="003")) == pytest.approx(0.94)
    assert float(melt["n_eff"]) < 1.2
    assert list(series["quantile"].values) == list(QUANTILES)
    updates = pd.read_csv(run_dir / "evaluation" / "da" / "updates.csv", parse_dates=["date"])
    assert list(updates["date"]) == dates
    np.testing.assert_allclose(updates["prior_mean"], 100.0)
    np.testing.assert_allclose(updates["innovation"], 0.0, atol=1e-9)
    np.testing.assert_allclose(updates["max_weight"], 0.94)


def test_sir_resampled_updates_are_not_weighted_twice(tmp_path: Path):
    run_dir = tmp_path / "sir"
    _write_historic_ensemble(run_dir)
    _write_assimilation(run_dir, resampled=True)
    _provenance(run_dir, "historic_sir")
    cfg = _config(tmp_path, _write_obs(tmp_path), season_start_month=10, evaluation={"figures": False})

    evaluate_run(run_dir, cfg)

    table = pd.read_csv(run_dir / "evaluation" / "historic" / "scores_by_water_year.csv")
    np.testing.assert_allclose(
        _season(table, "weighted")["crps"], _season(table, "ensemble")["crps"]
    )


def test_esp_run_writes_historic_and_forecast_scores(tmp_path: Path):
    run_dir = tmp_path / "esp_run"
    _write_historic_ensemble(run_dir)
    issues = _write_assimilation(run_dir)
    _write_esp(run_dir, issues)
    _provenance(run_dir, "esp_sis")
    cfg = _config(
        tmp_path,
        _write_obs(tmp_path),
        season_start_month=10,
        evaluation={"forecast": {"bootstrap_iterations": 0, "return_periods": [2, 5]}},
    )

    record = evaluate_run(run_dir, cfg)

    assert record["run_mode"] == "esp_sis"
    assert record["n_forecasts"] == 4
    out = run_dir / "evaluation"
    by_issue = pd.read_csv(out / "forecast" / "scores_by_issue.csv", parse_dates=["init_time"])
    assert list(by_issue["init_time"]) == issues
    assert list(by_issue["water_year"]) == [2001, 2002, 2003, 2004]
    assert set(by_issue["init_month"]) == {"03-01"}
    assert (by_issue["n_members"] == 3).all()
    # The middle forecast member is the truth.
    np.testing.assert_allclose(by_issue["volume_error_pct"], 0.0, atol=1e-6)
    assert (by_issue["volume_crpss"] > 0).all()
    summary = json.loads((out / "forecast" / "summary.json").read_text())
    month = summary["by_issue_month"]["03-01"]
    assert month["n_forecasts"] == 4
    assert month["volume_correlation"] == pytest.approx(1.0)
    assert "empirical" in summary["thresholds_source"]
    for name in (
        "evaluation.json",
        "historic/scores_by_water_year.csv",
        "forecast/metrics.nc",
        "da/updates.csv",
        "historic/figures/wy_2002.png",
        "historic/figures/scores_by_water_year.png",
        "historic/figures/pit_histogram.png",
        "forecast/figures/volume_scores.png",
    ):
        assert (out / name).is_file(), name
    assert set(record["outputs"]) >= {"historic/figures/wy_2002.png", "da/updates.csv"}


def test_forecast_mode_without_traces_is_an_error(tmp_path: Path):
    run_dir = tmp_path / "esp_run"
    _write_historic_ensemble(run_dir)
    _provenance(run_dir, "esp_open_loop")
    cfg = _config(tmp_path, _write_obs(tmp_path), evaluation={"figures": False})
    with pytest.raises(FileNotFoundError, match="no ESP mizuRoute traces"):
        evaluate_run(run_dir, cfg)


def test_safe_evaluation_keeps_the_run_and_records_the_failure(tmp_path: Path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    obs_csv = _write_obs(tmp_path)
    cfg = _config(tmp_path, obs_csv)
    logger = logging.getLogger("test_evaluation")
    assert evaluate_run_safely(run_dir, cfg, run_mode="historic_ensemble", run_log=logger) is None
    failed = run_dir / "evaluation" / "FAILED.txt"
    assert "FileNotFoundError" in failed.read_text()

    _write_historic_ensemble(run_dir)
    record = evaluate_run_safely(run_dir, cfg, run_mode="historic_ensemble", run_log=logger)
    assert record is not None and record["n_water_years"] == 4
    assert not failed.exists()

    no_evaluation = load_pysummada_config(_build_config(tmp_path))
    assert evaluate_run_safely(run_dir, no_evaluation, run_mode=None, run_log=logger) is None


def test_evaluate_historic_rejects_weights_that_miss_a_member():
    times = pd.date_range(START, END, freq="D")
    flow = pd.DataFrame({"a": _hydrograph(times), "b": 2 * _hydrograph(times)}, index=times)
    obs = pd.Series(_hydrograph(times), index=times)
    cfg_like = load_evaluation_defaults()
    weights = pd.DataFrame({"a": 1.0}, index=times)
    with pytest.raises(ValueError, match="do not cover every member"):
        evaluate_historic(flow, obs, cfg_like, weights=weights)


def load_evaluation_defaults():
    from snowda.config.pysummada import StreamflowEvaluationConfig

    return StreamflowEvaluationConfig(obs_csv=Path("/unused.csv"))


# ---- run modes and dispatch -------------------------------------------------


def test_run_mode_must_match_the_workflow():
    assert check_run_mode(None, "esp_sis") == "esp_sis"
    assert check_run_mode("esp_sis", esp_da_mode("sis")) == "esp_sis"
    with pytest.raises(ValueError, match="run_mode 'esp_sir'"):
        check_run_mode("esp_sir", esp_da_mode("sis"))
    assert historic_da_mode("sir") == "historic_sir"


def test_run_mode_is_recovered_from_older_provenance():
    assert run_mode_from_provenance({"run_mode": "esp_sir"}) == "esp_sir"
    assert run_mode_from_provenance({"variant": "historic"}) == "historic_ensemble"
    assert run_mode_from_provenance({"variant": "esp"}) == "esp_open_loop"
    assert run_mode_from_provenance({"variant": "da", "pf_scheme": "sir"}) == "esp_sir"
    assert (
        run_mode_from_provenance({"variant": "historic_da", "pf_scheme": "sis"})
        == "historic_sis"
    )
    assert run_mode_from_provenance({}) is None


def test_case_config_sets_the_filter_scheme_from_the_run_mode(tmp_path: Path):
    cfg = resolve_case_config(_build_config(tmp_path, extra={"run_mode": "esp_sir"}))
    assert cfg["pf_scheme"] == "sir"
    cfg = resolve_case_config(_build_config(tmp_path, extra={"run_mode": "historic_ensemble"}))
    assert "pf_scheme" not in cfg
    with pytest.raises(ValueError, match="uses pf_scheme 'sis'"):
        resolve_case_config(
            _build_config(tmp_path, extra={"run_mode": "esp_sis", "pf_scheme": "sir"})
        )
    with pytest.raises(ValueError, match="must define `run_mode`"):
        resolve_case_config(_build_config(tmp_path))


@pytest.mark.parametrize(
    ("run_mode", "module", "function"),
    [
        ("deterministic", "snowda.workflows.run_summa", "run_summa"),
        ("historic_ensemble", "snowda.workflows.run_summa_historic", "run_summa_historic"),
        ("historic_sis", "snowda.workflows.run_summa_historic_da", "run_summa_historic_da"),
        ("esp_open_loop", "snowda.workflows.run_summa_esp", "run_summa_esp"),
        ("esp_sis", "snowda.workflows.run_summa_da", "run_summa_da"),
        ("esp_sir", "snowda.workflows.run_summa_da", "run_summa_da"),
    ],
)
def test_run_case_dispatches_on_run_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_mode: str, module: str, function: str
):
    seen: dict = {}

    def fake(config):
        seen.update(config)
        return tmp_path / "run"

    monkeypatch.setattr(f"{module}.{function}", fake)
    result = run_case(_build_config(tmp_path, extra={"run_mode": run_mode}))
    assert result == tmp_path / "run"
    assert seen["run_mode"] == run_mode
    assert run_case_module._workflow(run_mode) is fake
