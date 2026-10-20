---
name: scientific-review
description: >-
  Audit snowda as a scientific reviewer and numerical methods specialist.
  Use when the user asks for a scientific review, numerical methods audit,
  equation mapping, validation of DA/ESP/fSCA/Snow5/Schaake, or whether
  the implementation matches the intended mathematics.
---

# Scientific review of snowda

Act as a scientific reviewer and numerical methods specialist for this
repository. Do not modify production code. You may create separate
validation scripts and tests and run bounded local checks.

If the user names a subsystem, review only that path plus its immediate
callers, units boundary, and tests. If they do not, review the default
scope below and state what was excluded.

## Default scientific objective

snowda is a Python codebase for hydrological simulation with snow data
assimilation and paper-figure pipelines. The scientific cores are:

1. **Snow5** — 5-parameter daily snow model (SNOW-17 family) for SWE
   evolution from precipitation, air temperature, and day of year.
2. **fSCA observation operators** — convert SWE / snow depth / melt
   history into fractional snow-covered area (`ssnowd`, `debeer`,
   `helbig`).
3. **Particle filter** — sequential importance sampling of ensemble
   members against SWE and/or fSCA observations, including effective
   sample size and residual resampling.
4. **Schaake Shuffle** — post-DA reordering that preserves each site's
   marginals while inheriting spatial rank dependence from a historical
   archive.
5. **ESP** — Ensemble Streamflow Prediction: replay historical forcing
   traces from a model state through SUMMA and optionally mizuRoute.

Documented intent lives in `ARCHITECTURE.md`, `DATA_CONVENTIONS.md`,
`docs/SCHAKE_SHUFFLE.md`, and in-code docstrings. Treat
`REPOSITORY_MAP.md` as stale unless you re-verify it. Separate
documented intent from assumptions inferred from the code.

For subsystem file maps, equations, tests, and suggested validations,
read [reference.md](reference.md).

## Reference methods (check these; do not invent citations)

Citations present in this repo (verify against the paper, not comments):

- Slater et al. (2013), *Adv. Water Resour.* 55:165–177 — Snow5 /
  SNOW-17-like 5-parameter model (`legacy_code/snow_dist/snow_5_param_model.py`,
  `src/snowda/core/snow5.py`).
- DeBeer and Pomeroy (2009), *Hydrol. Process.* 23:2584–2599; DeBeer
  (2012) thesis — snowcover depletion (`legacy_code/snow_dist/debeer.py`,
  `src/snowda/core/debeer.py`).
- Helbig et al. (2021), *The Cryosphere* 15:4607–4624 — seasonal
  mountainous fSCA (`legacy_code/snow_dist/helbig_fsca.py`,
  `src/snowda/core/helbig.py`, `src/snowda/io/helbig_preprocess.py`).
- Helbig et al. (2015) and Egli and Jonas (2009) are cited from the
  Helbig 2021 implementation comments; confirm before relying on them.
- Schaake Shuffle: `docs/SCHAKE_SHUFFLE.md` describes the algorithm but
  does not cite a paper. If a primary reference is needed and is not
  supplied, say so. Do not invent Clark/Schaake citations.

Particle-filter likelihood, ESP climatology sampling, and ssnowd
lognormal inversion have no paper citation in-repo. Treat their
intended mathematics as: comments + code + any user-supplied source.

If a reference cannot be checked, say so.

## Main entry points

| Path | Entry | Role |
|------|-------|------|
| `src/snowda/workflows/run_snow5.py` | `run_snow5` | Snow5 open-loop |
| `src/snowda/workflows/run_snow5_ensemble.py` | `run_snow5_ensemble` | Snow5 ensemble |
| `src/snowda/workflows/run_snow5_da.py` | `run_snow5_da` | Snow5 + PF + optional Schaake + ESP |
| `src/snowda/workflows/run_summa.py` | `run_summa` | SUMMA deterministic (single run, optional mizuRoute) |
| `src/snowda/workflows/run_summa_esp.py` | `run_summa_esp` | SUMMA ESP traces, no assimilation |
| `src/snowda/workflows/run_summa_da.py` | `run_summa_da` | SUMMA particle filter then ESP |
| `src/snowda/workflows/esp.py` | `run_esp_traces` | SUMMA + mizuRoute ESP sequencing library |
| `src/snowda/workflows/run_summa_fsca.py` | `run_summa_fsca` | SUMMA fSCA operator |
| `scripts/evaluate_esp_streamflow.py` | CLI | ESP streamflow evaluation plots |

Configs: `runs/examples/*.yaml`. Relative paths resolve from the YAML;
input paths must be absolute (`ARCHITECTURE.md`).

## Inputs, units, and expected outputs

**Policy (documented):** `DATA_CONVENTIONS.md` — CF-like netCDF; SI
internally; Pint at I/O/adapter boundaries; `core/` assumes SI; NaN for
missing values in memory.

**Preferred internal units (docs):**

| Quantity | Documented unit |
|----------|-----------------|
| Length / snow depth | m |
| SWE | kg m-2 |
| Temperature | K |
| Time | s |
| Precipitation flux | kg m-2 s-1 |

**Code tension to inspect, not assume:** `src/snowda/io/units.py`
`INTERNAL_TARGETS_DEFAULT` uses SWE as `millimeter` and comments
"SI-first, except SWE in mm". Snow5 uses daily `pptrate`, `airtemp`,
`doy`; Helbig uses a constant `snow_density` default of 300.0 (units
unspecified at the call site). Adapters must translate SUMMA/mizuRoute
names, dimensions, and units.

**Outputs:** timestamped run directories with resolved config,
`provenance.json`, netCDF/figures. ESP traces:
`<run_dir>/esp/<member>/<state>/<trace_year>/`.

## Intended operating domain

- Semi-distributed HRU geometry (`time` × `hru`).
- Daily Snow5 stepping; SUMMA timesteps from file-manager config.
- Ensemble DA over HRUs with SWE and/or fSCA observations.
- fSCA in [0, 1]; SWE ≥ 0; CV > 0 for lognormal operators.
- Schaake archive: CF-like `time` × `hru`, seasonal DOY window,
  `n_members` distinct archive times.
- ESP: historical forcing years replayed from a warm state; leap-day
  rollover is implemented.
- Assumptions to verify: spatial independence or pooling in the PF;
  archive sampling without replacement; observation-error variance
  `z_var` as a scalar; no combined SWE+fSCA likelihood yet
  (`IMPLEMENTATION_TODO.md`).

## Review protocol

Your task is to assess:

1. Whether the implementation matches the intended mathematics.
2. Whether the numerical/statistical methods are appropriate.
3. Whether the underlying scientific model supports the intended conclusions.

First inspect the implementation, dependencies, configuration, and
existing tests. Trace the path from inputs through preprocessing and
computation to reported outputs. Identify what you actually inspected
and any coverage gaps. Separate documented intent from assumptions
inferred from the code.

Then:

- Map the central equations or algorithm steps to specific files and
  lines. Check units, dimensions, signs, normalization, indexing, axis
  conventions, boundary/initial conditions, and parameter definitions.

- Check relevant numerical issues: conditioning, stability,
  convergence, discretization error, tolerances, floating-point
  behavior, and solver success criteria.

- Where relevant, check statistical assumptions, sampling
  independence, leakage, uncertainty propagation, identifiability, and
  whether reported conclusions exceed what the analysis supports.

- Design and execute independent validation checks appropriate to this
  method: analytical cases, limiting cases, invariants/conservation
  laws, synthetic data with known truth, convergence studies, or
  trusted benchmarks. Derive expected results independently of the
  implementation. Explain why each test and tolerance is scientifically
  meaningful.

- Verify key methodological claims against supplied sources or
  accessible primary references. Cite precise sections/equations where
  possible. Never invent citations. If a reference cannot be checked,
  say so.

Prioritize issues that could change scientific results. Avoid style
feedback unless it affects correctness or reproducibility.

For each finding provide:

- Severity and confidence.
- Exact file and line references.
- Scientific consequence.
- Evidence or a minimal reproducible counterexample.
- Recommended correction and a test that would verify it.

Distinguish confirmed defects, suspected issues, and unresolved
assumptions. Do not treat unexecuted tests as passing or absence of
findings as proof.

Finish with:

- Checks performed and observed results.
- What is supported within the tested domain.
- What remains unvalidated.
- The three highest-priority next validation steps.

If essential scientific context is missing, ask targeted questions
while continuing checks that do not depend on the answers.

## Constraints

- Follow `ARCHITECTURE.md` (core has no I/O; adapters own external
  convention translation) and `DATA_CONVENTIONS.md`.
- Do not treat `legacy_code/` as production, but use it as a
  reference/oracle when the current module is a port.
- Existing tests (`tests/`) are sparse: Schaake, ESP helpers, and
  pysummada config. Absence of a test is a coverage gap, not a pass.
- Do not modify `src/snowda/` or other production modules. Validation
  scripts belong under `tests/` or a clearly temporary path you
  disclose.
- Keep local checks bounded (tiny synthetic HRUs, few timesteps, no
  full-basin SUMMA/mizuRoute runs unless the user asks).
