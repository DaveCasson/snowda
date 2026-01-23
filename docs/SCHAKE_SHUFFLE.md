# Schaake Shuffle

The Schaake Shuffle module reorders ensemble members per site so that the spatial rank dependence
is inherited from a historical reference record while keeping each site’s marginal values untouched.
It runs as an optional post-processing step after the particle-filter update and any resampling to
preserve the ensemble’s SI units and CF metadata model outputs already carry.

## How it works

1. The sampler draws `n_members` samples from a historical archive that encodes
   realistic spatial dependence (the reference variable `Y`).
2. `Y` is ranked after a tiny deterministic jitter is added to break ties.
3. The current ensemble `X` for the configured target variable is sorted per site and reassigned
   according to the rankings of `Y`. This keeps each site’s marginal distribution unchanged.
4. Optionally, the same permutation can be applied to other state variables so each particle remains
   self-consistent.

## Archive requirements

- The archive must be a CF-like dataset with dimensions `time` × `hru` (or the equivalent site
  coordination that `Snow5` uses). The current implementation uses `snow_5` outputs and reads
  the variable specified by `schaake_archive_variable`.
- A 20‑year model simulation archive is recommended for realistic spatial dependence, but the
  logic is abstracted behind `Snow5ArchiveAdapter` so the source can be swapped later.
- Sampling supports memory-efficient reads via `xarray.open_dataset` (chunking), and candidate times
  are limited to a seasonal window around the current DOY to respect climatological cycles.

## Configuration

Add a `schaake_shuffle` section to your DA config:

```yaml
schaake_shuffle:
  use_schaake_shuffle: true
  schaake_window_days: 15
  schaake_seed: 42
  schaake_archive_source: /absolute/path/to/archive.nc
  schaake_archive_variable: swe
  schaake_target_variables: ["swe"]
  schaake_apply_same_permutation_to: ["snow_depth"]
```

- `use_schaake_shuffle`: enable/disable the post-processing step.
- `schaake_window_days`: half-width of the DOY window (seconds/rescribed) used to sample archive entries.
- `schaake_seed`: optional RNG seed to make sampling reproducible.
- `schaake_archive_source`: absolute path to the historical archive (must exist when enabled).
- `schaake_archive_variable`: name of the variable inside the archive (`swe` by default).
- `schaake_target_variables`: list of variables whose marginals are reordered; the first entry is
  used to draw the Schaake permutation from `Y`.
- `schaake_apply_same_permutation_to`: optional variables (e.g., additional state arrays) that should
  receive the same permutation without resorting.

The shuffle is applied after the particle update/resample and before the next forecasting step. When
new variables or archives are added, keep uneven data conversions at the I/O boundary and rely on the
new `schaake_shuffle` module so the core computation stays SI-compliant.
