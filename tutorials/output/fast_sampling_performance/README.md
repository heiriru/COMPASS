# Fast sampling performance outputs

`raw_results.csv` contains one row per training scheme, sampler variant, model seed, observation, step budget, and sampling seed. `summary.csv` reports the median symmetrised Gaussian KL and 10--90% bands across those rows.

The legacy time-space DPM and drift-only Euler settings are tutorial-only reference implementations. They permit one-change-at-a-time comparisons against the current sigma-space DPM and Euler--Maruyama implementations.
