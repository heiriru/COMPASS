# Fast sampling performance outputs

raw_results.csv contains one row per training scheme, sampler variant, model seed, observation, step budget, and sampling seed. summary.csv reports median direct errors in the generated posterior mean and covariance relative to the closed-form analytic posterior, together with 10--90% bands across those rows. No fitted-Gaussian KL is used.

training_time_sampling.png compares uniform and mixture diffusion-time training for sigma-space DPM-1 and DPM-2 on the log-noise grid. training_noise_diagnostic.png compares both models with the exact conditional score over noise scale and reports posterior width relative to the analytical target.

The legacy time-space DPM and drift-only Euler settings are tutorial-only reference implementations. They permit one-change-at-a-time comparisons against the current sigma-space DPM and Euler--Maruyama implementations.
