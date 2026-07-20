# Fast sampling performance: Gaussian

raw_results.csv contains one row per training scheme, sampler variant, model seed, observation, step budget, and sampling seed. summary.csv reports median direct errors in the generated posterior mean and covariance, together with 10--90% bands. The reference is the closed-form conditional Gaussian posterior.

sampler_pairplot.png uses the same Seaborn pairplot style as gaussian_hypotheses_pairplot.png: KDE diagonals, small translucent scatter points, and a categorical sampler legend.

training_time_sampling.png compares uniform and mixture diffusion-time training for sigma-space DPM-1 and DPM-2 on the log-noise grid. The score diagnostic uses the exact conditional score at every marginal noise scale.

