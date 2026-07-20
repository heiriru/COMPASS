# Fast sampling performance: Banana

raw_results.csv contains one row per training scheme, sampler variant, model seed, observation, step budget, and sampling seed. summary.csv reports median direct errors in the generated posterior mean and covariance, together with 10--90% bands. The reference is exact rejection sampling from the Gaussian prior using likelihood-only acceptance. Banana rows additionally report sliced_wasserstein, which compares the full curved distribution through 32 one-dimensional projections.

sampler_pairplot.png uses the same Seaborn pairplot style as gaussian_hypotheses_pairplot.png: KDE diagonals, small translucent scatter points, and a categorical sampler legend.

shape_ablation.png plots that sliced-Wasserstein distance so curved-distribution quality is visible directly.

training_time_sampling.png compares uniform and mixture diffusion-time training for sigma-space DPM-1 and DPM-2 on the log-noise grid. The score diagnostic uses matched denoising-score targets drawn from the exact reference posterior.

