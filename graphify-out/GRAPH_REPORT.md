# Graph Report - COMPASS  (2026-07-19)

## Corpus Check
- 43 files · ~2,213,463 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 843 nodes · 1864 edges · 47 communities (36 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 91 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- fast_sampling_performance.py
- MultiObsSampler
- Divergence_Head.py
- Compositional_Inference.py
- ModelTransfuser
- Gaussians_VP_test.py
- VESDE
- .__init__
- Sampler
- Trainer
- VPSDE
- PFODE
- Population_Dynamics.py
- Probability_Flow_validation.py
- test_instantaneous_divergence.py
- collect_xspace_likelihood_runtime_benchmark
- plot_pfode_hutchinson_2d_likelihood_diagnostic
- ScoreBasedInferenceModel
- log
- ConditionTransformer
- CHANGELOG
- condition_mask_for_model
- README.md
- to_numpy
- plot_pfode_hutchinson_2d_likelihood_diagnostic_from_archive
- collect_map_runtime_benchmark
- plot_xspace_likelihood_map_accuracy_runtime_benchmark
- .__init__
- .time_of_sigma
- README.md
- README.md
- README.md
- README.md
- bayes-compass
- analytic_posterior
- banana_reference_samples
- sampler_grid_mode
- MultiObsSampler
- ._get_score
- .sample
- TensorTupleDataset
- ._validate_precision
- .__init__
- .time_of_sigma
- README.md
- README.md

## God Nodes (most connected - your core abstractions)
1. `MultiObsSampler` - 40 edges
2. `PFODE` - 29 edges
3. `Sampler` - 27 edges
4. `VESDE` - 26 edges
5. `ScoreBasedInferenceModel` - 25 edges
6. `VPSDE` - 24 edges
7. `main()` - 23 edges
8. `run_banana_benchmark()` - 21 edges
9. `ModelTransfuser` - 20 edges
10. `log()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `MockSBIm` --uses--> `ConditionTransformer`  [INFERRED]
  tests/test_instantaneous_divergence.py → src/compass/ConditionTransformer.py
- `test_transformer_legacy_and_optional_return_contracts()` --calls--> `ConditionTransformer`  [INFERRED]
  tests/test_instantaneous_divergence.py → src/compass/ConditionTransformer.py
- `LinearGaussianModel` --uses--> `MultiObsSampler`  [INFERRED]
  tests/test_multiobs_analytic.py → src/compass/MultiObsSampler.py
- `MockSBIm` --uses--> `MultiObsSampler`  [INFERRED]
  tests/test_multiobs_analytic.py → src/compass/MultiObsSampler.py
- `_sample()` --calls--> `MultiObsSampler`  [INFERRED]
  tests/test_multiobs_analytic.py → src/compass/MultiObsSampler.py

## Import Cycles
- None detected.

## Communities (47 total, 11 thin omitted)

### Community 0 - "fast_sampling_performance.py"
Cohesion: 0.14
Nodes (20): banana_sample_once(), BananaData, evaluate_banana_training_calibration(), evaluate_banana_training_score_accuracy(), evaluate_training_calibration(), SBIm, Raw and normalized splits for the Banana_posterior.ipynb problem., Load or train one notebook-matched banana posterior model. (+12 more)

### Community 1 - "MultiObsSampler"
Cohesion: 0.15
Nodes (17): analytic_posterior(), LinearGaussianModel, MockSBIm, Analytic verification of the compositional (multi-observation) score modeling., Paper Sec. 5.1: p(theta)=N(0,I), p(x|theta)=0.5 N(theta, I/2)+0.5 N(-theta, I/2), Chempy-like structure: shared global theta_g, per-observation local theta_l,, Minimal stand-in for ScoreBasedInferenceModel., Nodes [th1, th2, x1, x2]; returns std_t * true diffused posterior score     (mat (+9 more)

### Community 2 - "Divergence_Head.py"
Cohesion: 0.07
Nodes (59): aggregate_raw(), aggregate_score(), analytic_epsilon_sigma(), benchmark_runtime(), build_summary(), cache_is_complete(), cache_paths(), configure_cpu_usage_limit() (+51 more)

### Community 3 - "Compositional_Inference.py"
Cohesion: 0.09
Nodes (73): Figure, Namespace, aggregate(), analytic_shared(), build_config(), build_model(), configure_cpu_usage_limit(), configure_plot_style() (+65 more)

### Community 4 - "ModelTransfuser"
Cohesion: 0.05
Nodes (30): ModelTransfuser, Remove a model from the transfuser.          Args:             model_name: The n, Initialize the Score-Based Inference Models with the given parameters          A, Train the models on the provided data          Args:             batch_size: Bat, Compare the models on the provided observations.         The results are saved i, Compute the log probability of the samples, Find the joint mode of the multivariate distribution, Per-observation information criterion, dispatching on the criterion         sele (+22 more)

### Community 5 - "Gaussians_VP_test.py"
Cohesion: 0.11
Nodes (41): circle_mean(), classifier_probs(), collect_baseline_results(), compass_per_observation_ic_probs(), compass_probs(), compute_true_marginal_probs(), cumulative_compass_ic_curve(), cumulative_true_model_curve() (+33 more)

### Community 6 - "VESDE"
Cohesion: 0.12
Nodes (28): Mean scaling of the perturbation kernel; identically 1 for the VESDE., Compute the standard deviation of p_{0t}(x(t) | x(0)) for VESDE.          Args:, VESDE, analytic_logpdf(), DiffusedPosteriorScore, _eval_points(), MockSBIm, _pfode_logprob() (+20 more)

### Community 7 - ".__init__"
Cohesion: 0.08
Nodes (16): DivergenceHead, FinalLayer, InputEmbedder, Mlp, modulate(), MLP for Output of Self-Attention, A ConditionTransformer block with adaptive layer norm zero (adaLN-Zero) conditio, The final layer of ConditionTransformer. (+8 more)

### Community 8 - "Sampler"
Cohesion: 0.11
Nodes (11): Dataset, Get score estimate with optional classifier-free guidance.          The samplers, Basic Euler-Maruyama sampling method                  Args:             data: In, Corrector steps using Langevin dynamics                  Args:             x: In, First-order solver (in noise-scale space).          The probability-flow ODE dx, Second-order solver (in noise-scale space), Third-order solver (in noise-scale space), Sample from the model using the specified method          Args:             data (+3 more)

### Community 9 - "Trainer"
Cohesion: 0.11
Nodes (9): Dataset, Return D_lambda = alpha * trace(d raw_score / dx) on latent nodes.          The, Dimension-normalized MSE for scalar latent divergence targets., Draw diffusion times according to the configured time_sampling scheme., Loss function for the score prediction task          Args:             score: Pr, Get score estimate from model, Training function for the score prediction task          Args:             rank:, TensorTupleDataset (+1 more)

### Community 10 - "VPSDE"
Cohesion: 0.11
Nodes (10): Mean scaling alpha(t) = exp(-B(t)/2)., Noise std sigma(t) = sqrt(1 - exp(-B(t))) of p_{0t}(x(t)|x(0))., Solve B(t) = B for t (quadratic in t, positive root)., Inverse of marginal_prob_std (sigma must be < 1)., Noise-to-signal scale lambda(t) = sigma(t)/alpha(t) = sqrt(exp(B(t)) - 1)., Compute sigma_t (noise standard deviation)., Noise-to-signal scale lambda(t) = sigma(t)/alpha(t); equals sigma(t) for the VES, Inverse of lambda_t; equals time_of_sigma for the VESDE. (+2 more)

### Community 11 - "PFODE"
Cohesion: 0.15
Nodes (8): PFODE, Log-probability of the latent dimensions of `data` given its conditioned, Return score s_y and instantaneous log-density drift D_lambda.          The netw, KDE-free MAP estimate of the latent dimensions by deterministic annealed, Return the common log-lambda integration grid and matching diffusion times., Take one Euler or Heun step for dy/dlambda = -lambda * score_y.          The eva, Sample by integrating the probability-flow ODE from noise to data.          y mu, Exact(-in-the-limit) log-probability evaluation through the probability-flow ODE

### Community 12 - "Population_Dynamics.py"
Cohesion: 0.13
Nodes (18): load_models(), logistic_prey(), lotka_volterra(), main(), observations_for(), plot_comparisons_by_true_model(), Compare population-dynamics hypotheses for each possible true model.  This is th, Logistic prey growth with predator satiation. (+10 more)

### Community 13 - "Probability_Flow_validation.py"
Cohesion: 0.19
Nodes (18): add_cov_ellipse(), compass_annealed_map_theta(), configure_cpu_usage_limit(), draw_likelihood_ellipse_grid(), evaluate_kde_density_grid(), manifold_norm(), manifold_raw(), pfode_likelihood_log_prob() (+10 more)

### Community 14 - "test_instantaneous_divergence.py"
Cohesion: 0.22
Nodes (12): configure_cpu_usage_limit(), _evaluation_data(), MockSBIm, Focused tests for the optional instantaneous divergence head., Hard-limit this process and its children to host CPU capacity., _small_model(), test_instantaneous_target_exact_hutchinson_and_masks(), test_learned_mode_requires_trained_head() (+4 more)

### Community 15 - "collect_xspace_likelihood_runtime_benchmark"
Cohesion: 0.19
Nodes (15): analytic_likelihood_map(), collect_xspace_likelihood_runtime_benchmark(), generate_model_data(), load_or_create_xspace_mc_reference(), model_mean(), normalize(), normalize_x(), progress() (+7 more)

### Community 16 - "plot_pfode_hutchinson_2d_likelihood_diagnostic"
Cohesion: 0.15
Nodes (14): benchmark_map_estimates(), benchmark_xspace_likelihood_map(), evaluate_pfode_likelihood_grid(), evaluate_pfode_likelihood_log_prob_grid(), likelihood_map_from_samples(), normalize_log_density_grid(), plot_pfode_hutchinson_2d_likelihood_diagnostic(), Return one MAP per observation and the elapsed inference time. (+6 more)

### Community 17 - "ScoreBasedInferenceModel"
Cohesion: 0.15
Nodes (6): Train the model on the provided data          Args:             theta: Training, Sample from the model using the specified method          Args:             data, Evaluate the log-probability of the latent dimensions of `data` given its, KDE-free MAP estimate of the latent dimensions of `data` given its         condi, ScoreBasedInferenceModel, test_pfode_public_sampling_paths_and_batched_masks()

### Community 18 - "log"
Cohesion: 0.24
Nodes (13): checkpoint_paths(), load_or_train_models(), log(), main(), make_pairplot(), plot_likelihood_reconstruction_distance(), plot_posterior_pfode_hutchinson_trace_samples(), promote_best_checkpoint() (+5 more)

### Community 19 - "ConditionTransformer"
Cohesion: 0.18
Nodes (6): ConditionTransformer, Diffusion model with a Transformer backbone., Forward pass of ConditionTransformer.         Args:             x:   (N, C, H, W, GaussianScoreWithDrift, Independent Gaussian score with an analytic learned drift head., test_transformer_legacy_and_optional_return_contracts()

### Community 20 - "CHANGELOG"
Cohesion: 0.20
Nodes (9): Bug Fixes, Bug Fixes, Bug Fixes, CHANGELOG, Chores, v0.1.5 (2025-05-02), v1.0.0 (2025-08-14), v1.0.1 (2025-08-14) (+1 more)

### Community 21 - "condition_mask_for_model"
Cohesion: 0.33
Nodes (9): calculate_posterior_samples_one_observation(), collect_likelihood_ellipse_cases(), collect_matched_likelihood_reconstruction_summaries(), compass_posterior_log_prob_grid(), condition_mask_for_model(), estimate_map_from_posterior_samples(), make_ellipse_observations(), plot_posterior_samples_one_observation() (+1 more)

### Community 22 - "README.md"
Cohesion: 0.25
Nodes (7): COMPASS: Comparison Of Models using Probabilistic Assessment in Simulation-based Settings, Contributing, Features, Installation, Model Comparison Example, Simulation-Based Inference Model, Usage

### Community 24 - "to_numpy"
Cohesion: 0.29
Nodes (8): compute_normalization(), data_path(), load_or_generate_data(), load_tensor_pair(), monte_carlo_likelihood_map(), Standalone KDE MAP used for the immutable Monte-Carlo reference., save_tensor_pair(), to_numpy()

### Community 25 - "plot_pfode_hutchinson_2d_likelihood_diagnostic_from_archive"
Cohesion: 0.25
Nodes (8): contour_boundary(), density_threshold_for_mass(), draw_density_contour(), plot_pfode_hutchinson_2d_likelihood_diagnostic_from_archive(), polygon_signed_area(), Return the largest 90%-mass contour as an open sequence of vertices., Return the signed area of an open polygon boundary., Overlay the repeated Hutchinson 90%-mass contours from a saved NPZ.

### Community 26 - "collect_map_runtime_benchmark"
Cohesion: 0.33
Nodes (7): collect_map_runtime_benchmark(), map_runtime_benchmark_configs(), plot_map_accuracy_runtime_benchmark(), Return the requested sampler and PF-ODE MAP benchmark sweep., Evaluate MAP offsets over all true-model and inference-model pairs., Plot mean MAP offset plus one standard deviation against mean runtime., save_csv()

### Community 27 - "plot_xspace_likelihood_map_accuracy_runtime_benchmark"
Cohesion: 0.33
Nodes (6): plot_xspace_likelihood_map_accuracy_runtime_benchmark(), plot_xspace_likelihood_map_accuracy_runtime_summary(), Plot separate matched-family x-space MAP accuracy/runtime summaries., Average matched-model x-space MAP errors over theta.      Only Line-generated da, Create separate matched-model x-space likelihood-MAP benchmark plots., summarise_xspace_likelihood_map_accuracy_runtime()

### Community 28 - ".__init__"
Cohesion: 0.15
Nodes (30): DataFrame, configure_plot_style(), finish_error_axes(), line_with_band(), plot_banana_map_pareto(), plot_banana_shape_ablation(), plot_common_outputs(), plot_dpm2_training_grid_pareto() (+22 more)

### Community 29 - ".time_of_sigma"
Cohesion: 0.14
Nodes (21): configure_cpu_usage_limit(), main(), make_banana_data(), plot_banana_posterior_examples(), Generate the nonlinear posterior problem from Banana_posterior.ipynb., Reserve one free GPU with autocvd, otherwise use CPU without waiting.      ``aut, Show the curved reference posterior beside principal sampler contrasts., Run the same suite on the nonlinear Banana_posterior.ipynb problem. (+13 more)

### Community 35 - "analytic_posterior"
Cohesion: 0.12
Nodes (20): Axes, add_covariance_ellipse(), analytic_posterior(), evaluate_training_score_accuracy(), plot_posterior_examples(), plot_sampler_pairplot(), Plot reference and sampler posteriors in the supplied Seaborn style., Show intuitive posterior samples for the current sampler variants. (+12 more)

### Community 36 - "banana_reference_samples"
Cohesion: 0.16
Nodes (17): banana_log_likelihood(), banana_observation_from_theta(), banana_reference_map(), banana_reference_samples(), posterior_errors(), Tensor, Unnormalised log likelihood; its maximum is zero., Return the noiseless banana simulator output for parameter values. (+9 more)

### Community 37 - "sampler_grid_mode"
Cohesion: 0.10
Nodes (32): banana_splits(), benchmark_summary(), density_grid(), evaluate_density(), evaluate_density_sweep(), evaluate_likelihood_convergence(), fixed_raw_joint(), joint_rows() (+24 more)

### Community 38 - "MultiObsSampler"
Cohesion: 0.15
Nodes (10): MultiObsSampler, Compositional score modeling for inference with multiple i.i.d. observations, Validate SPD full precisions and keep their composed precision SPD., Return (1-N) Lambda_prior + sum_j Lambda_j for full Gaussian fusion., Sample the final Gaussian conditional after stopping reverse diffusion., Noise that is shared across the observation axis on the hierarchy dimensions, Per-feature scale for initial noise / Langevin steps. For "fnpe" the bridging, Basic Euler-Maruyama sampling method          Args:             data: Input data (+2 more)

### Community 39 - "._get_score"
Cohesion: 0.18
Nodes (6): Get the composed score estimate with optional classifier-free guidance, Compose the per-observation scores on the hierarchy (shared parameter)         d, First-order solver (in noise-scale space).          The probability-flow ODE dx, Second-order solver (in noise-scale space), Third-order solver (in noise-scale space), Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinemen

### Community 40 - ".sample"
Cohesion: 0.25
Nodes (4): Resolve the Gaussian prior over the hierarchy dimensions to (mean, std) tensors., Estimate the precision of each single-observation posterior on the hierarchy, Moment-match each single-observation posterior with a full Gaussian., Sample from the multi-observation posterior via compositional score modeling.

## Knowledge Gaps
- **18 isolated node(s):** `bayes-compass`, `Bug Fixes`, `Bug Fixes`, `Bug Fixes`, `Chores` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ScoreBasedInferenceModel` connect `ScoreBasedInferenceModel` to `ModelTransfuser`, `MultiObsSampler`, `VESDE`, `Sampler`, `Trainer`, `VPSDE`, `PFODE`, `test_instantaneous_divergence.py`, `ConditionTransformer`, `ScoreBasedInferenceModel.py`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `MultiObsSampler` connect `MultiObsSampler` to `MultiObsSampler`, `._get_score`, `.sample`, `TensorTupleDataset`, `._validate_precision`, `PFODE`, `ScoreBasedInferenceModel`, `ConditionTransformer`, `ScoreBasedInferenceModel.py`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `ModelTransfuser` connect `ModelTransfuser` to `ScoreBasedInferenceModel`, `ScoreBasedInferenceModel.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `MultiObsSampler` (e.g. with `PFODE` and `ScoreBasedInferenceModel`) actually correct?**
  _`MultiObsSampler` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `PFODE` (e.g. with `MultiObsSampler` and `TensorTupleDataset`) actually correct?**
  _`PFODE` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Sampler` (e.g. with `PFODE` and `ScoreBasedInferenceModel`) actually correct?**
  _`Sampler` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `VESDE` (e.g. with `ScoreBasedInferenceModel` and `GaussianScoreWithDrift`) actually correct?**
  _`VESDE` has 16 INFERRED edges - model-reasoned connections that need verification._