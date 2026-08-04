# Graph Report - COMPASS  (2026-08-04)

## Corpus Check
- 177 files · ~4,938,655 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2279 nodes · 5332 edges · 108 communities (88 shown, 20 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 309 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0bf52dba`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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
- ScoreBasedInferenceModel.py
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
- Hierarchical_Linear_Quadratic.py
- infer_partial_pooling.py
- compare_shared_local_composition_methods.py
- main
- data_pipeline.py
- Tensor
- compositional_score_analytic.py
- migrate_artifact_names.py
- joint_vs_legacy_annealed_score_ascent.py
- Compare_Upstream_PFODE.py
- Compositional_Inference.py
- Compositional_Inference.py
- RunConfig
- RunConfig
- BenchmarkPaths
- test_benchmark_contracts.py
- PF_ODE_1.py
- plot_partial_pooling_comparison.py
- Tensor
- Tensor
- experiment_parabola
- experiment_parabola
- figures
- MultiObsSampler
- figures
- figures
- figures
- test_full_gaussian_composition.py
- test_variable_param_count.py
- __init__.py
- tensor_numpy
- tensor_numpy
- damping_d1_method_sweeps.py
- figures
- manifest.json
- BenchmarkConfig
- ._compositional_score
- test_hierarchical_map.py
- test_memory_bounded_attention.py
- figures
- ._shared_then_local_map
- test_composition_strategies.py
- Tall-data Figures 1–4 in COMPASS
- test_damping_tutorial.py
- JointScoreCompareMock
- plot_global_observation_sensitivity.py
- TimestepEmbedder
- test_multistart_avoids_low_density_posterior_mean
- TensorTupleDataset
- Partial-pooling benchmark summary
- COMPASS global/local inference validation
- Hierarchical Partial-Pooling Benchmark
- TensorTupleDataset
- ExactGaussianScore
- __init__.py
- __init__.py
- _script_template.py
- select_device
- select_device
- uniform_time_grid
- README.md

## God Nodes (most connected - your core abstractions)
1. `MultiObsSampler` - 103 edges
2. `VESDE` - 64 edges
3. `Sampler` - 43 edges
4. `ModelTransfuser` - 40 edges
5. `VPSDE` - 39 edges
6. `run()` - 31 edges
7. `ScoreBasedInferenceModel` - 30 edges
8. `PFODE` - 25 edges
9. `BenchmarkPaths` - 24 edges
10. `RunConfig` - 24 edges

## Surprising Connections (you probably didn't know these)
- `test_validation_uses_configured_training_batch_size()` --calls--> `Trainer`  [INFERRED]
  tests/test_memory_bounded_attention.py → src/compass/Trainer.py
- `FrozenErrorNet` --uses--> `Sampler`  [INFERRED]
  Compositional_score_testing/Gauss_test/analytic.py → src/compass/Sampler.py
- `FrozenErrorNet` --uses--> `VPSDE`  [INFERRED]
  Compositional_score_testing/Gauss_test/analytic.py → src/compass/SDE.py
- `AnalyticPosteriorScore` --uses--> `Sampler`  [INFERRED]
  Compositional_score_testing/Gauss_test/analytic.py → src/compass/Sampler.py
- `AnalyticPosteriorScore` --uses--> `VPSDE`  [INFERRED]
  Compositional_score_testing/Gauss_test/analytic.py → src/compass/SDE.py

## Import Cycles
- None detected.

## Communities (108 total, 20 thin omitted)

### Community 0 - "fast_sampling_performance.py"
Cohesion: 0.10
Nodes (35): analytic_posterior(), banana_observation_from_theta(), banana_sample_once(), BananaData, evaluate_banana_training_calibration(), evaluate_banana_training_score_accuracy(), evaluate_training_calibration(), evaluate_training_score_accuracy() (+27 more)

### Community 1 - "MultiObsSampler"
Cohesion: 0.13
Nodes (20): analytic_posterior(), hierarchical_shared_and_local_metrics(), LinearGaussianModel, MockSBIm, Analytic verification of the compositional (multi-observation) score modeling., Paper Sec. 5.1: p(theta)=N(0,I), p(x|theta)=0.5 N(theta, I/2)+0.5 N(-theta, I/2), Run the exact-score global/local benchmark and return plot-ready metrics., Write standalone metrics while keeping pytest side-effect free. (+12 more)

### Community 2 - "Divergence_Head.py"
Cohesion: 0.07
Nodes (60): aggregate_raw(), aggregate_score(), analytic_epsilon_sigma(), benchmark_runtime(), build_summary(), cache_is_complete(), cache_paths(), configure_cpu_usage_limit() (+52 more)

### Community 3 - "Compositional_Inference.py"
Cohesion: 0.05
Nodes (66): AnalyticPosteriorScore, AnalyticSBIm, FrozenErrorNet, gaussian_log_prob(), GaussianMixtureToy, GaussianToy, device, Tensor (+58 more)

### Community 4 - "ModelTransfuser"
Cohesion: 0.10
Nodes (12): ModelTransfuser, Plot the results from the Model Comparison.         Saves the Violin plots for i, Remove a model from the transfuser.          Args:             model_name: The n, Plot the attention weights for the best performing model for interpretability., Initialize the Score-Based Inference Models with the given parameters          A, Train the models on the provided data          Args:             batch_size: Bat, Add a trained model to the transfuser.          Args:             model_name: Th, Add multiple trained models to the transfuser.          Args:             models (+4 more)

### Community 5 - "Gaussians_VP_test.py"
Cohesion: 0.11
Nodes (41): circle_mean(), classifier_probs(), collect_baseline_results(), compass_per_observation_ic_probs(), compass_probs(), compute_true_marginal_probs(), cumulative_compass_ic_curve(), cumulative_true_model_curve() (+33 more)

### Community 6 - "VESDE"
Cohesion: 0.13
Nodes (25): analytic_logpdf(), DiffusedPosteriorScore, _eval_points(), MockSBIm, _pfode_logprob(), posterior_moments(), Analytic verification of the probability-flow-ODE log-probability, the score-asc, Train a tiny VPSDE model on the linear-Gaussian joint and check the     sampled (+17 more)

### Community 7 - ".__init__"
Cohesion: 0.13
Nodes (10): FinalLayer, InputEmbedder, Mlp, modulate(), MLP for Output of Self-Attention, A ConditionTransformer block with adaptive layer norm zero (adaLN-Zero) conditio, The final layer of ConditionTransformer., Embeds joint data into vector representations. (+2 more)

### Community 8 - "Sampler"
Cohesion: 0.15
Nodes (9): Get score estimate with optional classifier-free guidance.          The samplers, Basic Euler-Maruyama sampling method                  Args:             data: In, Corrector steps using Langevin dynamics                  Args:             x: In, First-order solver (in noise-scale space).          The probability-flow ODE dx, Sample from the model using the specified method          Args:             data, Second-order solver (in noise-scale space), Third-order solver (in noise-scale space), Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinemen (+1 more)

### Community 9 - "Trainer"
Cohesion: 0.20
Nodes (5): Draw diffusion times according to the configured time_sampling scheme., Loss function for the score prediction task          Args:             score: Pr, Get score estimate from model, Training function for the score prediction task          Args:             rank:, Trainer

### Community 10 - "VPSDE"
Cohesion: 0.11
Nodes (10): Mean scaling alpha(t) = exp(-B(t)/2)., Noise std sigma(t) = sqrt(1 - exp(-B(t))) of p_{0t}(x(t)|x(0))., Solve B(t) = B for t (quadratic in t, positive root)., Inverse of marginal_prob_std (sigma must be < 1)., Noise-to-signal scale lambda(t) = sigma(t)/alpha(t) = sqrt(exp(B(t)) - 1)., Compute sigma_t (noise standard deviation)., Noise-to-signal scale lambda(t) = sigma(t)/alpha(t); equals sigma(t) for the VES, Inverse of lambda_t; equals time_of_sigma for the VESDE. (+2 more)

### Community 11 - "PFODE"
Cohesion: 0.16
Nodes (7): PFODE, Score s_y = alpha * s_x and its divergence w.r.t. y over the latent dims., KDE-free MAP estimate of the latent dimensions by deterministic annealed, Log-probability of the latent dimensions of `data` given its conditioned, Exact(-in-the-limit) log-probability evaluation through the probability-flow ODE, ExactGaussianScore, Exact diffused p(g, local | x) score for the figure-01 experiment.

### Community 12 - "Population_Dynamics.py"
Cohesion: 0.13
Nodes (18): load_models(), logistic_prey(), lotka_volterra(), main(), observations_for(), plot_comparisons_by_true_model(), Compare population-dynamics hypotheses for each possible true model.  This is th, Logistic prey growth with predator satiation. (+10 more)

### Community 13 - "Probability_Flow_validation.py"
Cohesion: 0.19
Nodes (18): add_cov_ellipse(), compass_annealed_map_theta(), configure_cpu_usage_limit(), draw_likelihood_ellipse_grid(), evaluate_kde_density_grid(), manifold_norm(), manifold_raw(), pfode_likelihood_log_prob() (+10 more)

### Community 14 - "test_instantaneous_divergence.py"
Cohesion: 0.16
Nodes (14): configure_cpu_usage_limit(), _evaluation_data(), GaussianScoreWithDrift, MockSBIm, Focused tests for the optional instantaneous divergence head., Hard-limit this process and its children to host CPU capacity., Independent Gaussian score with an analytic learned drift head., _small_model() (+6 more)

### Community 15 - "collect_xspace_likelihood_runtime_benchmark"
Cohesion: 0.19
Nodes (15): analytic_likelihood_map(), collect_xspace_likelihood_runtime_benchmark(), generate_model_data(), load_or_create_xspace_mc_reference(), model_mean(), normalize(), normalize_x(), progress() (+7 more)

### Community 16 - "plot_pfode_hutchinson_2d_likelihood_diagnostic"
Cohesion: 0.15
Nodes (14): benchmark_map_estimates(), benchmark_xspace_likelihood_map(), evaluate_pfode_likelihood_grid(), evaluate_pfode_likelihood_log_prob_grid(), likelihood_map_from_samples(), normalize_log_density_grid(), plot_pfode_hutchinson_2d_likelihood_diagnostic(), Return one MAP per observation and the elapsed inference time. (+6 more)

### Community 17 - "ScoreBasedInferenceModel"
Cohesion: 0.10
Nodes (11): ConditionTransformer, Diffusion model with a Transformer backbone., Forward pass of ConditionTransformer.         Args:             x:   (N, C, H, W, Train the model on the provided data          Args:             theta: Training, Sample from the model using the specified method          Args:             data, Evaluate the log-probability of the latent dimensions of `data` given its, KDE-free MAP estimate of the latent dimensions of `data` given its         condi, Refine shared and local parameters with a compositional joint score.          Un (+3 more)

### Community 18 - "log"
Cohesion: 0.24
Nodes (13): checkpoint_paths(), load_or_train_models(), log(), main(), make_pairplot(), plot_likelihood_reconstruction_distance(), plot_posterior_pfode_hutchinson_trace_samples(), promote_best_checkpoint() (+5 more)

### Community 19 - "ConditionTransformer"
Cohesion: 0.06
Nodes (72): configure_style(), main(), mean(), plot_exact_score(), plot_hierarchy(), plot_hierarchy_pairplot(), plot_shared_local(), Figure (+64 more)

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
Cohesion: 0.14
Nodes (32): DataFrame, configure_plot_style(), finish_error_axes(), line_with_band(), plot_banana_map_pareto(), plot_banana_posterior_examples(), plot_banana_shape_ablation(), plot_common_outputs() (+24 more)

### Community 29 - ".time_of_sigma"
Cohesion: 0.12
Nodes (29): configure_cpu_usage_limit(), main(), make_banana_data(), plot_sampler_pairplot(), Plot reference and sampler posteriors in the supplied Seaborn style., Generate the nonlinear posterior problem from Banana_posterior.ipynb., Reserve one free GPU with autocvd, otherwise use CPU without waiting.      ``aut, Document each problem-specific result directory. (+21 more)

### Community 35 - "analytic_posterior"
Cohesion: 0.50
Nodes (4): add_covariance_ellipse(), plot_posterior_examples(), Axes, Show intuitive posterior samples for the current sampler variants.

### Community 36 - "banana_reference_samples"
Cohesion: 0.33
Nodes (6): banana_log_likelihood(), banana_reference_map(), banana_reference_samples(), Unnormalised log likelihood; its maximum is zero., Return the exact posterior MAP for one banana observation., Draw an exact posterior reference by rejection from the Gaussian prior.

### Community 37 - "sampler_grid_mode"
Cohesion: 0.10
Nodes (36): banana_splits(), benchmark_summary(), density_grid(), evaluate_density(), evaluate_density_sweep(), evaluate_likelihood_convergence(), file_sha256(), fixed_raw_joint() (+28 more)

### Community 38 - "MultiObsSampler"
Cohesion: 0.12
Nodes (8): Analytic denoising (Tweedie) step from the final diffusion time t_end to t=0, Noise that is shared across the observation axis on the hierarchy dimensions, Per-feature scale for initial noise / Langevin steps. For "fnpe" the bridging, Basic Euler-Maruyama sampling method          Args:             data: Input data, Adaptive stochastic Heun solver for the reverse SDE.          Each proposal comp, Pure annealed Langevin dynamics: at every noise level, run `steps_per_level`, Corrector steps using Langevin dynamics          Args:             x: Input data, Build the reverse schedule; covariance modes always run through eps.

### Community 39 - "._get_score"
Cohesion: 0.22
Nodes (5): Get the composed score estimate with optional classifier-free guidance, First-order solver (in noise-scale space).          The probability-flow ODE dx, Second-order solver (in noise-scale space), Third-order solver (in noise-scale space), Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinemen

### Community 40 - ".sample"
Cohesion: 0.09
Nodes (14): Validate and normalize covariance input to shape (1 or N, H, H)., Validate optional Gaussian means to shape (1 or N, H)., Reset precision-repair counters without changing the sample return API., Regularize empirical covariance in float64 with shrinkage and a nugget., Return batched precision matrices using Cholesky solves., Estimate per-observation means and covariances from ordinary draws., Backward-compatible covariance-only wrapper., Estimate per-observation precision in memory-bounded draw batches. (+6 more)

### Community 42 - "._validate_precision"
Cohesion: 0.08
Nodes (66): analytic_maps(), build_config(), build_model(), build_summary_rows(), checkpoint_signature(), configure_cpu_usage_limit(), configure_plot_style(), create_plots() (+58 more)

### Community 43 - ".__init__"
Cohesion: 0.11
Nodes (13): Variance Exploding Stochastic Differential Equation (VESDE) class.         The V, Mean scaling of the perturbation kernel; identically 1 for the VESDE., Compute the standard deviation of p_{0t}(x(t) | x(0)) for VESDE.          Args:, Inverse of marginal_prob_std: the diffusion time t at which the marginal, VESDE, CovarianceDrawSampler, test_automatic_covariance_estimation_is_full_and_memory_bounded(), test_automatic_posterior_moments_return_matching_mean_and_covariance() (+5 more)

### Community 44 - ".time_of_sigma"
Cohesion: 0.10
Nodes (58): aggregate_by_n(), analytic_problem(), analytic_variant(), build_parser(), CacheSettingsMismatch, calibration_curve(), configure_cpu_limit(), configure_style() (+50 more)

### Community 47 - "Hierarchical_Linear_Quadratic.py"
Cohesion: 0.09
Nodes (51): analytic_references(), build_config(), checkpoint_signature(), Config, configure_cpu_limit(), configure_plots(), confusion_rows(), create_outputs() (+43 more)

### Community 48 - "infer_partial_pooling.py"
Cohesion: 0.09
Nodes (52): inference_dataset_glob(), _complete_artifact_joint_map(), _dataset_legend(), density_panel_counts(), diffusion_metadata(), estimated_runtime_seconds(), estimated_sweep_runtime_seconds(), _finish_figure() (+44 more)

### Community 49 - "compare_shared_local_composition_methods.py"
Cohesion: 0.09
Nodes (41): build_single_observation_covariance_bank(), empirical_covariances(), load_result(), load_score_checkpoint(), main(), metric_rows(), ndarray, Path (+33 more)

### Community 50 - "main"
Cohesion: 0.10
Nodes (41): build_single_observation_covariance_bank(), empirical_covariances(), load_result(), load_score_checkpoint(), main(), metric_rows(), ndarray, Path (+33 more)

### Community 51 - "data_pipeline.py"
Cohesion: 0.10
Nodes (38): _concatenate(), generate_test_data(), generate_training_data(), _leaves(), _normalizers(), Deterministic sharded data generation for every benchmark path., _sde_split(), _slice() (+30 more)

### Community 52 - "Tensor"
Cohesion: 0.10
Nodes (15): BenchmarkTask, JRNMMTask, LotkaVolterraTask, normal_cdf(), Tensor, Simulators and analytic toy scores used by the Figure 1--4 experiments.  The ben, SIR benchmark with ten binomial observations., Lotka--Volterra benchmark with the paper's 20-dimensional summary. (+7 more)

### Community 53 - "compositional_score_analytic.py"
Cohesion: 0.11
Nodes (37): benchmark(), finish(), gaussian_density(), grid_variants(), limit_cpus(), LinearGaussianScore, main(), make_problem() (+29 more)

### Community 54 - "migrate_artifact_names.py"
Cohesion: 0.16
Nodes (37): checkpoint_directory(), checkpoint_manifest_path(), checkpoint_path(), checkpoint_tag(), inference_dataset_filename(), inference_report_filename(), normalization_path(), Readable, stable names for partial-pooling artifacts. (+29 more)

### Community 55 - "joint_vs_legacy_annealed_score_ascent.py"
Cohesion: 0.11
Nodes (31): add_covariance_ellipse(), aggregate_metric(), analytic_posterior_particles(), AnalyticReference, benchmark(), build_parser(), configure_style(), error_metrics() (+23 more)

### Community 56 - "Compare_Upstream_PFODE.py"
Cohesion: 0.11
Nodes (31): checkpoint_comparison(), density_plot(), historical_training_config(), log(), loss_fix_effect_plot(), main(), run_training_worker(), stage2_checkpoint() (+23 more)

### Community 57 - "Compositional_Inference.py"
Cohesion: 0.14
Nodes (35): aggregate(), configure_cpu_usage_limit(), load_raw_data(), plot_gaussian_test(), plot_parabola_data_separation(), plot_parabola_model_selection(), plot_parabola_recovery(), plot_selected_time_sampling_sampler_grid() (+27 more)

### Community 58 - "Compositional_Inference.py"
Cohesion: 0.14
Nodes (35): aggregate(), configure_cpu_usage_limit(), load_raw_data(), plot_gaussian_test(), plot_parabola_data_separation(), plot_parabola_model_selection(), plot_parabola_recovery(), plot_selected_time_sampling_sampler_grid() (+27 more)

### Community 59 - "RunConfig"
Cohesion: 0.12
Nodes (34): build_config(), build_gaussian_test_model(), build_model(), build_parabola_model(), checkpoint_dir(), configure_plot_style(), evaluate_sampler_grid(), experiment_observation_scaling() (+26 more)

### Community 60 - "RunConfig"
Cohesion: 0.12
Nodes (34): build_config(), build_gaussian_test_model(), build_model(), build_parabola_model(), checkpoint_dir(), configure_plot_style(), evaluate_sampler_grid(), experiment_observation_scaling() (+26 more)

### Community 61 - "BenchmarkPaths"
Cohesion: 0.09
Nodes (14): parser(), Shared command-line implementation; wrappers enforce CPU affinity first., run_stage(), get_config(), Central benchmark configuration and presets., Reproducible hierarchical partial-pooling benchmark for COMPASS., BenchmarkPaths, Filesystem layout for all benchmark artifacts. (+6 more)

### Community 62 - "test_benchmark_contracts.py"
Cohesion: 0.09
Nodes (23): draw_batch_size(), _global_kde_log_density(), _global_kde_state(), _hierarchy_validation_state(), _is_numerical_map_failure(), _joint_candidate_diagnostics(), Evaluate a tail-robust marginal KDE, or return ``None`` if degenerate.      ``ga, Keep subject rows per transformer call within the training batch size. (+15 more)

### Community 63 - "PF_ODE_1.py"
Cohesion: 0.10
Nodes (28): analytic_log_prob(), configure_cpu_usage_limit(), initial_noise(), integrate_samples(), kde_log_prob(), main(), make_time_grid(), Method (+20 more)

### Community 64 - "plot_partial_pooling_comparison.py"
Cohesion: 0.16
Nodes (23): Return shared density limits resistant to a minority of solver outliers., _robust_density_limits(), _artifact_for_dataset(), _dataset_ids(), _finish(), _load_artifacts(), _load_json(), _load_run() (+15 more)

### Community 65 - "Tensor"
Cohesion: 0.11
Nodes (24): exact_hierarchical_posterior(), gaussian_test_analytic_posterior(), known_shared_covariance_full(), known_shared_mean(), known_shared_precision(), parabola_mean_torch(), Tensor, Joint posterior for [global, local_1, ..., local_N]. (+16 more)

### Community 66 - "Tensor"
Cohesion: 0.11
Nodes (24): exact_hierarchical_posterior(), gaussian_test_analytic_posterior(), known_shared_covariance_full(), known_shared_mean(), known_shared_precision(), parabola_mean_torch(), Tensor, Joint posterior for [global, local_1, ..., local_N]. (+16 more)

### Community 67 - "experiment_parabola"
Cohesion: 0.13
Nodes (22): exact_parabola_evidence_series(), exact_parabola_posterior(), experiment_parabola(), information_criterion_weights(), matched_parabola_cases(), parabola_log_likelihood_grid(), parabola_mean_numpy(), parabola_quadrature() (+14 more)

### Community 68 - "experiment_parabola"
Cohesion: 0.13
Nodes (22): exact_parabola_evidence_series(), exact_parabola_posterior(), experiment_parabola(), information_criterion_weights(), matched_parabola_cases(), parabola_log_likelihood_grid(), parabola_mean_numpy(), parabola_quadrature() (+14 more)

### Community 69 - "figures"
Cohesion: 0.10
Nodes (19): completed_datasets, correction, figures, inference_method, error_distributions.png, error_vs_observations.png, global_forest.png, global_parity.png (+11 more)

### Community 70 - "MultiObsSampler"
Cohesion: 0.20
Nodes (17): MultiObsSampler, Diagnostics for covariance conditioning and composed-precision repairs., Compositional score modeling for inference with multiple i.i.d. observations, configured_sampler(), MockSBIm, Focused CPU tests for compositional error damping and adaptive sampling., test_adaptive_evaluation_limit_fails_clearly(), test_adaptive_reverse_sde_is_finite_and_synchronized() (+9 more)

### Community 71 - "figures"
Cohesion: 0.11
Nodes (18): completed_datasets, correction, figures, inference_method, error_distributions.png, error_vs_observations.png, global_forest.png, global_parity.png (+10 more)

### Community 72 - "figures"
Cohesion: 0.11
Nodes (18): completed_datasets, correction, figures, inference_method, error_distributions.png, error_vs_observations.png, global_forest.png, global_parity.png (+10 more)

### Community 73 - "figures"
Cohesion: 0.11
Nodes (18): completed_datasets, correction, figures, inference_method, error_distributions.png, error_vs_observations.png, global_forest.png, global_parity.png (+10 more)

### Community 74 - "test_full_gaussian_composition.py"
Cohesion: 0.20
Nodes (18): configured_full_sampler(), configured_joint_sampler(), The shared composition must ignore current local states entirely., test_composed_precision_repair_updates_matrix_and_numerator(), test_covariance_modes_share_complete_eps_schedule(), test_full_gaussian_clamps_overflowing_tail_scores_before_composition(), test_full_gaussian_matches_algorithm_2_matrix_formula(), test_full_gaussian_reduces_to_diagonal_gauss() (+10 more)

### Community 75 - "test_variable_param_count.py"
Cohesion: 0.15
Nodes (14): _build_mtf(), _make_observations(), MockSBIm, Regression test for model comparison with models that have DIFFERENT numbers of, Analytic MAP: the posterior is Gaussian, so the mode equals the mean         the, Analytic likelihood log p(x | theta): N(theta[:2], SIGMA_X^2 I) over         the, (i) compare() runs with models of different theta dims,        (ii) param_count, criterion='bic' must run, penalize the extra parameter harder than AIC     (k*ln (+6 more)

### Community 76 - "__init__.py"
Cohesion: 0.14
Nodes (12): Choice-only hierarchical Bernoulli simulator., Hierarchical Bernoulli--lognormal full-observation simulator., Choice-only and full-observation hierarchical Bernoulli benchmarks., simulate_choice_only(), simulate_full_observation(), Scientific simulators used by the benchmark., beta_from_normal(), diffused_gaussian_prior_score() (+4 more)

### Community 77 - "tensor_numpy"
Cohesion: 0.27
Nodes (18): analytic_shared(), experiment_contract(), experiment_gaussian_global(), experiment_multi_vs_individual(), experiment_time_sampling(), experiment_time_sampling_sampler_grid(), Cross training-time draws with three composition methods and two grids., Run MTF once for observations from each candidate Gaussian model. (+10 more)

### Community 78 - "tensor_numpy"
Cohesion: 0.27
Nodes (18): analytic_shared(), experiment_contract(), experiment_gaussian_global(), experiment_multi_vs_individual(), experiment_time_sampling(), experiment_time_sampling_sampler_grid(), Cross training-time draws with three composition methods and two grids., Run MTF once for observations from each candidate Gaussian model. (+10 more)

### Community 79 - "damping_d1_method_sweeps.py"
Cohesion: 0.27
Nodes (17): build_parser(), cache_matches(), cell_spec(), collect_specs(), dedicated_cell_path(), failure_path(), is_numerical_divergence(), load_matching_failure() (+9 more)

### Community 80 - "figures"
Cohesion: 0.12
Nodes (16): completed_datasets, figures, inference_method, error_distributions.png, error_vs_observations.png, global_forest.png, global_parity.png, global_residuals.png (+8 more)

### Community 81 - "manifest.json"
Cohesion: 0.12
Nodes (16): completed_datasets, correction, figures, inference_method, error_distributions.png, global_parity.png, global_residuals.png, local_parity.png (+8 more)

### Community 82 - "BenchmarkConfig"
Cohesion: 0.18
Nodes (6): BenchmarkConfig, Configuration fields that can change simulated data or its layout., Training configuration, retaining legacy VE checkpoint signatures., Backward-compatible alias for the model/training signature., Signature used by pre-VPSDE data generated with the same preset., _signature()

### Community 83 - "._compositional_score"
Cohesion: 0.13
Nodes (8): Replace selected learned scores by scores of the supplied Gaussian moments., Return cached per-observation effective global precisions and R blocks., Return clean-moment GAUSS precisions for p(g | x_j)., Evaluate each pilot Gaussian p(g_t | x_j), independent of locals.          ``the, Repair and solve a composed global precision in float64., Return the noised Gaussian-prior score in y=x/alpha coordinates., Compose the per-observation scores on the hierarchy (shared parameter)         d, Exponential schedule with explicit data/noise endpoint semantics.

### Community 84 - "test_hierarchical_map.py"
Cohesion: 0.34
Nodes (11): analytic_shared_local_map(), make_joint(), MockSBIm, Focused tests for joint hierarchical score-ascent MAP refinement., Exact diffused score for g,l ~ Normal and x = g + l + noise., SharedLocalGaussianScore, test_all_global_and_single_observation_are_supported(), test_damped_joint_map_recovers_exact_shared_and_local_gaussian_mode() (+3 more)

### Community 85 - "test_memory_bounded_attention.py"
Cohesion: 0.22
Nodes (9): AttentionRecordingModel, DeterministicSampler, estimate_precision(), run_single_observation_sampler(), test_attention_capture_can_be_disabled_or_retained(), test_batched_and_unbatched_precision_match_deterministic_sampler(), test_precision_estimation_handles_non_divisible_draw_count(), test_precision_estimation_preserves_256_draws_with_bounded_calls() (+1 more)

### Community 86 - "figures"
Cohesion: 0.17
Nodes (11): completed_datasets, figures, error_distributions.png, global_forest.png, global_parity.png, global_residuals.png, local_parity.png, local_residuals.png (+3 more)

### Community 87 - "._shared_then_local_map"
Cohesion: 0.18
Nodes (6): Compare the models on the provided observations.         The results are saved i, Compute the log probability of the samples, Jointly optimize a KDE over the synchronized shared sample block.          The K, Find the marginal shared MAP, freeze it, then refine local MAPs., Find the joint mode of the multivariate distribution, test_shared_marginal_kde_optimizes_correlated_block_jointly()

### Community 88 - "test_composition_strategies.py"
Cohesion: 0.36
Nodes (9): sampler(), test_composition_changes_only_declared_global_coordinates(), test_diffused_prior_score_matches_analytic_ve_vp(), test_exact_required_formulas(), test_minibatch_damped_has_full_score_expectation(), test_minibatch_is_selected_before_network_forward_and_counted(), test_r1_required_names_reduce_to_single_score(), test_required_methods_are_finite_across_diffusion_time() (+1 more)

### Community 89 - "Tall-data Figures 1–4 in COMPASS"
Cohesion: 0.22
Nodes (8): Environment, Execute Figure 1, Execute Figure 2, Execute Figure 3, Execute Figure 4, Lightweight smoke configuration, Tall-data Figures 1–4 in COMPASS, What each command creates

### Community 91 - "JointScoreCompareMock"
Cohesion: 0.22
Nodes (3): JointScoreCompareMock, SymmetricMixtureScore, test_model_transfuser_joint_score_dispatches_without_changing_legacy_modes()

### Community 92 - "plot_global_observation_sensitivity.py"
Cohesion: 0.43
Nodes (7): create_figure(), main(), _parser(), Visualize observed DDM data while varying one global parameter at a time., _simulate_sweep(), _style_axis(), _violin_panel()

### Community 93 - "TimestepEmbedder"
Cohesion: 0.33
Nodes (4): Embeds scalar timesteps into vector representations., Create sinusoidal timestep embeddings.          Args:             t: a 1-D Tenso, Forward pass of TimestepEmbedder.                  Args:             t: (N,) ten, TimestepEmbedder

### Community 94 - "test_multistart_avoids_low_density_posterior_mean"
Cohesion: 0.29
Nodes (4): Return coherent, diverse joint starts and local-neighbourhood scales., Evaluate the composed joint log posterior for each refined candidate., AnalyticMixtureDensity, test_multistart_avoids_low_density_posterior_mean()

### Community 96 - "Partial-pooling benchmark summary"
Cohesion: 0.33
Nodes (5): ancestral, complete_pooling, native_joint, no_pooling, Partial-pooling benchmark summary

### Community 97 - "COMPASS global/local inference validation"
Cohesion: 0.33
Nodes (5): COMPASS global/local inference validation, Regenerate all figures without training or inference, Reproduce the exact-score metrics, Rerun the hierarchy stress test, Retrain and rerun the learned-score experiment

### Community 98 - "Hierarchical Partial-Pooling Benchmark"
Cohesion: 0.40
Nodes (4): Hierarchical Partial-Pooling Benchmark, Model and inference, Reproduction, Scientific model

## Knowledge Gaps
- **146 isolated node(s):** `completed_datasets`, `global_parity.png`, `local_parity.png`, `global_residuals.png`, `local_residuals.png` (+141 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ScoreBasedInferenceModel` connect `ScoreBasedInferenceModel` to `Compositional_Inference.py`, `ModelTransfuser`, `MultiObsSampler`, `VESDE`, `Sampler`, `Trainer`, `VPSDE`, `PFODE`, `.__init__`, `test_instantaneous_divergence.py`, `migrate_artifact_names.py`, `ScoreBasedInferenceModel.py`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `MultiObsSampler` connect `MultiObsSampler` to `MultiObsSampler`, `Compositional_Inference.py`, `PFODE`, `ScoreBasedInferenceModel`, `ConditionTransformer`, `ScoreBasedInferenceModel.py`, `MultiObsSampler`, `._get_score`, `.sample`, `TensorTupleDataset`, `.__init__`, `.time_of_sigma`, `compositional_score_analytic.py`, `joint_vs_legacy_annealed_score_ascent.py`, `test_full_gaussian_composition.py`, `._compositional_score`, `test_hierarchical_map.py`, `test_memory_bounded_attention.py`, `test_composition_strategies.py`, `JointScoreCompareMock`, `test_multistart_avoids_low_density_posterior_mean`, `ExactGaussianScore`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `ModelTransfuser` connect `ModelTransfuser` to `test_variable_param_count.py`, `.time_of_sigma`, `.__init__`, `PFODE`, `Hierarchical_Linear_Quadratic.py`, `ScoreBasedInferenceModel`, `ConditionTransformer`, `ScoreBasedInferenceModel.py`, `test_hierarchical_map.py`, `joint_vs_legacy_annealed_score_ascent.py`, `._shared_then_local_map`, `JointScoreCompareMock`, `test_multistart_avoids_low_density_posterior_mean`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `MultiObsSampler` (e.g. with `gauss()` and `ScoreBasedInferenceModel`) actually correct?**
  _`MultiObsSampler` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 48 inferred relationships involving `VESDE` (e.g. with `ScoreBasedInferenceModel` and `sampler()`) actually correct?**
  _`VESDE` has 48 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `Sampler` (e.g. with `AnalyticPosteriorScore` and `AnalyticSBIm`) actually correct?**
  _`Sampler` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `ModelTransfuser` (e.g. with `ScoreBasedInferenceModel` and `AnalyticMixtureDensity`) actually correct?**
  _`ModelTransfuser` has 18 INFERRED edges - model-reasoned connections that need verification._