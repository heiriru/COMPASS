# COMPASS validation

Can COMPASS's model-comparison output be trusted, i.e. attributed to the models
rather than to the network, the composition rule, the PF-ODE or the criterion?

- **Plan, experiments, critical-thinking notes, status, findings:** [`TODO.md`](TODO.md)
- **Library changes** live in `src/compass/`; this folder holds experiments, tests and
  reports, one subfolder per TODO item.

| item | folder | status |
|---|---|---|
| A1 parameter counting in AIC/AICc/BIC | [`a1_information_criteria/`](a1_information_criteria/README.md) | done (closed form + pipeline) |
| A2 MLE plug-in instead of MAP | [`a2_maximum_likelihood/`](a2_maximum_likelihood/README.md) | done on oracles, incl. full pipeline; trained-network check open |
| B5/B6 PF-ODE likelihood accuracy with trained networks | [`b5_b6_pfode_accuracy/`](b5_b6_pfode_accuracy/README.md) | done (30 networks, GPU) |
| Likelihood improvement (training recipe) | [`likelihood_improvement/`](likelihood_improvement/README.md) | done: menu + EDM recommended (opt-in) |
| C9 seed noise and checkpoint rule | [`c9_network_seeds/`](c9_network_seeds/README.md) | done |
| A3, A4, B7, B8, C10–E20 | — | open, see `TODO.md` |

## Running

All commands run from the repository root with the project venv. Every script caps
itself to 3 CPU threads (`common/runtime.py`) and hides the GPUs.

```bash
.COMPASS/bin/python -m pytest -q validation/tests                         # ~25 s
.COMPASS/bin/python -m validation.a1_information_criteria.known_answer    # ~1 min
.COMPASS/bin/python -m validation.a2_maximum_likelihood.exactness         # ~30 s
# ~40 s per dataset; shard with --shard/--shards
.COMPASS/bin/python -m validation.a2_maximum_likelihood.pipeline_selection --n 16 --reps 400
.COMPASS/bin/python -m validation.a2_maximum_likelihood.analyze_pipeline
# figures (from saved results; PNG + PDF in each item's results/figures/)
.COMPASS/bin/python -m validation.a1_information_criteria.plots
.COMPASS/bin/python -m validation.a2_maximum_likelihood.plots
# B5/B6 (GPU via autocvd; ~3 h total): train 30 networks, evaluate, analyse + figures
bash validation/b5_b6_pfode_accuracy/run_training.sh 3
bash validation/b5_b6_pfode_accuracy/run_evaluation.sh 2
.COMPASS/bin/python -m validation.b5_b6_pfode_accuracy.analyze
```

## Shared helpers

- `common/runtime.py` — CPU cap and import paths; import it first.
- `common/plotting.py` — shared figure style (new = blue, legacy = orange, exact = aqua).
- `common/gaussian_oracles.py`:
  - linear-Gaussian models with exact-score COMPASS networks (`oracle_sbim`);
  - the nested pair used by A1/A2;
  - χ² and Wilson helpers.
