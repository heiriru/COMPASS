# benchmarks/SimFormer — COMPASS vs Simformer on Simformer's own benchmarks

Paper: Gloeckler et al., *All-in-one simulation-based inference*, ICML 2024.
Plan: [`PLAN.md`](PLAN.md). Running results: [`RESULTS.md`](RESULTS.md).

COMPASS is plugged into the **official** Simformer harness (`scoresbibm`) as an
extra method, so both methods share training data, observations, reference
posteriors, conditional masks and the C2ST code. See `PLAN.md` for why.

## Layout

```
simformer_official/     clone of github.com/mackelab/simformer (setup.py patched to not pip-install)
.simformer/             py3.10 venv: jax 0.4.23+cuda12 (cuDNN 8.9 pins), torch 2.1 cpu, sbi 0.22, sbibm
bridge/
  compass_method.py     registers `compass_*` methods with scoresbibm; proxy model
  compass_worker.py     COMPASS in its own .COMPASS env, driven over a pipe
  run_scoresbi.py       official Hydra entry point + COMPASS registration
  config/method/        compass_legacy.yaml, compass_tokenized.yaml
run_benchmark.py        sequential driver (autocvd, one GPU, one job at a time)
report.py               tables + figures after each job
results/<phase>/        harness output (summary.csv, per-run dirs); logs/, tables/, figures/
```

## Running

```bash
cd COMPASS/benchmarks/SimFormer
../../.COMPASS/bin/python run_benchmark.py pilot     # two_moons 10k, 3 methods
../../.COMPASS/bin/python run_benchmark.py fig4a     # 4 sbibm tasks x 3 budgets
../../.COMPASS/bin/python run_benchmark.py fig4b     # 4 all-conditional tasks x 3 budgets
../../.COMPASS/bin/python report.py                  # regenerate all tables/figures
```

The driver is restartable: cells already in `results/<phase>/summary.csv` are skipped.

## Setup notes (things that cost time)

- `jax[cuda12]==0.4.23` pulls cuDNN 9 and then fails with `CUDNN_STATUS_INTERNAL_ERROR`;
  install `simformer_official/src/requirements.txt` afterwards (cuDNN 8.9.4 + CUDA 12.2 libs).
- Their `setup.py` runs `pip install torch==2.1.0+cpu` / `sbibm` via `os.system` at
  install time; that is disabled here and done explicitly.
- Their Hydra `config_path` resolves to a top-level `config` package, so
  `run_scoresbi.py` puts `simformer_official/src/scoresbibm` on `sys.path`.
- Never call `jax.devices()` without `CUDA_VISIBLE_DEVICES` set: it opens a
  context on every GPU of this shared host.
- COMPASS `Trainer(verbose=True)` sleeps 0.2 s per epoch; at 1k simulations
  (1 step/epoch, 5000 epochs) that alone is 17 minutes. The worker trains with
  `verbose=False`.
