# Arruda compositional diffusion runs

This directory is an isolated reproduction of the BayesFlow 2.0.13
compositional-diffusion tutorial for COMPASS's two shared-global benchmarks:

* `gm`: the asymmetric, bimodal Gaussian-mixture tall posterior;
* `ik`: the shared first-link angle in inverse kinematics.

It uses an amortized *single-observation* diffusion posterior, then composes
the learned scores with the exact clean prior score through BayesFlow's
`CompositionalApproximator.compositional_sample`.  This is the Arruda et al.
method, not the earlier Geffner annealed-Langevin implementation.

## Environment

`setup.sh` installs the tutorial-pinned `bayesflow==2.0.13` under `.vendor`.
It deliberately reuses COMPASS's PyTorch/CUDA environment, avoiding a second
multi-gigabyte torch installation.  It sets `KERAS_BACKEND=torch` and adds the
vendor directory to `PYTHONPATH`.

## Run

From `COMPASS`:

```bash
bash Arruda/setup.sh
bash Arruda/run.sh --benchmark gm --stage all
bash Arruda/run.sh --benchmark ik --stage all
```

The defaults mirror the tutorial's offline training setup: 16,384 simulations,
batch size 64, and 1,000 epochs.  Sampling uses the tutorial's
`method="two_step_adaptive"`, `steps="adaptive"`, and `mini_batch_size=10`.
For each tall dataset it records both the tutorial-default bridge (`d0=d1=1`)
and a damped bridge (`d0=d1=0.1`), because Arruda et al. identify growth of the
composed score with the number of groups as the numerical failure mode.

Results and plots are saved below `artifacts/<benchmark>/`.  The exact
posterior curves are diagnostic references only; inference itself uses the
learned BayesFlow local score plus the supplied analytic prior score.
