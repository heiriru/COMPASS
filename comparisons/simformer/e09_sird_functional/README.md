# E09 — SIRD with ∞-dimensional parameters — **BLOCKED**

**Paper:** §4.3, Fig. 6, coverage in Fig. A13.

## What they did

SIRD epidemiological model with a **time-varying contact rate** β(t) drawn from
a Gaussian-process prior (RBF kernel, length-scale 7, scale 2.5², squashed
through a sigmoid to [0,1]) alongside two scalar parameters γ, μ ~ U(0, 0.5).
Log-normal observation noise, σ = 0.05. Irregularly-spaced measurements of I, R
and D.

Two capabilities make this work, and COMPASS has neither:

1. **Fourier index embeddings.** A function-valued variable's token identity is
   a shared embedding vector plus a random Fourier embedding of its index (time),
   so the network can be queried at *any* time, not only a training grid (§3.1).
2. **Random subsampling of the index set** during training (A1.2), which by the
   Kolmogorov extension theorem is what makes the learned finite-dimensional
   marginals consistent.

They then demonstrate parameter-conditioned posteriors: condition on four
measurements of β(t) plus one infection measurement and sample the rest.

## Why this cannot run today

COMPASS's token identity comes from a fixed schema (`src/compass/schema.py`:
`Atomic`, `IID`, `Group`) with a learned per-block embedder; there is no
continuous index embedding and no training-time subsampling of nodes. The node
vector has a fixed width baked into the checkpoint.

## Route in, if it is ever wanted

`schema.py` is the right hook — a `Functional` block carrying (shared embedding,
Fourier index features) beside the existing three, with the trainer subsampling
its items per batch. That is a real feature, not a benchmark script, and it
should wait until the `Group` block work in `docs/group_block_progress.md` has
landed and settled.

## Recommendation

Deferred. The scientific demonstration is attractive but the prerequisite is a
core-library feature; do not start it from inside a comparison directory.
