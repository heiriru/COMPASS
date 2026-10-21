# E03 — C2ST over arbitrary conditionals

**Paper:** §4.1, Fig. 4b; Fig. A5b/A6b. Tasks Tree and HMM are introduced here
(A2.2).

## What they did

Score not just the posterior but *arbitrary* conditionals of the joint. Four
tasks — Tree, HMM, Two Moons, SLCP — with 100 randomly selected conditional or
full-joint distributions each, reference samples by MCMC (slice + MH for Two
Moons/SLCP, HMC for Tree/HMM; recipes in A2.2 and in `common/paper.py`), and
C2ST against those references. Conclusion: Simformer models all conditionals
accurately, and training on all of them does not cost posterior accuracy.

Tree: a nonlinear tree factorisation with highly multimodal conditionals.
HMM: θ is a 10-step Gaussian random walk, `x_i ~ N(θ_i², 0.5²)` — a bimodal,
correlated posterior over a Markov chain.

## What we do — and why this is the interesting one

This is the direct test of COMPASS's key-padding convention. Under
`key_padding_mask = 1 - c`, a **latent** token is not a key, so latent tokens
never attend to one another and all latent–latent dependence must travel through
the residual stream and the MLP. Simformer's attention is dense.

The prediction is specific: COMPASS should be at its weakest exactly where the
conditional couples many latent coordinates to each other. HMM is the sharpest
case (10 coupled latent θ), Tree next, then SLCP, then Two Moons.

If the prediction holds, the follow-up is small and is the actual contribution:
add a `dense_attention` switch that admits latent tokens as keys, retrain, and
re-score. That ablation is a result about score-network design, not a bug fix.

## Running

```bash
python e03_all_conditionals/run.py masks     --task hmm          # freeze 100 masks
python e03_all_conditionals/run.py reference --task hmm          # expensive
# train the arm in e01 first — checkpoints are shared
python e03_all_conditionals/run.py sample    --task hmm --arm paper_matched --budget 10000
python e03_all_conditionals/run.py score     --task hmm --arm paper_matched --budget 10000
```

## Watch out for

- **The mask set must be frozen once and shared by every arm.** `masks` refuses
  to overwrite without `--overwrite`, and overwriting invalidates every
  reference and every score already computed against it.
- **Reference cost.** 100 conditionals × 10,000 chains × 5,000 HMC steps is the
  dominant cost in this whole directory. Consider fewer chains for a pilot, but
  never fewer for a published number — C2ST is biased by small reference sets.
- **HMC step size.** `common/reference.py` defaults to 0.1 with 20 leapfrog
  steps; the paper states only the step count. Check the acceptance rate
  reported by the `reference` stage (aim 0.6–0.9) before trusting the draws.
- **The paper does not state its mask distribution.** We draw uniformly over
  non-degenerate subsets. Any other choice is defensible; it just has to be the
  same for every arm and stated in the write-up.
