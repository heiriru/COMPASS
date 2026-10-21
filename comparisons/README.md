# `comparisons/`

Apples-to-apples reproductions of other methods' published experiments, run with
COMPASS.

One subdirectory per paper. Each is self-contained: the paper's task
definitions, the paper's protocol and metric, and COMPASS arms configured to
that protocol rather than to COMPASS's own defaults. The point is to produce
numbers that can sit in the same table as the published ones, so the rule
throughout is that a setting stated in the paper is used **even where COMPASS
would choose differently** — and every place COMPASS cannot match is recorded
explicitly rather than quietly substituted.

| directory | paper | status |
|---|---|---|
| `simformer/` | Gloeckler et al., *All-in-one simulation-based inference*, ICML 2024 (arXiv:2404.09636) | scaffolded; nothing trained |

Distinct from `Compositional_score_testing/`, which reproduces the *tall-data*
paper's figures inside COMPASS's own framing (bounded priors mapped to a
standard-normal latent, COMPASS composition rules as the subject). Here the
framing belongs to the paper being compared against.
