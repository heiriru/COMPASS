"""Noise twins (Gaussian / Laplace / exponential) with exact scores: the rules alone.

Reuses ``tutorials/.../exponential_score_comparison/tilt_noise_comparison.py``
unmodified: its exact analytic joint-score model, datasets, grid reference and
statistics. New rules are added to its ``RULES`` table in-process. The
exponential twin is the adversarial case for ``laplace``: its local conditional
is Uniform(g, x_j), flat with a g-dependent support, so no Gaussian-in-l
approximation holds there.

    python twins.py -- --device cpu --methods gauss_jacobian gauss_jacobian_FD arruda_direct arruda_laplace
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

BENCH = (common.COMPASS / "tutorials" / "Compositional_Inference_Fix" / "New_Attempt"
         / "exponential_score_comparison")
sys.path.insert(0, str(BENCH))

import tilt_noise_comparison as bench  # noqa: E402

CHURN = {"churn_eta": 4.0}   # the table's own DPM2 setting


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    for name, (settings, kwargs) in common.NEW_METHODS.items():
        bench.RULES[name] = {**kwargs, **CHURN}
    bench.MultiObsSampler = common.configured_sampler_class()
    original = bench.run_rule

    def run_rule(model, module, x, grid, name, arguments, device):
        common.ACTIVE.clear()
        common.ACTIVE.update(common.NEW_METHODS.get(name, ({}, {}))[0])
        # Only the shared marginal is graded here, and the analytic model has
        # no single-observation sampler, so the ancestral local stage is off.
        common.ACTIVE["ancestral_locals"] = False
        return original(model, module, x, grid, name, arguments, device)

    bench.run_rule = run_rule
    if "--output" not in argv:
        argv += ["--output", str(HERE / "artifacts" / "twins")]
    return bench.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
