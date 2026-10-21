#!/usr/bin/env python3
"""Run ``pp.py infer`` (partial pooling) with a gauss_jacobian_FD speed-up variant.

pp.py and the Partial_Pooling pipeline are used unchanged; this wrapper only
(1) replaces the Jacobian schedule pp.py's plan passes with the variant's
(variants.py), (2) adds a Laplace warm start when the variant asks for one, and
(3) adds the sampler's row counters to each dataset's saved ``solver_stats``.

    python bench_pp.py <variant> [pp.py infer arguments ...]
    python bench_pp.py band1-15 --mode hierarchical --precondition Rg --train-prior-scale 1.5 \
        --datasets 10 --tag speed_band1-15

The outputs land where pp.py puts them (recovery/<tag>/gauss_jacobian_FD-s100-t50).
"""
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPASS = HERE.parent
sys.path[:0] = [str(HERE), str(COMPASS / "experiments_arruda_schur")]

from variants import parse  # noqa: E402

JACOBIAN = {"gauss_jacobian", "gauss_jacobian_FD"}
SCHEDULE_KEYS = ("jacobian_refresh", "jacobian_refresh_low_noise", "jacobian_refresh_lambda")
COUNTERS = ("evaluated_row_samples", "jacobian_row_samples", "jacobian_refreshes",
            "jacobian_secant_updates", "jacobian_secant_skipped")


def main():
    spec, rest = sys.argv[1], sys.argv[2:]
    kwargs, warm = parse(spec)

    import pp  # sets up paths, the CPU cap and the GPU
    from compass.experimental.marginal_schur import MarginalSchurSampler

    totals = defaultdict(int)
    cache = {}
    original_sample = MarginalSchurSampler.sample

    def sample(self, world_size, data, condition_mask=None, **options):
        if options.get("correction") in JACOBIAN:
            for key in SCHEDULE_KEYS:
                options.pop(key, None)
            options.update(kwargs)
            if warm is not None:
                # One Laplace fit per dataset, shared by its draw batches.
                if "fit" not in cache:
                    cache["fit"] = self.laplace_fit(
                        data, condition_mask, warm, options["hierarchy"],
                        prior=options.get("prior"), local_prior=options.get("local_prior"),
                        correction=options["correction"],
                        denoise_clamp=options.get("denoise_clamp", 5.0),
                        device=options.get("device", "cuda"))
                    diagnostics = cache["fit"]["diagnostics"]
                    totals["warm_rows"] += diagnostics["row_samples"]  # already includes FD copies
                init = self.laplace_draw(cache["fit"], options["num_samples"])
                options.update(start_lambda=warm, init_state=init)
        out = original_sample(self, world_size, data, condition_mask=condition_mask, **options)
        stats = self.solver_stats or {}
        for key in COUNTERS:
            totals[key] += int(stats.get(key, 0))
        return out

    MarginalSchurSampler.sample = sample
    original_dataset = pp.ipp._sample_dataset

    def sample_dataset(*args, **options):
        totals.clear()
        cache.clear()
        posterior, stats, runtime = original_dataset(*args, **options)
        stats = dict(stats)
        stats.update(totals)
        stats["variant"] = spec
        return posterior, stats, runtime

    pp.ipp._sample_dataset = sample_dataset
    sys.argv = ["pp.py", "infer", "--method", "gauss_jacobian_FD", *rest]
    pp.main()


if __name__ == "__main__":
    main()
