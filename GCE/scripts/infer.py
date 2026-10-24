#!/usr/bin/env python3
"""Composed (many-star) inference with a trained single-star network.

Sources: an observed/mock sample from gce.data_sources (``tng_mock``, ``nissen``,
``chempy_tng_mock``, ``chempy_alt_mock``) or a known-answer galaxy file
(``galaxy:<path to galaxy-XXXXX.npz>``, noise drawn with the network's profile).
Writes the posterior (npz) and, for a model with a nested null, its Bayes factor
against the universal IMF (json).

    python scripts/infer.py --network <dir> --name universal-tng_mock-s7 --source tng_mock --stars 200
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gce.paths import configure  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stars", type=int, default=0, help="> 0: random subset of this many stars")
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--draw-batch", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    configure(cpus=3, gpu=arguments.device.startswith("cuda"))

    import numpy as np
    from gce import compass_setup, data_sources
    from gce.models import birth_time_from_u
    from gce.observation import make_observations

    model, layout, normalizer, sidecar = compass_setup.load(arguments.network, arguments.name,
                                                            arguments.checkpoint == "final", arguments.device)
    truth, notes = {}, []
    if arguments.source.startswith("galaxy:"):
        galaxy = np.load(arguments.source.split(":", 1)[1])
        ok = galaxy["failure"] == ""
        notes.append(f"{int((~ok).sum())} stars dropped (Chempy failed)")
        u = galaxy["locals"][ok][:, layout.model.local_names.index("u_birth")]
        x = make_observations(galaxy["abundances"][ok], birth_time_from_u(u), layout.profile,
                              np.random.default_rng(arguments.seed))
        truth = {"globals": galaxy["globals"].tolist(), "locals": galaxy["locals"][ok].tolist()}
    else:
        sample = data_sources.SOURCES[arguments.source]()
        notes += sample.notes
        truth = {k: v for k, v in sample.truth.items() if not isinstance(v, np.ndarray)}
        x, more = sample.x(layout.profile)
        notes += more
    if arguments.stars and arguments.stars < len(x):
        index = np.sort(np.random.default_rng(arguments.subset_seed).choice(len(x), arguments.stars, replace=False))
        x = x[index]
    composed = compass_setup.compose(model, layout, normalizer, x, sidecar, arguments.draws, arguments.draw_batch,
                                     arguments.device, arguments.seed, timesteps=arguments.timesteps)
    bayes = compass_setup.nested_bayes_factor(layout.model, composed) if layout.model.nested_null else {}
    out = Path(arguments.out or Path(arguments.network) / "posteriors" /
               f"{arguments.name}-{arguments.source.replace(':', '_').replace('/', '_')}-n{len(x)}-s{arguments.seed}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), globals=composed["globals"], locals=composed["locals"], x=x)
    summary = {"network": arguments.network, "name": arguments.name, "source": arguments.source, "stars": len(x),
               "draws": arguments.draws, "timesteps": arguments.timesteps, "checkpoint": arguments.checkpoint,
               "global_names": list(composed["global_names"]),
               "global_median": np.median(composed["globals"], 0).tolist(),
               "global_sd": composed["globals"].std(0).tolist(), "truth": truth, "notes": notes,
               "nested_bayes_factor": bayes, "solver_stats": {k: v for k, v in composed["solver_stats"].items()
                                                              if isinstance(v, (int, float, str))}}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("stars", "global_names", "global_median", "global_sd",
                                               "nested_bayes_factor")}, indent=2, default=float))


if __name__ == "__main__":
    main()
