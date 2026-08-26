#!/usr/bin/env python3
"""Assemble the churn-vs-baseline-vs-F-NPSE comparison into one plottable root.

``plot_partial_pooling_comparison.py`` compares *completed recovery runs*, and it
imposes two constraints this comparison violates out of the box:

1. **Distinct run signatures.** Churn is a monkeypatch on ``MultiObsSampler``, so
   ``infer_partial_pooling.py`` cannot see it and the churned and stock runs of
   the same method get the *identical* signature (``a1a823fc7f09e730`` here).
   They only differ by which artifact root they were written to.
2. **Distinct method keys.** ``_validate_runs`` rejects two runs that resolve to
   the same method, and ``method_metadata`` reads that key from each artifact's
   ``inference_method`` field -- which is also identical for both.

So this script builds a **new** root that holds copies of the selected runs under
distinct directory suffixes, and rewrites ``inference_method`` in the copied
artifacts so each arm resolves to its own key, label and colour. Unknown keys
fall back to a title-cased label and a colourblind-safe fallback colour, which is
exactly what we want for ``dpm2_gauss_hierarchical_churn2``.

Nothing is modified in place: the source runs under ``artifacts_churn_eta*/`` and
``artifacts/`` are only read, and every rewrite happens on the copy.

Usage:
    python Partial_Pooling/build_churn_comparison.py
    python Partial_Pooling/build_churn_comparison.py --arm eta4:dpm2_gauss_hierarchical
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
PRESET = "small"

# (source root, inference method, suffix for the copied method key)
DEFAULT_ARMS = (
    ("artifacts_churn_eta2", "dpm2_gauss_hierarchical", "churn2"),
    ("artifacts_churn_eta0", "dpm2_gauss_hierarchical", None),
    ("artifacts_churn_eta0", "langevin_fnpse", None),
)


def source_directory(root_name, method):
    parent = ROOT / root_name / "partial_pooling_recovery" / PRESET
    matches = sorted(path for path in parent.glob(f"{method}-*") if path.is_dir())
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {method!r} run below {parent}; found {matches}"
        )
    return matches[0]


def copy_arm(source, destination_parent, method, suffix):
    """Copy one run, renaming its method key when a suffix is given.

    The signature is the trailing component after the final '-', which is what
    ``_load_run`` globs on, so prefixing it keeps the directory discoverable
    while making it unique within this root.
    """
    signature = source.name.rsplit("-", 1)[1]
    if suffix:
        new_method = f"{method}_{suffix}"
        new_signature = f"{suffix}{signature}"
    else:
        new_method, new_signature = method, signature
    destination = destination_parent / f"{new_method}-{new_signature}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

    if suffix:
        # Rewrite the method identity on the copy only. `method_metadata` reads
        # the artifact first and the run config second, so both must move.
        for path in sorted(destination.glob("*-dataset-*.pt")):
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            artifact["inference_method"] = new_method
            artifact["inference_signature"] = new_signature
            torch.save(artifact, path)
        for path in sorted(destination.glob("*-run_config.json")):
            config = json.loads(path.read_text())
            config["inference_method"] = new_method
            path.write_text(json.dumps(config, indent=2, sort_keys=True))
        # Filenames carry the method too; keep them consistent so the globs in
        # `_load_artifacts` and `_load_run` still match exactly one candidate.
        for path in sorted(destination.iterdir()):
            if path.name.startswith(f"{method}-"):
                path.rename(path.with_name(
                    path.name.replace(f"{method}-", f"{new_method}-", 1)
                ))
    return new_signature, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "artifacts_churn_comparison")
    parser.add_argument("--arm", action="append", default=None,
                        help="root:method[:suffix]; repeatable, overrides the "
                             "default three-arm comparison.")
    parser.add_argument("--skip-plot", action="store_true")
    arguments = parser.parse_args()

    arms = DEFAULT_ARMS
    if arguments.arm:
        arms = []
        for entry in arguments.arm:
            parts = entry.split(":")
            arms.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))

    root = arguments.output_root
    root.mkdir(parents=True, exist_ok=True)
    # The plotter needs the normalizers (data) and nothing else from the real
    # tree; symlink rather than copy 330 MB of shards.
    for name in ("data", "checkpoints"):
        link = root / name
        if not link.exists():
            link.symlink_to(ROOT / "artifacts" / name, target_is_directory=True)
    recovery = root / "partial_pooling_recovery" / PRESET
    recovery.mkdir(parents=True, exist_ok=True)

    signatures = []
    for root_name, method, suffix in arms:
        source = source_directory(root_name, method)
        signature, destination = copy_arm(source, recovery, method, suffix)
        signatures.append(signature)
        print(f"  {root_name}/{method}"
              f"{'  -> ' + suffix if suffix else ''}  ->  {destination.name}")

    if len(set(signatures)) != len(signatures):
        raise SystemExit(f"signatures are not distinct: {signatures}")

    manifest = root / "churn_comparison_arms.json"
    manifest.write_text(json.dumps(
        [{"source_root": r, "method": m, "suffix": s, "signature": sig}
         for (r, m, s), sig in zip(arms, signatures)],
        indent=2,
    ))
    print(f"Wrote {manifest}")

    if arguments.skip_plot:
        return 0

    command = [
        sys.executable, str(ROOT / "plot_partial_pooling_comparison.py"),
        "--preset", PRESET, "--root", str(root),
    ]
    for signature in signatures:
        command += ["--run-signature", signature]
    command += ["--output-signature", signatures[0]]
    print("\n$ " + " ".join(command), flush=True)
    return subprocess.call(command, cwd=str(ROOT.parent))


if __name__ == "__main__":
    raise SystemExit(main())
