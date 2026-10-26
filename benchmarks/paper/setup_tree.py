#!/usr/bin/env python3
"""Build the benchmark tree: a detached worktree of the commit to benchmark, plus copies of the
untracked data and checkpoints from the main COMPASS tree (which has them all).

    python setup_tree.py [commit]         # default HEAD of this repository

- Files the commit tracks are never overwritten (rsync --ignore-existing). Every job therefore runs
  exactly the committed code.
- Symlinks into /export/data stay symlinks. They are inputs (simulation sets), never written.
- Earlier results of the main tree come along as read-only context. Every paper job writes under its
  own *_paper_* name.
- The archived posteriors of upstream BayesFlow's own sampler are copied into the fair benchmark's
  paper folder.
- The roadmap-2 / covariance-output GM and toy networks are copied without their results.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jobs as J  # noqa: E402

REPO = HERE.parents[1]
MAIN = Path("/export/home/rheinric/COMPASS")
EXCLUDE = [".git", ".COMPASS", ".bayesflow", "benchmarks", "graphify-out", "GCE/vendor", "__pycache__",
           "*.swp"]


def run(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def main():
    commit = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", commit], capture_output=True, text=True,
                         check=True).stdout.strip()
    tree = J.TREE
    if tree.exists():
        have = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if have != sha:
            raise SystemExit(f"{tree} is at {have[:8]}, not {sha[:8]}: remove it first (git worktree remove)")
    else:
        tree.parent.mkdir(parents=True, exist_ok=True)
        run("git", "-C", REPO, "worktree", "add", "--detach", tree, sha)
    rsync = ["rsync", "-a", "--ignore-existing"] + [f"--exclude={e}" for e in EXCLUDE]
    run(*rsync, f"{MAIN}/", f"{tree}/")
    venv = tree / ".COMPASS"
    if not venv.exists():
        venv.symlink_to(MAIN / ".COMPASS")
    # fair benchmark: upstream BayesFlow's archived posteriors (library-independent)
    posteriors = J.OUT / "fair" / "posteriors"
    posteriors.mkdir(parents=True, exist_ok=True)
    for path in Path("/export/data/rheinric/compass_pp_fair/posteriors").glob("bf_*.npz"):
        if not (posteriors / path.name).exists():
            shutil.copy2(path, posteriors / path.name)
    # networks only (no results): GM covariance-output nets and the Gaussian-toy nets
    for source, target, names in (
            (Path("/export/data/rheinric/covout/gm"), J.OUT / "gm", ("cov_s7", "cov_s8")),
            (Path("/export/data/rheinric/covout/toy"), J.OUT / "toy", ("cov", "cov_det"))):
        for name in names:
            out = target / name
            out.mkdir(parents=True, exist_ok=True)
            for f in (source / name).iterdir():
                if f.is_file() and f.suffix in (".pt", ".json") and f.name not in ("summary.json",):
                    if not (out / f.name).exists():
                        shutil.copy2(f, out / f.name)
    (J.OUT / "COMMIT").write_text(sha + "\n")
    print(f"benchmark tree {tree} at {sha}")


if __name__ == "__main__":
    main()
