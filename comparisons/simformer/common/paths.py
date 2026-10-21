"""Where each stage writes, and the manifest every stage must leave behind."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPASS_ROOT = ROOT.parents[1]

#: Simulations and reference posteriors are shared by every experiment: they are
#: a property of the task and the seed, not of the arm being scored. Generating
#: them once is what keeps e01, e03, e06 and e07 comparable to each other.
SHARED_DATA = ROOT / "data"


def experiment_dir(name):
    return ROOT / name


def artifacts(name, *parts):
    path = experiment_dir(name).joinpath("artifacts", *parts)
    return path


def data_path(*parts):
    return SHARED_DATA.joinpath(*parts)


def ensure(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "-C", str(COMPASS_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_manifest(path, payload):
    """Record what produced an artifact. Every stage writes one.

    The partial-pooling benchmark's hardest-won lesson (``docs/``,
    ``compass-benchmark-gotchas``) is that an artifact without a manifest is an
    artifact you cannot later prove is comparable to another one. Data
    signature, seed, arm and revision go in here, always.
    """
    path = Path(path)
    ensure(path.parent)
    record = dict(payload)
    record.setdefault("written_at", datetime.now(timezone.utc).isoformat())
    record.setdefault("git_revision", git_revision())
    record.setdefault("python", platform.python_version())
    record.setdefault("host", platform.node())
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return path
