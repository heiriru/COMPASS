"""Atomic serialization, configuration compatibility, and manifests."""

import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy
import scipy
import torch

from .config import REFERENCE_REVISION, REFERENCE_URL


def _atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    return path, Path(temporary)


def atomic_json(path, payload):
    path, temporary = _atomic_path(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path, text):
    path, temporary = _atomic_path(path)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path, payload):
    path, temporary = _atomic_path(path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    path, temporary = _atomic_path(path)
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_csv_atomic(path, row, key_fields=None):
    path = Path(path)
    existing = []
    if path.exists():
        with path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    if key_fields:
        existing = [
            old for old in existing
            if any(str(old.get(key)) != str(row.get(key)) for key in key_fields)
        ]
    atomic_csv(path, [*existing, row], fieldnames=list(row))


def reusable(path, signatures, force=False, signature_key="config_signature"):
    path = Path(path)
    if not path.exists() or force:
        return False
    payload = torch.load(path, map_location="cpu") if path.suffix == ".pt" else json.loads(path.read_text())
    expected = (
        {signatures} if isinstance(signatures, str)
        else set(signatures)
    )
    actual = payload.get(signature_key, payload.get("config_signature"))
    if actual not in expected:
        raise RuntimeError(
            f"Refusing to reuse {path}: signature {actual!r} is not in {sorted(expected)!r}. "
            "Pass --force to replace it."
        )
    return True


def manifest(config, stage, cpu_limit, signature=None, signature_kind="model", **extra):
    signature = config.signature if signature is None else signature
    return {
        "stage": stage, "config": config.to_dict(),
        "config_signature": signature,
        f"{signature_kind}_signature": signature,
        "root_seed": config.root_seed,
        "reference_url": REFERENCE_URL,
        "reference_revision": REFERENCE_REVISION,
        "git_revision": repository_revision(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": str(numpy.__version__), "scipy": str(scipy.__version__),
        "cpu_limit": cpu_limit, **extra,
    }


def repository_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
