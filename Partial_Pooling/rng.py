"""Stable named seed derivation from one root seed."""

import hashlib
import numpy as np


def derive_seed(root_seed, *parts):
    key = ":".join((str(int(root_seed)), *(str(part) for part in parts)))
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def generator(root_seed, *parts):
    return np.random.default_rng(derive_seed(root_seed, *parts))
