"""Moved to :mod:`compass.latent_attention` (now a library feature).

Kept as an alias so existing drivers, checkpoints with a ``latent_attention.json``
sidecar, and ``from compass.experimental.latent_attention import ...`` keep working.
New code should pass ``latent_attention=`` / ``latent_roles=`` to
``ScoreBasedInferenceModel``, which records the mode in the checkpoint itself.
"""

from ..latent_attention import (  # noqa: F401
    MODES, ROLE_CODES, SIDECAR, LatentAttentionTransformer, _token_roles,
    enable_latent_attention, latent_attention_payload, latent_edges,
    load_latent_attention, save_latent_attention,
)
