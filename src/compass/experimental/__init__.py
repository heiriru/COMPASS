"""Experimental extensions that leave the production modules untouched.

- ``latent_attention``: latent-to-latent attention rules for the shared trunk.
- ``marginal_schur``: Arruda-style marginal-global composition via a Schur
  reduction of the joint score (``correction="arruda_schur"``).
"""

from .latent_attention import (
    MODES as LATENT_ATTENTION_MODES, LatentAttentionTransformer,
    enable_latent_attention, save_latent_attention, load_latent_attention,
)
from .marginal_schur import MarginalSchurSampler, install as install_marginal_schur
