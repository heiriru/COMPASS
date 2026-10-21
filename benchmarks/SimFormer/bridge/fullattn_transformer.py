"""COMPASS trunks with unmasked attention between all tokens.

The legacy blocks pass ``key_padding_mask = 1 - c`` to attention, so a latent
token can only attend to *observed* tokens (plus the bias key/value): latents
never see each other. For a posterior with dependent parameters (the Two Moons
crescent, the HMM chain, the Tree) or the joint/likelihood conditionals, the
only path between latent coordinates is the final flattened linear read-out.
Simformer attends over all tokens and tells the network which ones are observed
with a condition-state token embedding instead.

Here the blocks get an all-ones ``c`` (no key is masked) and the condition state
enters through the per-node ``e_observed`` / ``e_latent`` embedding of
``cond_embed_transformer`` (otherwise the network could not tell a clean
observed value from a noisy latent one). Two bases:

  "legacy"      the full legacy trunk (per-node adaLN time modulation)
  "notime_add"  the 0.45M trunk without modulation, time added per block

``install(base)`` must precede constructing or loading a model.
"""

import importlib

import torch
import torch.nn as nn

from cond_embed_transformer import CondEmbedConditionTransformer
from notime_transformer import NoTimeModConditionTransformer


class FullAttnConditionTransformer(CondEmbedConditionTransformer):
    def forward(self, x, t, c, return_attn_weights=False):
        cc = c.reshape(-1, self.nodes_size, 1).to(x.dtype)
        tokens = (self.x_embedder(x) + self.node_bias
                  + cc * self.state_embedding[1] + (1 - cc) * self.state_embedding[0])
        t = self.t_embedder(t)
        keys = torch.ones_like(c)
        weights = []
        for block in self.blocks:
            if return_attn_weights:
                tokens, w = block(tokens, keys, t, True)
                weights.append(w.mean(0))
            else:
                tokens = block(tokens, keys, t)
        out = self.final_layer(tokens, t)
        return (out, torch.stack(weights, 0)) if return_attn_weights else out


class FullAttnSmallConditionTransformer(NoTimeModConditionTransformer):
    def __init__(self, nodes_size, hidden_size=64, depth=6, num_heads=16, mlp_ratio=4.0):
        super().__init__(nodes_size, hidden_size=hidden_size, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio)
        self.state_embedding = nn.Parameter(0.02 * torch.randn(2, nodes_size, hidden_size))

    def forward(self, x, t, c, return_attn_weights=False):
        cc = c.reshape(-1, self.nodes_size, 1).to(x.dtype)
        x = self.x_embedder(x) + cc * self.state_embedding[1] + (1 - cc) * self.state_embedding[0]
        te = self.t_embedder(t) if self.t_embedder is not None else None
        if self.t_input is not None:
            x = x + self.t_input(te).unsqueeze(1)
        keys = torch.ones_like(c)
        weights = []
        for block in self.blocks:
            if return_attn_weights:
                x, w = block(x, keys, te, True)
                weights.append(w.mean(0))
            else:
                x = block(x, keys, te)
        out = (nn.SiLU()(self.norm_final(x)).flatten(1) @ self.readout).squeeze(-1)
        return (out, torch.stack(weights, 0)) if return_attn_weights else out


def install(base="legacy", time_embedding_size=256):
    assert base in ("legacy", "notime_add"), base
    module = importlib.import_module("compass.ScoreBasedInferenceModel")
    if base == "legacy":
        module.ConditionTransformer = FullAttnConditionTransformer
    else:
        NoTimeModConditionTransformer.MODE = "add"
        NoTimeModConditionTransformer.TIME_EMBEDDING_SIZE = int(time_embedding_size)
        module.ConditionTransformer = FullAttnSmallConditionTransformer
