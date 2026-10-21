"""Legacy ConditionTransformer plus Simformer-style token identity.

The legacy token is ``value * learned per-node vector``: which variable a token
is, and whether it is observed or latent, reach the network only through that
multiplicative vector and the ``1 - c`` attention key mask. Simformer's token
concatenates a value embedding, a variable-ID embedding and a condition-state
embedding. This variant adds, before the first block,

    token += node_bias[node] + (c * e_observed + (1 - c) * e_latent)[node]

(``2 * nodes * hidden + nodes * hidden`` parameters, <1% of the model). The
per-node adaLN, the attention mask, the read-out and the initialisation are the
legacy trunk unchanged.

``install()`` swaps the class ``ScoreBasedInferenceModel`` builds for
``arch="legacy"``; call it before constructing *or loading* a model.
"""

import importlib

import torch
import torch.nn as nn

from compass.ConditionTransformer import ConditionTransformer


class CondEmbedConditionTransformer(ConditionTransformer):
    def __init__(self, nodes_size, hidden_size=64, depth=6, num_heads=16, mlp_ratio=4.0, **kwargs):
        super().__init__(nodes_size, hidden_size=hidden_size, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, **kwargs)
        self.node_bias = nn.Parameter(0.02 * torch.randn(1, nodes_size, hidden_size))
        self.state_embedding = nn.Parameter(0.02 * torch.randn(2, nodes_size, hidden_size))

    def forward(self, x, t, c, return_attn_weights=False):
        cc = c.reshape(-1, self.nodes_size, 1).to(x.dtype)
        tokens = (self.x_embedder(x) + self.node_bias
                  + cc * self.state_embedding[1] + (1 - cc) * self.state_embedding[0])
        t = self.t_embedder(t)
        weights = []
        for block in self.blocks:
            if return_attn_weights:
                tokens, w = block(tokens, c, t, True)
                weights.append(w.mean(0))
            else:
                tokens = block(tokens, c, t)
        out = self.final_layer(tokens, t)
        return (out, torch.stack(weights, 0)) if return_attn_weights else out


def install():
    module = importlib.import_module("compass.ScoreBasedInferenceModel")
    module.ConditionTransformer = CondEmbedConditionTransformer
