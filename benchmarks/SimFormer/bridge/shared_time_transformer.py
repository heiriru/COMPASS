"""Legacy ConditionTransformer with the time conditioning shared across nodes.

In ``compass.ConditionTransformer`` every block's adaLN layer maps the time
embedding to ``6 * nodes * hidden`` values -- a separate time-dependent shift,
scale and gate for every variable -- and the final layer to ``2 * nodes *
hidden``. That layer is ~95% of the parameters (2.8M at 4 variables, 12.9M at
20; Simformer has 180k at any size). This variant is the DiT-standard form:
the adaLN maps time to ``6 * hidden`` (``2 * hidden`` in the final layer) and
the same modulation is broadcast over all tokens.

The per-node adaLN shift was also the only additive per-variable term in the
legacy trunk (its token is ``value * learned vector``, so a z-scored value near
0 gives a near-zero token whatever the variable). To keep node identity, the
input embedding gets a learned per-node bias (``nodes * hidden`` parameters),
the analogue of Simformer's variable-ID embedding. Everything else -- joint
LayerNorm over (nodes, hidden), the ``1 - c`` key-padding mask, the flattened
read-out, the initialisation -- is the legacy trunk unchanged.

``install(time_embedding_size)`` swaps the class that
``ScoreBasedInferenceModel`` builds for ``arch="legacy"``; it must be called
before a model is constructed *or loaded*, so checkpoints round-trip.
"""

import importlib

import torch
import torch.nn as nn

from compass.ConditionTransformer import Mlp, TimestepEmbedder, modulate


class SharedInputEmbedder(nn.Module):
    def __init__(self, nodes_size, hidden_size):
        super().__init__()
        self.embedding_params = nn.Parameter(torch.ones(1, nodes_size, hidden_size))
        self.node_bias = nn.Parameter(0.02 * torch.randn(1, nodes_size, hidden_size))

    def forward(self, x):
        return x.unsqueeze(-1) * self.embedding_params + self.node_bias


class SharedTimeBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, nodes_size, mlp_ratio=4.0, time_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.nodes_size = nodes_size
        self.norm1 = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, batch_first=True)
        self.norm2 = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_size, 6 * hidden_size))

    def forward(self, x, c, t, return_attn_weights=False):
        # (B, 1, 6H): one modulation for all nodes, broadcast over the node axis.
        mod = self.adaLN_modulation(t).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        key_padding_mask = (1 - c).type(torch.bool)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=return_attn_weights,
                                              key_padding_mask=key_padding_mask)
        x = x + gate_msa * attn_output
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return (x, attn_weights) if return_attn_weights else x


class SharedTimeFinalLayer(nn.Module):
    def __init__(self, hidden_size, nodes_size, time_embedding_size=256):
        super().__init__()
        self.norm_final = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)
        self.embedding_params = nn.Parameter(torch.zeros(nodes_size * hidden_size, nodes_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_size, 2 * hidden_size))

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).unsqueeze(1).chunk(2, dim=-1)
        x = nn.SiLU()(modulate(self.norm_final(x), shift, scale))
        return (x.flatten(1) @ self.embedding_params).squeeze(-1)


class SharedTimeConditionTransformer(nn.Module):
    TIME_EMBEDDING_SIZE = 256   # set by install()

    def __init__(self, nodes_size, hidden_size=64, depth=6, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        te = self.TIME_EMBEDDING_SIZE
        self.nodes_size = nodes_size
        self.num_heads = num_heads
        self.time_embedding_size = te
        self.x_embedder = SharedInputEmbedder(nodes_size, hidden_size)
        self.t_embedder = TimestepEmbedder(nodes_size, hidden_size, mlp_ratio=mlp_ratio,
                                           frequency_embedding_size=te)
        self.blocks = nn.ModuleList([
            SharedTimeBlock(hidden_size, num_heads, nodes_size, mlp_ratio=mlp_ratio, time_embedding_size=te)
            for _ in range(depth)])
        self.final_layer = SharedTimeFinalLayer(hidden_size, nodes_size, time_embedding_size=te)
        self.initialize_weights()

    def initialize_weights(self):
        # Same scheme as the legacy trunk (xavier, small time MLP, adaLN-Zero).
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, c, return_attn_weights=False):
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        weights = []
        for block in self.blocks:
            if return_attn_weights:
                x, w = block(x, c, t, True)
                weights.append(w.mean(0))
            else:
                x = block(x, c, t)
        x = self.final_layer(x, t)
        return (x, torch.stack(weights, 0)) if return_attn_weights else x


def install(time_embedding_size=256):
    SharedTimeConditionTransformer.TIME_EMBEDDING_SIZE = int(time_embedding_size)
    module = importlib.import_module("compass.ScoreBasedInferenceModel")   # the module, not the class
    module.ConditionTransformer = SharedTimeConditionTransformer
