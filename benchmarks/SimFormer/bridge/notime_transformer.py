"""Legacy ConditionTransformer with the adaLN time modulation removed.

The legacy trunk's per-node adaLN (shift/scale/gate per variable and block) is
~95% of the parameters. These variants drop all modulation -- every block is a
plain pre-LayerNorm residual block -- and differ only in how (or whether) the
diffusion time enters:

  "input"  time embedding projected to ``hidden`` and added once to every token
  "add"    a shared time vector added to the tokens in every block (Simformer:
           Linear(time) added per layer, no scale/gate)
  "none"   no time input at all; the score is ``f(x_t) / sigma(t)`` through
           COMPASS's output scaling only, so the noise level must be inferred
           from x_t itself

All keep the per-node ID bias of the shared-time variant (with no per-node
modulation it is the only thing that identifies a token's variable), the joint
LayerNorm, the ``1 - c`` key-padding mask and the flattened read-out.
``install(mode)`` must precede constructing or loading a model.
"""

import importlib

import torch
import torch.nn as nn

from compass.ConditionTransformer import Mlp, TimestepEmbedder
from shared_time_transformer import SharedInputEmbedder


class PlainBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, nodes_size, mlp_ratio, time_embedding_size, add_time):
        super().__init__()
        self.norm1 = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, batch_first=True)
        self.norm2 = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"))
        self.time_proj = nn.Linear(time_embedding_size, hidden_size) if add_time else None

    def forward(self, x, c, t, return_attn_weights=False):
        if self.time_proj is not None:
            x = x + self.time_proj(t).unsqueeze(1)
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=return_attn_weights, key_padding_mask=(1 - c).type(torch.bool))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return (x, w) if return_attn_weights else x


class NoTimeModConditionTransformer(nn.Module):
    MODE = "input"            # set by install()
    TIME_EMBEDDING_SIZE = 256

    def __init__(self, nodes_size, hidden_size=64, depth=6, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        te, mode = self.TIME_EMBEDDING_SIZE, self.MODE
        self.nodes_size, self.num_heads, self.mode = nodes_size, num_heads, mode
        self.x_embedder = SharedInputEmbedder(nodes_size, hidden_size)
        self.t_embedder = (TimestepEmbedder(nodes_size, hidden_size, mlp_ratio=mlp_ratio,
                                            frequency_embedding_size=te) if mode != "none" else None)
        self.t_input = nn.Linear(te, hidden_size) if mode == "input" else None
        self.blocks = nn.ModuleList([
            PlainBlock(hidden_size, num_heads, nodes_size, mlp_ratio, te, add_time=(mode == "add"))
            for _ in range(depth)])
        self.norm_final = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)
        # Zero read-out: the network starts at score 0, as with adaLN-Zero.
        self.readout = nn.Parameter(torch.zeros(nodes_size * hidden_size, nodes_size))
        self._init()

    def _init(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        if self.t_embedder is not None:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, c, return_attn_weights=False):
        x = self.x_embedder(x)
        te = self.t_embedder(t) if self.t_embedder is not None else None
        if self.t_input is not None:
            x = x + self.t_input(te).unsqueeze(1)
        weights = []
        for block in self.blocks:
            if return_attn_weights:
                x, w = block(x, c, te, True)
                weights.append(w.mean(0))
            else:
                x = block(x, c, te)
        out = (nn.SiLU()(self.norm_final(x)).flatten(1) @ self.readout).squeeze(-1)
        return (out, torch.stack(weights, 0)) if return_attn_weights else out


def install(mode="input", time_embedding_size=256):
    assert mode in ("input", "add", "none"), mode
    NoTimeModConditionTransformer.MODE = mode
    NoTimeModConditionTransformer.TIME_EMBEDDING_SIZE = int(time_embedding_size)
    module = importlib.import_module("compass.ScoreBasedInferenceModel")
    module.ConditionTransformer = NoTimeModConditionTransformer
