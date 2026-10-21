"""Legacy ConditionTransformer with low-rank per-node time conditioning.

Legacy adaLN: Linear(te -> 6 * nodes * hidden) per block, i.e. a full
time-dependent shift/scale/gate per variable (95% of the parameters, and the
model memorises 1k simulations when trained to the end). Sharing it across
nodes (shared_time_transformer.py) helped Tree/HMM/Gaussian Mixture at 1k but
hurt Two Moons by +0.08, so *some* per-node time dependence matters. Here

    mod(t)[node] = A t_emb + P[node] (B t_emb),   B: te -> rank,  P: nodes x rank x 6H

i.e. the shared DiT modulation plus a rank-``r`` per-node correction, plus the
per-node ID bias of the shared variant. Both output maps are zero-initialised
(adaLN-Zero). Everything else is the legacy trunk.
"""

import importlib

import torch
import torch.nn as nn

from shared_time_transformer import SharedInputEmbedder, SharedTimeConditionTransformer, modulate


class LowRankModulation(nn.Module):
    def __init__(self, time_embedding_size, nodes_size, width, rank):
        super().__init__()
        self.act = nn.SiLU()
        self.shared = nn.Linear(time_embedding_size, width)
        self.down = nn.Linear(time_embedding_size, rank)
        self.per_node = nn.Parameter(torch.zeros(nodes_size, rank, width))
        nn.init.zeros_(self.shared.weight)
        nn.init.zeros_(self.shared.bias)

    def forward(self, t):
        h = self.act(t)
        return self.shared(h).unsqueeze(1) + torch.einsum("br,nrw->bnw", self.down(h), self.per_node)


class LowRankTimeConditionTransformer(SharedTimeConditionTransformer):
    RANK = 8

    def __init__(self, nodes_size, hidden_size=64, depth=6, num_heads=16, mlp_ratio=4.0):
        super().__init__(nodes_size, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        te, r = self.time_embedding_size, self.RANK
        for block in self.blocks:
            block.adaLN_modulation = LowRankModulation(te, nodes_size, 6 * hidden_size, r)
        self.final_layer.adaLN_modulation = LowRankModulation(te, nodes_size, 2 * hidden_size, r)
        self._patch_forward()

    def _patch_forward(self):
        # The shared blocks unsqueeze(1) their (B, 6H) modulation; ours is
        # already (B, nodes, 6H), so give them a forward that does not.
        for block in self.blocks:
            block.forward = _block_forward.__get__(block)
        self.final_layer.forward = _final_forward.__get__(self.final_layer)


def _block_forward(self, x, c, t, return_attn_weights=False):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
    x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
    attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=return_attn_weights,
                                          key_padding_mask=(1 - c).type(torch.bool))
    x = x + gate_msa * attn_output
    x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    return (x, attn_weights) if return_attn_weights else x


def _final_forward(self, x, t):
    shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
    x = nn.SiLU()(modulate(self.norm_final(x), shift, scale))
    return (x.flatten(1) @ self.embedding_params).squeeze(-1)


def install(rank=8, time_embedding_size=256):
    LowRankTimeConditionTransformer.RANK = int(rank)
    SharedTimeConditionTransformer.TIME_EMBEDDING_SIZE = int(time_embedding_size)
    module = importlib.import_module("compass.ScoreBasedInferenceModel")
    module.ConditionTransformer = LowRankTimeConditionTransformer
