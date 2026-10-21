"""Make ``time_embedding_size`` reachable on the LEGACY trunk, without editing src.

Two separate defects block the one capacity lever that should transfer from the
partial-pooling ladder to this directory:

1. ``ScoreBasedInferenceModel.__init__`` pins ``self.time_embedding_size`` to
   ``256 if arch == "legacy"`` (``ScoreBasedInferenceModel.py:136``) and never
   forwards it to ``ConditionTransformer`` at all, so the argument is
   unreachable from a legacy model.

2. Reaching past that does not work either: ``ConditionTransformer.__init__``
   forwards ``time_embedding_size`` to ``t_embedder`` **only**, not to
   ``TransformerBlock`` or ``FinalLayer`` (``ConditionTransformer.py:243-246``),
   both of which keep their own 256 default. The result is a model that builds
   and counts parameters but raises on the first forward pass:
   ``mat1 and mat2 shapes cannot be multiplied (B x te and 256 x 6NH)``.

This module does the rewiring after construction, on an ordinary legacy model,
so nothing under ``src/compass`` changes. It is deliberately a surgical swap of
three module types rather than a subclass: a subclass would have to duplicate
``ConditionTransformer.forward``, and then the experiment would no longer be
testing the library's own forward pass.

Why this is the right lever here rather than the tokenized trunk: the ladder's
reduction worked because 99.3% of the partial-pooling model replicated one
function of diffusion time across 90 *exchangeable* trial scalars. At
``nodes_size = 6`` there is no exchangeability to trade for, and the measured
tokenized arms are far worse (endpoint 0.676 vs 0.097). But the adaLN modulation
is still 94.1% of the legacy model at h16d2, and it is a function of ``t``
alone -- so shrinking its *input* width is the reduction that does apply.
"""

import copy

import torch
import torch.nn as nn

from compass.ConditionTransformer import (ConditionTransformer, FinalLayer,
                                          TimestepEmbedder, TransformerBlock)

DEFAULT_TIME_EMBEDDING = 256


def _reinit_linear(linear, zero=False):
    """Match ``ConditionTransformer.initialize_weights`` for a replaced Linear."""
    if zero:
        nn.init.constant_(linear.weight, 0)
        nn.init.constant_(linear.bias, 0)
    else:
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0)


def rewire_trunk(trunk, time_embedding_size):
    """Rebuild ``trunk``'s time-conditioned layers at a new embedding width.

    Operates in place and returns the trunk. ``time_embedding_size=256`` is a
    no-op in shape but still rebuilds the layers, so callers get identical
    treatment across the sweep and a te=256 arm is a genuine control.
    """
    if not isinstance(trunk, ConditionTransformer):
        raise TypeError(
            f"rewire_trunk expects a ConditionTransformer (the legacy trunk); "
            f"got {type(trunk).__name__}. The tokenized and grouped trunks "
            f"already honour time_embedding_size through the public API."
        )
    te = int(time_embedding_size)
    if te < 2 or te % 2:
        raise ValueError(f"time_embedding_size must be a positive even number; got {te}")

    device = next(trunk.parameters()).device
    n, h = trunk.nodes_size, trunk.final_layer.hidden_size
    mlp_ratio = trunk.t_embedder.mlp[0].out_features // h

    trunk.t_embedder = TimestepEmbedder(
        n, h, mlp_ratio=mlp_ratio, frequency_embedding_size=te,
    )
    nn.init.normal_(trunk.t_embedder.mlp[0].weight, std=0.02)
    nn.init.constant_(trunk.t_embedder.mlp[0].bias, 0)
    nn.init.normal_(trunk.t_embedder.mlp[2].weight, std=0.02)
    nn.init.constant_(trunk.t_embedder.mlp[2].bias, 0)

    for block in trunk.blocks:
        block.adaLN_modulation[1] = nn.Linear(te, 6 * n * h, bias=True)
        _reinit_linear(block.adaLN_modulation[1], zero=True)
    trunk.final_layer.adaLN_modulation[1] = nn.Linear(te, 2 * n * h, bias=True)
    _reinit_linear(trunk.final_layer.adaLN_modulation[1], zero=True)

    trunk.time_embedding_size = te
    trunk.to(device)
    return trunk


def rewire(model, time_embedding_size):
    """Rewire a legacy ``ScoreBasedInferenceModel`` in place and return it.

    ``model.time_embedding_size`` is set too, which is what makes ``SBIm.save``
    record the width in the checkpoint payload (it writes the field only when it
    differs from 256). ``SBIm.load`` cannot use it -- the constructor discards
    it for legacy -- so reload through :func:`load_rewired`.
    """
    rewire_trunk(model.model, time_embedding_size)
    model.time_embedding_size = int(time_embedding_size)
    return model


def width_of(state_dict):
    """Recover the embedding width from the weights themselves.

    ``SBIm.architecture_payload`` returns ``{}`` for a legacy model *by design*
    -- a legacy checkpoint must stay byte-identical to what it always was, so
    ``time_embedding_size`` is never recorded there however it is set. A rewired
    legacy checkpoint is therefore not self-describing through the payload, and
    the width has to come from the one place it is unambiguous: the input
    dimension of a block's adaLN modulation.
    """
    key = "blocks.0.adaLN_modulation.1.weight"
    if key not in state_dict:
        raise KeyError(f"{key} absent; this is not a legacy ConditionTransformer state dict")
    return int(state_dict[key].shape[1])


def load_rewired(path, build_fn, device="cpu"):
    """Load a rewired legacy checkpoint. ``build_fn(te, device) -> SBIm``.

    ``SBIm.load`` reconstructs the network before loading the state dict, and
    for a legacy checkpoint it reconstructs at te=256 unconditionally, so it
    raises on a rewired checkpoint. This reads the width off the weights and
    rebuilds at it first.
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    te = width_of(checkpoint["model_state_dict"])
    model = build_fn(te, device)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.model.to(device)
    return model, te


def parameter_report(trunk):
    """Total and the adaLN-modulation share, the quantity the sweep is about."""
    total = sum(p.numel() for p in trunk.parameters())
    adaln = sum(p.numel() for name, p in trunk.named_parameters()
                if "adaLN" in name)
    t_emb = sum(p.numel() for p in trunk.t_embedder.parameters())
    return {
        "total": total,
        "adaLN": adaln,
        "t_embedder": t_emb,
        "time_only": adaln + t_emb,
        "data_dependent": total - adaln - t_emb,
        "adaLN_share": adaln / total,
        "data_dependent_share": (total - adaln - t_emb) / total,
    }
