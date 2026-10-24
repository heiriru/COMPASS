"""Compact state of the multi-observation sampler: one global block per draw.

The hierarchical sampler used to carry its state as one array of shape
``(N, M, F)``: ``N`` observation rows, ``M`` posterior draws and the ``F`` node
features of the network input. The ``d_g`` global coordinates were stored ``N``
times (synchronised across rows) and the observed data ``M`` times. Every
solver update, churn kick and noise draw then worked on ``N M F`` numbers,
although only ``M d_g + N M d_l`` of them are free.

:class:`HierarchicalState` keeps exactly those:

    g : (M, d_g)        the global coordinates, once per draw
    l : (N, M, d_l)     the local coordinates of every observation row

and :class:`HierarchicalLayout` holds everything that does not change along the
trajectory (the observed node values, the condition mask and the feature
indices). The layout rebuilds the ``(N, M, F)`` row array only for a network
evaluation (:meth:`HierarchicalLayout.assemble`), with exactly the values the
synchronised array held, so the network input, the per-row Jacobians and the
composition are unchanged.

Arithmetic. A state behaves like a tensor under the elementwise operations the
solvers use (``+``, ``-``, ``*``, ``/``, unary ``-``, ``torch.sqrt``,
``torch.maximum``, ...): the operation is applied to ``g`` and ``l``
separately, with ordinary broadcasting of tensor or scalar operands, and a
state operand is matched part by part. ``a * state + b * score`` is therefore
the update ``G' = a G + b u^g``, ``L' = a L + b u^l`` of the hierarchical
solver, computed elementwise with the same floating-point operations as on the
synchronised array.

Noise. :meth:`HierarchicalLayout.noise` gives one global vector per draw and
independent local noise per row, the covariance of the hierarchical forward
diffusion. ``"legacy"`` (the sampler's default) consumes the random stream exactly
as the synchronised sampler did (a full ``(N, M, F)`` draw and a ``(1, M, F)``
shared draw) and slices it, so a seeded run reproduces the old trajectories bit
for bit; ``"compact"`` draws only the numbers that are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

NOISE_STREAMS = ("compact", "legacy")


def _part(value, name):
    """Replace every state in ``value`` (possibly nested in lists/tuples) by its part."""
    if isinstance(value, HierarchicalState):
        return getattr(value, name)
    if isinstance(value, (list, tuple)):
        return type(value)(_part(item, name) for item in value)
    return value


def _elementwise(func, args, kwargs):
    g = func(*_part(args, "g"), **{k: _part(v, "g") for k, v in kwargs.items()})
    l = func(*_part(args, "l"), **{k: _part(v, "l") for k, v in kwargs.items()})
    return HierarchicalState(g=g, l=l)


@dataclass
class HierarchicalState:
    """Global coordinates once per draw, local coordinates once per row.

    g : (M, d_g),  l : (N, M, d_l).  Holds sampler coordinates or, with the same
    layout, composed scores (``g = u^g``, ``l = u^l``).
    """

    g: torch.Tensor
    l: torch.Tensor

    # Elementwise torch functions (torch.sqrt, torch.maximum, torch.full_like, ...)
    # act on the two parts separately.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return _elementwise(func, args, kwargs or {})

    def __add__(self, other):
        return _elementwise(torch.add, (self, other), {})

    def __radd__(self, other):
        return _elementwise(torch.add, (other, self), {})

    def __sub__(self, other):
        return _elementwise(torch.sub, (self, other), {})

    def __rsub__(self, other):
        return _elementwise(torch.sub, (other, self), {})

    def __mul__(self, other):
        return _elementwise(torch.mul, (self, other), {})

    def __rmul__(self, other):
        return _elementwise(torch.mul, (other, self), {})

    def __truediv__(self, other):
        return _elementwise(torch.div, (self, other), {})

    def __neg__(self):
        return HierarchicalState(g=-self.g, l=-self.l)

    def abs(self):
        return HierarchicalState(g=self.g.abs(), l=self.l.abs())

    def clone(self):
        return HierarchicalState(g=self.g.clone(), l=self.l.clone())

    def detach(self):
        return HierarchicalState(g=self.g.detach(), l=self.l.detach())

    def max(self):
        """Largest entry over both parts (a 0-d tensor)."""
        parts = [p.max() for p in (self.g, self.l) if p.numel()]
        return torch.stack(parts).max()

    def all(self):
        """True iff every entry of both parts is (a 0-d bool tensor)."""
        return torch.stack([self.g.all(), self.l.all()]).all()

    def all_finite(self):
        return bool(torch.isfinite(self.g).all()) and bool(torch.isfinite(self.l).all())

    @property
    def dtype(self):
        return self.g.dtype

    @property
    def device(self):
        return self.g.device

    def numel(self):
        return self.g.numel() + self.l.numel()


class HierarchicalLayout:
    """Static part of the sampler state and the map between both layouts.

    ``frame`` is the ``(N, M, F)`` node array with the observed values in place
    and zeros on the latent coordinates (``_prepare_data``'s output; it may be an
    expanded view of an ``(N, 1, F)`` array, which costs no memory), and
    ``condition_mask`` the matching mask. ``hierarchy`` lists the global
    features, ``local`` the row-local latent ones; every other feature is
    observed in every row.
    """

    def __init__(self, frame, condition_mask, hierarchy, local):
        self.frame = frame
        self.condition_mask = condition_mask
        self.hierarchy = list(hierarchy)
        self.local = list(local)
        self.rows, self.samples, self.features = frame.shape
        device = frame.device
        self.h_index = torch.as_tensor(self.hierarchy, dtype=torch.long, device=device)
        self.l_index = torch.as_tensor(self.local, dtype=torch.long, device=device)

    @staticmethod
    def supports(condition_mask, hierarchy, local):
        """The compact layout needs every latent feature latent in every row and
        every draw (a feature observed in only some rows has no fixed place)."""
        mask = condition_mask.reshape(-1, condition_mask.shape[-1])
        latent = sorted(list(hierarchy) + list(local))
        observed = [i for i in range(mask.shape[-1]) if i not in set(latent)]
        return bool((mask[:, latent] == 0).all()) and bool((mask[:, observed] == 1).all())

    # ------------------------------------------------------------- conversions
    def split(self, rows):
        """(N, M, F) synchronised rows -> state (globals read from row 0)."""
        return HierarchicalState(g=rows[0][:, self.h_index], l=rows[:, :, self.l_index])

    def assemble(self, state):
        """State -> the (N, M, F) node array the network and the composition read.

        A real allocation of N M F numbers that lives for one score evaluation;
        the global block is broadcast into every row.
        """
        rows = torch.empty(self.rows, self.samples, self.features,
                           dtype=self.frame.dtype, device=self.frame.device)
        rows.copy_(self.frame)
        rows[:, :, self.h_index] = state.g.unsqueeze(0).to(rows.dtype)
        rows[:, :, self.l_index] = state.l.to(rows.dtype)
        return rows

    def assemble_latent(self, state):
        """Like :meth:`assemble` with zeros on the observed coordinates (for
        increments and scores, e.g. in a saved trajectory)."""
        rows = torch.zeros(self.rows, self.samples, self.features,
                           dtype=self.frame.dtype, device=self.frame.device)
        rows[:, :, self.h_index] = state.g.unsqueeze(0).to(rows.dtype)
        rows[:, :, self.l_index] = state.l.to(rows.dtype)
        return rows

    def zeros(self):
        return HierarchicalState(
            g=torch.zeros(self.samples, len(self.hierarchy), dtype=self.frame.dtype, device=self.frame.device),
            l=torch.zeros(self.rows, self.samples, len(self.local), dtype=self.frame.dtype,
                          device=self.frame.device))

    def feature_values(self, per_feature):
        """A per-feature (F,) tensor as a state of broadcastable (d,) vectors."""
        return HierarchicalState(g=per_feature[self.h_index], l=per_feature[self.l_index])

    # ------------------------------------------------------------------- noise
    def noise(self, stream="compact"):
        """Standard normal noise in the compact layout.

        "compact": one global vector per draw and independent local noise per
        row and draw. "legacy": the synchronised sampler's draws (``randn`` of the
        full array, then ``randn`` of one shared row) sliced to the same values,
        so seeded runs reproduce the old trajectories exactly.
        """
        dtype, device = self.frame.dtype, self.frame.device
        if stream == "legacy":
            full = torch.randn(self.rows, self.samples, self.features, dtype=dtype, device=device)
            shared = torch.randn(1, self.samples, self.features, device=device)
            return HierarchicalState(g=shared[0][:, self.h_index].to(dtype), l=full[:, :, self.l_index])
        if stream != "compact":
            raise ValueError(f"noise stream must be one of {NOISE_STREAMS}, got {stream!r}.")
        return HierarchicalState(
            g=torch.randn(self.samples, len(self.hierarchy), dtype=dtype, device=device),
            l=torch.randn(self.rows, self.samples, len(self.local), dtype=dtype, device=device))
