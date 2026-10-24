"""COMPASS wiring for the GCE models: node layout, normalisation, network, training,
composed (many-star) inference and Bayes factors for the nested IMF hypotheses.

Node layout of one star (the "single-observation" network)::

    theta = [globals (G) | locals (L)]      x = ObsProfile.x_names

and the hierarchy composed over n stars is ``range(G)``. The recipe is the
partial-pooling one that worked there: tokenized shared trunk + hierarchical
latent attention (g attends g; l attends g and l), six-conditional mask menu
mixed with Bernoulli, ``time_sampling="log_sigma"``, globals trained under a
1.5x-wide prior that ``set_training_prior`` divides back out, and the
final-epoch weights (not the best-validation checkpoint) for inference.
gauss_jacobian_FD with ``jacobian_refresh=9`` composes the stars.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import models as model_registry
from .models import HierarchicalModel, birth_time_from_u
from .observation import PROFILES, ObsProfile, make_observations

RECIPE = dict(hidden_size=96, depth=6, heads=8, time_embedding_size=64, mlp_ratio=4,
              sde_type="vesde", sigma=25.0, attention="hierarchical",
              posterior_weight=1.0, bernoulli_fraction=0.25, time_sampling="log_sigma",
              batch_size=256, max_epochs=500, patience=100, lr=1e-3)
# For wiring checks only: tiny, two epochs, CPU-friendly.
SMOKE_RECIPE = dict(RECIPE, hidden_size=32, depth=2, heads=4, time_embedding_size=32,
                    batch_size=64, max_epochs=2, patience=2)
SAMPLER = dict(correction="gauss_jacobian_FD", jacobian_refresh=9, timesteps=50, order=2,
               method="dpm", corrector_steps=0, final_corrector_steps=0, terminal_corrector_steps=0)


@dataclass(frozen=True)
class Layout:
    model: HierarchicalModel
    profile: ObsProfile

    @property
    def n_global(self):
        return len(self.model.global_names)

    @property
    def n_local(self):
        return len(self.model.local_names)

    @property
    def theta_size(self):
        return self.n_global + self.n_local

    @property
    def roles(self):
        g, l = self.n_global, self.n_local
        return {"g": list(range(g)), "l": list(range(g, g + l)),
                "x": list(range(g + l, g + l + self.profile.x_size))}

    def describe(self):
        return {"model": self.model.name, "profile": self.profile.name,
                "theta": list(self.model.theta_names), "x": list(self.profile.x_names)}


class Normalizer:
    """Per-coordinate affine map fitted on the training split only."""

    def __init__(self, theta_mean, theta_scale, x_mean, x_scale):
        self.theta_mean, self.theta_scale = np.asarray(theta_mean, float), np.asarray(theta_scale, float)
        self.x_mean, self.x_scale = np.asarray(x_mean, float), np.asarray(x_scale, float)

    @classmethod
    def fit(cls, theta, x):
        return cls(theta.mean(0), theta.std(0) + 1e-12, x.mean(0), x.std(0) + 1e-12)

    def theta(self, values):
        return (np.asarray(values) - self.theta_mean) / self.theta_scale

    def theta_inverse(self, values):
        return np.asarray(values) * self.theta_scale + self.theta_mean

    def x(self, values):
        return (np.asarray(values) - self.x_mean) / self.x_scale

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("theta_mean", "theta_scale", "x_mean", "x_scale")}

    @classmethod
    def from_dict(cls, d):
        return cls(d["theta_mean"], d["theta_scale"], d["x_mean"], d["x_scale"])


def training_arrays(bank, layout: Layout, seed=0):
    """(theta, x) for a bank under the layout's observation profile (fresh noise per seed)."""
    rng = np.random.default_rng(seed)
    theta = np.concatenate([bank["globals"], bank["locals"]], axis=1)
    u = bank["locals"][:, layout.model.local_names.index("u_birth")]
    x = make_observations(bank["abundances"], birth_time_from_u(u), layout.profile, rng)
    return theta.astype(np.float32), x.astype(np.float32)


def build_network(layout: Layout, recipe=RECIPE, device="cpu", seed=7):
    import torch
    from compass import Atomic, ScoreBasedInferenceModel, six_conditional_menu
    from compass.experimental.latent_attention import enable_latent_attention

    torch.manual_seed(seed)
    model = ScoreBasedInferenceModel(
        theta_size=layout.theta_size, x_spec=[Atomic("x", layout.profile.x_size)], arch="tokenized",
        trunk="shared", time_embedding_size=recipe["time_embedding_size"], sde_type=recipe["sde_type"],
        sigma=recipe["sigma"], hidden_size=recipe["hidden_size"], depth=recipe["depth"],
        num_heads=recipe["heads"], mlp_ratio=recipe["mlp_ratio"], device=device)
    if recipe["attention"] != "none":
        enable_latent_attention(model, recipe["attention"], layout.roles)
    menu = six_conditional_menu(model.schema, global_size=layout.n_global, local_size=layout.n_local,
                                weights=[recipe["posterior_weight"]] * 3 + [1.0] * 3,
                                bernoulli_fraction=recipe["bernoulli_fraction"])
    return model, menu


def train(layout: Layout, train_bank, validation_bank, out_dir, name, recipe=RECIPE, device="cuda",
          training_scale=1.5, seed=7):
    """Train one single-star network and write everything inference needs next to it."""
    import torch
    from compass.experimental.latent_attention import save_latent_attention

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    theta, x = training_arrays(train_bank, layout, seed)
    theta_val, x_val = training_arrays(validation_bank, layout, seed + 1)
    normalizer = Normalizer.fit(theta, x)
    model, menu = build_network(layout, recipe, device, seed)
    save_latent_attention(model, out_dir)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    torch.manual_seed(seed)
    model.train(t(normalizer.theta(theta)), t(normalizer.x(x)), t(normalizer.theta(theta_val)),
                t(normalizer.x(x_val)), batch_size=recipe["batch_size"], max_epochs=recipe["max_epochs"],
                lr=recipe["lr"], device=device, verbose=True, path=str(out_dir), name=name,
                early_stopping_patience=recipe["patience"], mask_sampler=menu,
                time_sampling=recipe["time_sampling"])
    sidecar = {"layout": layout.describe(), "normalizer": normalizer.to_dict(), "recipe": recipe,
               "training_scale": training_scale, "seed": seed, "name": name,
               "local_moments": {"mean": normalizer.theta(theta)[:, layout.n_global:].mean(0).tolist(),
                                 "std": normalizer.theta(theta)[:, layout.n_global:].std(0).tolist()},
               "train_rows": int(len(theta)), "validation_rows": int(len(theta_val)),
               "bank": train_bank.get("manifest", {})}
    (out_dir / f"{name}.gce.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return model, normalizer


def load(out_dir, name, final=True, device="cuda"):
    """The network, its layout, normaliser and sidecar. ``final`` = final-epoch weights."""
    from compass import ScoreBasedInferenceModel
    from compass.experimental.latent_attention import load_latent_attention

    out_dir = Path(out_dir)
    sidecar = json.loads((out_dir / f"{name}.gce.json").read_text())
    path = out_dir / (f"{name}.pt" if final else f"{name}_checkpoint.pt")
    model = ScoreBasedInferenceModel.load(str(path), device=device, arch="tokenized")
    load_latent_attention(model, out_dir, required=sidecar["recipe"]["attention"] != "none")
    model.model.eval()
    layout = Layout(model_registry.get(sidecar["layout"]["model"]), PROFILES[sidecar["layout"]["profile"]])
    if list(layout.profile.x_names) != sidecar["layout"]["x"]:
        raise RuntimeError("observation profile changed since training: "
                           f"{layout.profile.x_names} vs {sidecar['layout']['x']}")
    return model, layout, Normalizer.from_dict(sidecar["normalizer"]), sidecar


def compose(model, layout: Layout, normalizer: Normalizer, x_obs, sidecar, draws=200, draw_batch=50,
            device="cuda", seed=0, **sampler):
    """Joint posterior of the globals and every star's locals given ``x_obs`` [n, x_size]
    (physical units, the profile's layout). Returns globals [draws, G] and locals
    [n, draws, L] in inference coordinates."""
    import torch

    settings = dict(SAMPLER, **sampler)
    g = layout.n_global
    mean, sd = layout.model.global_prior()
    prior = ((mean - normalizer.theta_mean[:g]) / normalizer.theta_scale[:g], sd / normalizer.theta_scale[:g])
    scale = sidecar["training_scale"]
    if scale != 1.0:
        model.multi_obs_sampler.set_training_prior((prior[0], prior[1] * scale))
    else:
        model.multi_obs_sampler.set_training_prior(None)
    local_prior = (torch.tensor(sidecar["local_moments"]["mean"]), torch.tensor(sidecar["local_moments"]["std"]))
    x = torch.as_tensor(normalizer.x(x_obs), dtype=torch.float32, device=device)
    mask = torch.cat([torch.zeros(layout.theta_size), torch.ones(layout.profile.x_size)]).to(device)
    torch.manual_seed(seed)
    chunks = []
    for start in range(0, draws, draw_batch):
        count = min(draw_batch, draws - start)
        chunk = model.sample(x=x, condition_mask=mask, num_samples=count, multi_obs_inference=True,
                             hierarchy=list(range(g)),
                             prior=(torch.as_tensor(prior[0], dtype=torch.float32),
                                    torch.as_tensor(prior[1], dtype=torch.float32)),
                             local_prior=local_prior, device=device, verbose=False,
                             capture_attention=False, **settings).cpu().numpy()
        chunks.append(chunk)
    normalized = np.concatenate(chunks, axis=1)          # [n, draws, theta]
    if not np.isfinite(normalized).all():
        raise RuntimeError("composed sampler returned non-finite draws")
    if np.abs(normalized[:, :, :g] - normalized[:1, :, :g]).max() > 1e-5:
        raise RuntimeError("globals are not synchronised across stars")
    physical = normalizer.theta_inverse(normalized)
    return {"globals": physical[0, :, :g], "locals": physical[:, :, g:],
            "global_names": layout.model.global_names, "local_names": layout.model.local_names,
            "solver_stats": dict(model.multi_obs_sampler.solver_stats or {})}


# ---- nested IMF hypotheses ----------------------------------------------------------------
def savage_dickey(samples, prior_mean, prior_sd, point=0.0):
    """log BF(null : alternative) for a point null ``theta = point`` nested in a model with a
    Gaussian prior on theta: log p(point | X) - log p(point), the posterior density by KDE.
    ``reliable`` is False when the point is outside the bulk of the draws, where a KDE cannot
    resolve the density; then report a bound, not a number."""
    from scipy.stats import gaussian_kde, norm
    samples = np.asarray(samples, dtype=float)
    density = float(gaussian_kde(samples)(point)[0])
    z = abs(point - samples.mean()) / samples.std()
    log_prior = float(norm.logpdf(point, prior_mean, prior_sd))
    return {"log_bf_null": float(np.log(max(density, 1e-300)) - log_prior), "posterior_density": density,
            "z_of_point": float(z), "reliable": bool(z < 3.0)}


def interval_null_bayes_factor(samples, prior_mean, prior_sd, interval):
    """log BF for ``theta in interval`` against the encompassing model:
    log P(I | X) - log P(I) (Morey & Rouder 2011). Used for sigma_alpha ~ 0, where a
    point null sits on the boundary. ``None`` when no draw falls in I (then only an upper
    bound log(1 / draws) - log P(I) is available)."""
    from scipy.stats import norm
    samples = np.asarray(samples, dtype=float)
    low, high = interval
    prior_mass = float(norm.cdf(high, prior_mean, prior_sd) - norm.cdf(low, prior_mean, prior_sd))
    inside = int(((samples >= low) & (samples <= high)).sum())
    upper = float(np.log(1.0 / len(samples)) - np.log(prior_mass))
    if inside == 0:
        return {"log_bf_null": None, "upper_bound": upper, "prior_mass": prior_mass, "inside": 0}
    p = inside / len(samples)
    return {"log_bf_null": float(np.log(p) - np.log(prior_mass)), "prior_mass": prior_mass, "inside": inside,
            "mc_se": float(np.sqrt((1 - p) / (p * len(samples))))}


def nested_bayes_factor(model: HierarchicalModel, composed):
    """Bayes factor of the universal IMF against ``model`` from one composed posterior."""
    out = {}
    for name, null in model.nested_null.items():
        column = model.global_names.index(name)
        prior = dict(model.globals)[name]
        draws = composed["globals"][:, column]
        if isinstance(null, tuple):
            out[name] = interval_null_bayes_factor(draws, prior.mean, prior.sd, null)
        else:
            out[name] = savage_dickey(draws, prior.mean, prior.sd, null)
    return out
