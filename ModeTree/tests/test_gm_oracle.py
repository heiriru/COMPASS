"""The oracle bridge experiment is only worth anything if its algebra is exact.

Every claim the report makes rests on four identities, all pinned here:

* the clean factorisation ``p(g, l_1..l_n | X) = p(g)^{1-n} prod_i p(g, l_i | x_i)``
  holds up to a constant for this benchmark;
* every composition rule reduces to the exact clean tall score as ``lambda -> 0``
  (this is the section-3 endpoint requirement for the F-NPSE bridge, and it
  holds for the damped and precision-weighted rules too);
* the exact diffused tall score, the row scores and the row Hessians agree with
  autograd on the corresponding log densities;
* the exact prior Hessian is the mixture's, not ``-1/(tau_g^2 + lambda^2)``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ModeTree"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Gaussian_mixtures_compositional_score_modelling"))
sys.path.insert(0, str(ROOT / "ModeTree" / "experiments"))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import numpy as np   # noqa: E402
import pytest        # noqa: E402
import torch         # noqa: E402

import datasets      # noqa: E402
import model as M    # noqa: E402

from gm_bridges import ComposedField   # noqa: E402
from gm_oracle import GMOracle         # noqa: E402


@pytest.fixture(scope="module")
def oracle():
    rng = np.random.default_rng(0)
    dataset = datasets.make_dataset("asymmetric", 8, rng, M.DEFAULT,
                                    branch_mass=0.75)
    return GMOracle(M.DEFAULT, dataset["x"], sigma=15.0)


@pytest.fixture(scope="module")
def states(oracle):
    generator = torch.Generator().manual_seed(3)
    return torch.randn(6, oracle.dim, dtype=torch.float64, generator=generator)


def test_clean_factorisation_is_exact(oracle, states):
    rows = oracle.rows_state(states)
    composed = ((1 - oracle.n) * oracle.prior.log_prob(states[..., 0], 0.0)
                + oracle.rows.log_prob(rows, 0.0).sum(-1))
    delta = states.unsqueeze(-2) - oracle.tall.means
    solved = torch.linalg.solve(oracle.tall.cov, delta.unsqueeze(-1)).squeeze(-1)
    exact = torch.logsumexp(
        oracle.tall.log_weights - 0.5 * (delta * solved).sum(-1), dim=-1
    )
    gap = composed - exact
    assert float(gap.max() - gap.min()) < 1e-10


def test_diffused_tall_score_matches_autograd(oracle, states):
    lam = 0.37
    live = states.clone().requires_grad_(True)
    cov = oracle.tall.cov + lam ** 2 * torch.eye(oracle.dim, dtype=torch.float64)
    delta = live.unsqueeze(-2) - oracle.tall.means
    solved = torch.linalg.solve(cov, delta.unsqueeze(-1)).squeeze(-1)
    value = torch.logsumexp(
        oracle.tall.log_weights - 0.5 * (delta * solved).sum(-1), dim=-1
    ).sum()
    gradient, = torch.autograd.grad(value, live)
    assert torch.allclose(gradient, oracle.tall.score(states, lam), atol=1e-10)


def test_row_scores_and_hessians_match_autograd(oracle, states):
    lam = 0.37
    live = states.clone().requires_grad_(True)
    value = oracle.rows.log_prob(oracle.rows_state(live), lam).sum()
    gradient, = torch.autograd.grad(value, live)
    scores = oracle.rows.scores(oracle.rows_state(states), lam)
    assert torch.allclose(gradient[..., 0], scores[..., 0].sum(-1), atol=1e-10)
    assert torch.allclose(gradient[..., 1:], scores[..., 1], atol=1e-10)

    point = oracle.rows_state(states)[0, 0].clone()
    jacobian = torch.autograd.functional.jacobian(
        lambda z: oracle.rows.scores(
            z.reshape(1, 1, 2).expand(1, oracle.n, 2).clone(), lam)[0, 0],
        point,
    )
    hessian = oracle.rows.hessians(oracle.rows_state(states), lam)[0, 0]
    assert torch.allclose(hessian, jacobian, atol=1e-10)


def test_prior_hessian_is_the_mixture_hessian(oracle, states):
    lam = 0.37
    g = states[..., 0].clone().requires_grad_(True)
    value = oracle.prior.score(g, lam).sum()
    gradient, = torch.autograd.grad(value, g)
    exact = oracle.prior.hessian(states[..., 0], lam)
    assert torch.allclose(gradient, exact, atol=1e-10)
    # and it is genuinely not the Gaussian answer anywhere useful
    gaussian = -1.0 / (M.DEFAULT.tau_g ** 2 + lam ** 2)
    assert float((exact - gaussian).abs().max()) > 1.0


@pytest.mark.parametrize(
    "rule,options",
    [("naive", {}), ("fnpe", {"fnpe_prior": "clean"}),
     ("fnpe", {"fnpe_prior": "diffused"}), ("damped", {"damping_d1": 0.2}),
     ("gauss", {}), ("jac", {})],
)
def test_every_rule_reduces_to_the_exact_clean_score(oracle, states, rule, options):
    """The section-3 endpoint check, applied to all six rules.

    ``t = 0`` puts ``a(0) = 1`` and ``d(0) = 1``, so each bridge's target is the
    exact tall posterior; the tolerance is loose for the precision-weighted
    rules only because they are evaluated at a small but non-zero lambda, where
    the Tweedie inversion is ill-conditioned rather than wrong.
    """
    lam = 1e-4 if rule in ("gauss", "jac", "damped") else 1e-12
    field = ComposedField(oracle, rule, **options)
    composed = field(states, t=0.0, lam=lam)
    exact = oracle.tall.score(states, lam)
    assert float((composed - exact).abs().max()) < 1e-4
