"""Stage 0 regression: ``arch="legacy"`` must reproduce its frozen outputs.

Two fixtures, both produced by ``Partial_Pooling/freeze_stage0.py``:

* ``legacy_arch_regression.pt`` -- freshly constructed, fixed-seed networks.
  Depends on nothing but the source tree, so this is the test that has to keep
  passing on a clean checkout for the whole IID-embedding ladder. It pins the
  weight initialization *and* the forward pass, so a reordered ``nn.Module``
  attribute (which shifts the RNG stream) fails it just as loudly as a changed
  formula.
* ``legacy_checkpoint_regression.pt`` -- the trained 5,957,984-parameter
  ``small`` checkpoint through all four network entry points (Trainer's raw
  score, Sampler, PFODE, MultiObsSampler). Skipped when the checkpoint or the
  benchmark data are absent.

Comparison is bit-for-bit by default. ``COMPASS_REGRESSION_TOL`` relaxes it to a
max-absolute-difference bound, for the case of a genuinely different BLAS or
torch build rather than a changed model.
"""

import os
from pathlib import Path

import pytest
import torch

# Must match Partial_Pooling/freeze_stage0.py: BLAS reduction order, and hence
# the last bits of every matmul, depends on the thread count.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

from compass import ScoreBasedInferenceModel

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARCH_FIXTURE = FIXTURES / "legacy_arch_regression.pt"
CHECKPOINT_FIXTURE = FIXTURES / "legacy_checkpoint_regression.pt"
TOLERANCE = float(os.environ.get("COMPASS_REGRESSION_TOL", "0.0"))


def assert_identical(actual, expected, what):
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    difference = (actual.double() - expected.double()).abs().max().item()
    if TOLERANCE == 0.0:
        assert torch.equal(actual, expected), (
            f"{what}: legacy output changed (max |delta| = {difference:.3e}). "
            "Stage 0 forbids any change to arch='legacy'."
        )
    else:
        assert difference <= TOLERANCE, (
            f"{what}: max |delta| = {difference:.3e} exceeds {TOLERANCE:.3e}."
        )


def load(path):
    if not path.exists():
        pytest.skip(f"Missing fixture {path}; run Partial_Pooling/freeze_stage0.py.")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def arch_fixture():
    return load(ARCH_FIXTURE)


@pytest.fixture(scope="module")
def checkpoint_fixture():
    return load(CHECKPOINT_FIXTURE)


def case_names():
    if not ARCH_FIXTURE.exists():
        return ["missing"]
    return sorted(torch.load(ARCH_FIXTURE, map_location="cpu", weights_only=False)["cases"])


@pytest.mark.parametrize("name", case_names())
def test_legacy_construction_and_forward(arch_fixture, name):
    entry = arch_fixture["cases"][name]
    torch.manual_seed(arch_fixture["arch_seed"])
    model = ScoreBasedInferenceModel(device="cpu", **entry["config"])
    model.model.eval()

    parameters = sum(p.numel() for p in model.model.parameters())
    assert parameters == entry["parameters"], (
        f"{name}: {parameters:,} parameters, frozen value is {entry['parameters']:,}."
    )

    with torch.no_grad():
        raw = model.model(x=entry["x"], t=entry["t"], c=entry["condition_mask"])
        score = model.output_scale_function(entry["t"], raw)
    assert_identical(raw, entry["raw_output"], f"{name} raw output")
    assert_identical(score, entry["score"], f"{name} score")


def test_legacy_is_the_default_signature():
    """``nodes_size=`` alone must keep meaning the legacy architecture."""
    model = ScoreBasedInferenceModel(nodes_size=12, hidden_size=4, depth=1,
                                     num_heads=1, mlp_ratio=2, device="cpu")
    assert getattr(model, "arch", "legacy") == "legacy"


def test_checkpoint_scores(checkpoint_fixture):
    entry = checkpoint_fixture
    checkpoint = Path(__file__).resolve().parent.parent / entry["checkpoint"]
    if not checkpoint.exists():
        pytest.skip(f"Missing checkpoint {checkpoint}.")
    model = ScoreBasedInferenceModel.load(str(checkpoint), device="cpu")
    model.model.eval()
    assert sum(p.numel() for p in model.model.parameters()) == entry["parameters"]
    with torch.no_grad():
        raw = model.model(x=entry["x"], t=entry["t"], c=entry["condition_mask"])
        score = model.output_scale_function(entry["t"], raw)
    assert_identical(raw, entry["raw_output"], "checkpoint raw output")
    assert_identical(score, entry["score"], "checkpoint score")


def test_checkpoint_sampler_pfode_and_composition(checkpoint_fixture):
    entry = checkpoint_fixture
    checkpoint = Path(__file__).resolve().parent.parent / entry["checkpoint"]
    if not checkpoint.exists():
        pytest.skip(f"Missing checkpoint {checkpoint}.")
    model = ScoreBasedInferenceModel.load(str(checkpoint), device="cpu")
    model.model.eval()
    observations = entry["observations"]
    nodes_size = model.nodes_size

    torch.manual_seed(entry["sample_seed"])
    # churn_eta=0.0 pins the *legacy* deterministic predictor these fixtures
    # were frozen under. The library default is now the churned predictor
    # (compass.churn), and this fixture exists to detect an unintended change to
    # the architecture -- not to restate the sampler's current default.
    draws = model.sample(
        x=observations, timesteps=20, num_samples=16, method="dpm",
        churn_eta=0.0, corrector_steps=5, final_corrector_steps=3,
        device="cpu", verbose=False, capture_attention=False,
    )
    assert_identical(draws, entry["posterior_draws"], "posterior draws")

    condition_mask = torch.cat([torch.zeros(10), torch.ones(nodes_size - 10)])
    joint = torch.cat([torch.zeros(observations.shape[0], 10), observations], dim=1)
    torch.manual_seed(entry["sample_seed"])
    log_prob = model.log_prob(
        data=joint, condition_mask=condition_mask, timesteps=20,
        divergence="exact", device="cpu", verbose=False,
    )
    assert_identical(log_prob, entry["log_prob"], "PF-ODE log-probability")

    # The fixture was frozen under the previous composition arithmetic; "cholesky"
    # reproduces it bit for bit (the default "elimination" agrees to float rounding).
    # Like churn_eta=0.0 below, this pins the sampler so that the fixture detects
    # changes to the architecture.
    model.multi_obs_sampler.set_schur_mode("cholesky")
    model.multi_obs_sampler.set_eigh_precision(torch.float64, 2048)
    torch.manual_seed(entry["sample_seed"])
    composed = model.sample(
        x=observations, timesteps=20, num_samples=16, method="dpm", order=2,
        multi_obs_inference=True, hierarchy=list(range(7)),
        correction="gauss_jacobian", jacobian_refresh=20,
        # Same reason as above: this fixture was frozen under the stock
        # predictor-only sampler, so it pins churn off explicitly.
        churn_eta=0.0, corrector_steps=0, final_corrector_steps=0,
        device="cpu", verbose=False,
    )
    assert_identical(composed, entry["composed_draws"], "gauss_jacobian composed draws")
