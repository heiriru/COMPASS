"""Regression: ``arch="tokenized"`` must reproduce its frozen outputs.

``tests/test_legacy_arch_regression.py`` pins ``arch="legacy"``, which the ladder
left behind. Every result in ``docs/ladder_progress.md`` and
``docs/followup_results.md`` rests on the tokenized architecture instead, and
until this file existed nothing pinned it. The Group/Sequence work edits
``schema.py`` and ``TokenizedTransformer.py`` directly, so this is the guard that
says whether an edit meant to add a new block type quietly changed the existing
ones.

The fixture (``Partial_Pooling/freeze_tokenized.py``) has two independent halves:

* ``cases`` -- freshly constructed fixed-seed networks over four schemas plus the
  463,364-parameter Stage-4 winner. Depends on nothing but the source tree, so it
  fails on a clean checkout. It pins weight initialization *and* the forward
  pass, so a reordered ``nn.Module`` attribute (which shifts the RNG stream)
  fails it as loudly as a changed formula.
* ``checkpoint`` -- the trained ``mask6mix`` arm through all four network entry
  points. Skipped when the checkpoint or the benchmark data are absent.

Comparison is bit-for-bit by default; ``COMPASS_REGRESSION_TOL`` relaxes it for a
genuinely different BLAS or torch build.
"""

import os
from pathlib import Path

import pytest
import torch

# Must match Partial_Pooling/freeze_tokenized.py: BLAS reduction order, and hence
# the last bits of every matmul, depends on the thread count.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

from compass import IID, NodeSchema, ScoreBasedInferenceModel  # noqa: F401

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tokenized_arch_regression.pt"
TOLERANCE = float(os.environ.get("COMPASS_REGRESSION_TOL", "0.0"))

GLOBALS = 7
THETA = 10


def assert_identical(actual, expected, what):
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    difference = (actual.double() - expected.double()).abs().max().item()
    if TOLERANCE == 0.0:
        assert torch.equal(actual, expected), (
            f"{what}: tokenized output changed (max |delta| = {difference:.3e}). "
            "Every ladder result rests on this architecture."
        )
    else:
        assert difference <= TOLERANCE, (
            f"{what}: max |delta| = {difference:.3e} exceeds {TOLERANCE:.3e}."
        )


def load():
    if not FIXTURE.exists():
        pytest.skip(f"Missing fixture {FIXTURE}; run Partial_Pooling/freeze_tokenized.py.")
    return torch.load(FIXTURE, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def fixture():
    return load()


def case_names():
    if not FIXTURE.exists():
        return ["missing"]
    return sorted(torch.load(FIXTURE, map_location="cpu", weights_only=False)["cases"])


@pytest.mark.parametrize("name", case_names())
def test_tokenized_construction_and_forward(fixture, name):
    entry = fixture["cases"][name]
    torch.manual_seed(fixture["arch_seed"])
    model = ScoreBasedInferenceModel(device="cpu", **entry["config"])
    model.model.eval()

    assert model.nodes_size == entry["nodes_size"]
    assert model.token_count == entry["token_count"], (
        f"{name}: {model.token_count} tokens, frozen value is {entry['token_count']}."
    )
    parameters = sum(p.numel() for p in model.model.parameters())
    assert parameters == entry["parameters"], (
        f"{name}: {parameters:,} parameters, frozen value is {entry['parameters']:,}."
    )

    with torch.no_grad():
        raw = model.model(x=entry["x"], t=entry["t"], c=entry["condition_mask"])
        score = model.output_scale_function(entry["t"], raw)
    assert_identical(raw, entry["raw_output"], f"{name} raw output")
    assert_identical(score, entry["score"], f"{name} score")


def test_the_winner_is_still_463364_parameters_at_40_tokens(fixture):
    """The number every table in docs/ quotes, pinned where it cannot drift."""
    entry = fixture["cases"]["winner"]
    assert entry["parameters"] == 463_364
    assert entry["token_count"] == 40


def checkpoint_entry(fixture):
    entry = fixture.get("checkpoint")
    if entry is None:
        pytest.skip("Fixture carries no checkpoint half; retrain or refreeze.")
    path = Path(__file__).resolve().parent.parent / entry["checkpoint"]
    if not path.exists():
        pytest.skip(f"Missing checkpoint {path}.")
    return entry, path


def test_checkpoint_scores(fixture):
    entry, path = checkpoint_entry(fixture)
    model = ScoreBasedInferenceModel.load(str(path), device="cpu")
    model.model.eval()
    assert sum(p.numel() for p in model.model.parameters()) == entry["parameters"]
    assert model.schema.to_dict() == entry["schema"]
    with torch.no_grad():
        raw = model.model(x=entry["x"], t=entry["t"], c=entry["condition_mask"])
        score = model.output_scale_function(entry["t"], raw)
    assert_identical(raw, entry["raw_output"], "mask6mix raw output")
    assert_identical(score, entry["score"], "mask6mix score")


def test_checkpoint_sampler_pfode_and_composition(fixture):
    entry, path = checkpoint_entry(fixture)
    model = ScoreBasedInferenceModel.load(str(path), device="cpu")
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

    condition_mask = torch.cat([torch.zeros(THETA), torch.ones(nodes_size - THETA)])
    joint = torch.cat([torch.zeros(observations.shape[0], THETA), observations], dim=1)
    torch.manual_seed(entry["sample_seed"])
    log_prob = model.log_prob(
        data=joint, condition_mask=condition_mask, timesteps=20,
        divergence="exact", device="cpu", verbose=False,
    )
    assert_identical(log_prob, entry["log_prob"], "PF-ODE log-probability")

    torch.manual_seed(entry["sample_seed"])
    composed = model.sample(
        x=observations, timesteps=20, num_samples=16, method="dpm", order=2,
        multi_obs_inference=True, hierarchy=list(range(GLOBALS)),
        correction="gauss_jacobian", jacobian_refresh=20,
        # Same reason as above: this fixture was frozen under the stock
        # predictor-only sampler, so it pins churn off explicitly.
        churn_eta=0.0, corrector_steps=0, final_corrector_steps=0,
        device="cpu", verbose=False,
    )
    assert_identical(composed, entry["composed_draws"], "gauss_jacobian composed draws")
