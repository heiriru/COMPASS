"""A1: counted information criteria (hand-computed values and compare() wiring).

    .COMPASS/bin/python -m pytest -q validation/tests
"""

import validation.common.runtime  # noqa: F401

import contextlib
import io
import math

import pytest
import torch

from compass import information_criteria as ic
from compass.ModelTransfuser import ModelTransfuser
from validation.common.gaussian_oracles import nested_offset_pair, oracle_sbim


LL = torch.tensor([-1.0, -2.0, -0.5, -1.5])   # N = 4 rows, logL = -5


def test_effective_parameter_count():
    assert ic.effective_parameter_count(2, 3, 10) == 32
    assert ic.effective_parameter_count(0, 2, 7) == 14


def test_aic_counts_globals_once_and_locals_per_row():
    r = ic.information_criterion(LL, "aic", k_global=1, k_local=2, n_per_observation=3)
    K = 1 + 4 * 2
    assert r["K"] == K
    assert r["total"].item() == pytest.approx(10.0 + 2 * K)


def test_aicc_value_and_undefined_flag():
    r = ic.information_criterion(LL, "aicc", 1, 0, n_per_observation=3)  # K=1, n=12
    assert r["aicc_defined"]
    assert r["total"].item() == pytest.approx(10.0 + 2 + 2 * 1 * 2 / (12 - 1 - 1))
    r = ic.information_criterion(LL, "aicc", 0, 3, n_per_observation=3)  # K=12, n=12
    assert not r["aicc_defined"]
    assert r["total"].item() == pytest.approx(10.0 + 24)


def test_bic_charges_each_block_its_own_data():
    r = ic.information_criterion(LL, "bic", k_global=2, k_local=1, n_per_observation=5)
    expected = 10.0 + 2 * math.log(20) + 4 * 1 * math.log(5)
    assert r["total"].item() == pytest.approx(expected)


@pytest.mark.parametrize("criterion", ["aic", "aicc", "bic"])
def test_per_row_terms_sum_to_total(criterion):
    r = ic.information_criterion(LL, criterion, 1, 1, 3)
    assert r["per_row"].sum().item() == pytest.approx(r["total"].item())


def test_aicc_falls_back_for_all_models_jointly():
    # Model A: K = 4 on n = 8 (defined); model B: K = 8 (undefined).
    results, used = ic.compare_criteria([LL, LL], "aicc", [0, 0], [1, 2], 2)
    assert used == "aic"
    assert results[0]["total"].item() == pytest.approx(10.0 + 8)
    assert results[1]["total"].item() == pytest.approx(10.0 + 16)
    results, used = ic.compare_criteria([LL, LL], "aicc", [0, 1], [0, 0], 4)
    assert used == "aicc"


def test_bic_local_penalty_warning_threshold():
    assert ic.bic_local_penalty_warning("bic", 1, 2) is not None      # ln 2 < 1
    assert ic.bic_local_penalty_warning("bic", 1, 3) is None          # ln 3 > 1
    assert ic.bic_local_penalty_warning("bic", 0, 2) is None          # no locals
    assert ic.bic_local_penalty_warning("aic", 1, 2) is None


def test_compare_counts_hierarchy_dims_as_globals():
    """Through compare() with an oracle: the global g is counted once."""
    n = 6
    joint1, _ = nested_offset_pair(s_g=1.0)
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(0))
    mtf = ModelTransfuser()
    with contextlib.redirect_stdout(io.StringIO()):
        mtf.add_model("m1", oracle_sbim(joint1))
        mtf.trained_models = True
        mtf.compare(x=x, num_samples=64, timesteps=20, device="cpu",
                    multi_obs_inference=True, hierarchy=[0],
                    prior=(torch.zeros(1), torch.ones(1)),
                    correction="gauss_hierarchical", criterion="aic_plain",
                    point_estimate="map")
    st = mtf.stats["m1"]
    assert (st["k_global"], st["k_local"], st["K_effective"]) == (1, 1, 1 + n)
    expected = -2 * st["log_probs"].double().sum() + 2 * (1 + n)
    assert st["AIC"].item() == pytest.approx(expected.item(), rel=1e-5)
