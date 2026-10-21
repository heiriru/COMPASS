"""The architecture mapping constructs, and the mismatches are the stated ones.

No training and no sampling: this builds the networks and pushes one random
tensor through, which is what catches a hidden_size/num_heads incompatibility
before a GPU job discovers it an hour in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from common import paper, tasks as task_module
from common.model import (ARMS, PAPER_HIDDEN_SIZE, build_model,
                          sampling_kwargs, training_kwargs)


@pytest.mark.parametrize("name", ["two_moons", "slcp", "hmm"])
def test_paper_matched_model_builds_and_runs_forward(name):
    task = task_module.get_task(name)
    model = build_model(task, arm="paper_matched")
    assert model.schema.nodes_size == task.nodes_size
    z = torch.randn(4, task.nodes_size)
    t = torch.rand(4, 1)
    mask = torch.zeros(4, task.nodes_size)
    mask[:, task.theta_dim:] = 1.0
    score = model.model(z, t, mask)
    assert score.shape == z.shape
    assert torch.isfinite(score).all()


def test_hidden_size_must_divide_by_heads():
    task = task_module.get_task("two_moons")
    with pytest.raises(ValueError, match="divisible"):
        build_model(task, hidden_size=paper.ARCHITECTURE["token_dim"])
    assert PAPER_HIDDEN_SIZE % paper.ARCHITECTURE["heads"] == 0


def test_legacy_arch_also_builds():
    task = task_module.get_task("two_moons")
    model = build_model(task, arch="legacy")
    assert model.schema.nodes_size == task.nodes_size


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_has_a_training_and_sampling_recipe(arm):
    task = task_module.get_task("two_moons")
    model = build_model(task, arm=arm)
    kwargs = training_kwargs(task, model, arm=arm)
    assert kwargs["batch_size"] == paper.TRAINING["batch_size"]
    sampler = kwargs["mask_sampler"]
    mask = sampler.sample((16, task.nodes_size))
    assert mask.shape == (16, task.nodes_size)

    sampling = sampling_kwargs(arm=arm)
    assert sampling["eps"] == paper.TIME_INTERVAL[0]
    if arm == "compass_native":
        assert sampling["method"] == "dpm" and sampling["churn_eta"] == 2.0
    else:
        assert sampling["method"] == "euler"
        assert sampling["timesteps"] == paper.SAMPLING["steps"]


def test_posterior_only_arm_trains_on_one_mask():
    task = task_module.get_task("two_moons")
    model = build_model(task, arm="posterior_only")
    sampler = training_kwargs(task, model, arm="posterior_only")["mask_sampler"]
    mask = sampler.sample((5, task.nodes_size))
    assert torch.equal(mask, mask[0].expand_as(mask))
    assert mask[0, : task.theta_dim].sum() == 0
