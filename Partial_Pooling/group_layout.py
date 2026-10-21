"""Mapping between 100 independent subjects and ``100/K`` composed groups.

The flat arms compose 100 single-subject factors. A grouped arm composes
``100/K`` factors of ``K`` subjects each, which changes nothing about the
metrics -- the recovery report is still per subject -- but does change the shape
of everything between the test file and the sampler. These are the pure
functions for that, kept out of ``infer_partial_pooling.py`` so they can be
tested without running inference.

Two properties they exist to guarantee:

* **The grouping is an explicit argument, never implicit.** The exact Bayesian
  posterior does not depend on which subjects share a group, so any dependence
  is an artifact of the representation and has to be measurable. ``assignment``
  is therefore threaded through rather than assumed contiguous.
* **Un-grouping restores the original subject order**, so every downstream
  metric, CSV and figure is produced by the existing per-subject code with no
  knowledge that grouping happened at all.
"""

import torch

# The partial-pooling node vector under a Group: [7 group-level, K x (3
# subject-level, 3*trials observations)]. See Partial_Pooling/schema.py.
GLOBAL_DIM = 7
LOCAL_DIM = 3


def grouping(subjects, per_group, seed=None):
    """``(rows, per_group)`` subject indices, one row per composed factor.

    ``seed=None`` gives the contiguous grouping (subjects 0..K-1 in row 0, and
    so on), which is the default an ordinary run uses. Passing a seed gives an
    independent random re-assignment of the same subjects, which is what the
    partition-sensitivity study varies: the posterior should not care, and the
    point is to measure whether it does.
    """
    subjects, per_group = int(subjects), int(per_group)
    if per_group < 1:
        raise ValueError(f"per_group must be >= 1; got {per_group}.")
    if subjects % per_group:
        raise ValueError(
            f"per_group={per_group} must divide subjects={subjects}, so every "
            "composed factor holds the same number of subjects."
        )
    order = torch.arange(subjects)
    if seed is not None:
        order = order[torch.randperm(subjects, generator=torch.Generator().manual_seed(int(seed)))]
    return order.reshape(subjects // per_group, per_group)


def group_observations(observations, assignment):
    """``(subjects, obs)`` -> ``(rows, per_group * obs)``, repeat-major.

    Within a row the node order is ``[locals_0, trials_0, locals_1, ...]``; the
    *observed* subset of that, in order, is ``[trials_0, trials_1, ...]``, which
    is exactly this concatenation. That is what lets the sampler's
    ``joint_data[condition_mask == 1] = data.flatten()`` line up.
    """
    rows, per_group = assignment.shape
    gathered = observations[assignment.reshape(-1)]
    return gathered.reshape(rows, per_group * observations.shape[-1])


def condition_mask(per_group, observation_dim, local_dim=LOCAL_DIM,
                   global_dim=GLOBAL_DIM):
    """The deployed posterior mask for a grouped row: only trials conditioned.

    Built explicitly because there is no contiguous ``[theta, x]`` split to
    infer it from -- each repeat's latent locals sit inside its observed trials,
    and ``ScoreBasedInferenceModel.sample`` refuses to guess for exactly that
    reason.
    """
    repeat = torch.cat((torch.zeros(local_dim), torch.ones(observation_dim)))
    return torch.cat((torch.zeros(global_dim), repeat.repeat(per_group)))


def split_latents(latents, per_group, global_dim=GLOBAL_DIM, local_dim=LOCAL_DIM):
    """``(rows, draws, global + per_group*local)`` -> globals and per-slot locals.

    ``Sampler`` returns the latent columns in node order, so the layout is
    ``[g, l_0, l_1, ...]`` and this is a slice plus a reshape.
    """
    expected = global_dim + per_group * local_dim
    if latents.shape[-1] != expected:
        raise ValueError(
            f"Expected {expected} latent columns for per_group={per_group}, got "
            f"{latents.shape[-1]}."
        )
    shared = latents[..., :global_dim]
    locals_ = latents[..., global_dim:]
    return shared, locals_.reshape(*locals_.shape[:-1], per_group, local_dim)


def ungroup_locals(locals_, assignment):
    """``(rows, draws, per_group, local)`` -> ``(draws, subjects, local)``.

    The inverse of :func:`grouping` composed with :func:`group_observations`:
    slot ``(row, slot)`` goes back to subject ``assignment[row, slot]``, so the
    result is in the test file's own subject order however the grouping was
    drawn. Everything downstream is then the existing per-subject code.
    """
    rows, per_group = assignment.shape
    if locals_.shape[0] != rows or locals_.shape[2] != per_group:
        raise ValueError(
            f"locals shaped {tuple(locals_.shape)} do not match an "
            f"assignment of {rows} rows x {per_group} subjects."
        )
    draws, local_dim = locals_.shape[1], locals_.shape[3]
    flat = locals_.permute(1, 0, 2, 3).reshape(draws, rows * per_group, local_dim)
    restored = torch.empty_like(flat)
    restored[:, assignment.reshape(-1)] = flat
    return restored


def tile_local_prior(prior, per_group):
    """Repeat a per-subject ``(mean, std)`` prior across a row's slots."""
    mean, std = prior
    return (
        torch.as_tensor(mean).repeat(per_group),
        torch.as_tensor(std).repeat(per_group),
    )


def inverse_transform(shared, locals_, theta_normalizer, global_dim=GLOBAL_DIM):
    """Undo normalization with the joint-theta constants, applied per subject."""
    mean, scale = theta_normalizer.mean, theta_normalizer.scale
    shared = shared * scale[:global_dim] + mean[:global_dim]
    locals_ = locals_ * scale[global_dim:] + mean[global_dim:]
    return shared, locals_


def map_condition_mask(per_group, observation_dim, local_dim=LOCAL_DIM,
                       global_dim=GLOBAL_DIM):
    """Stage-2 mask of the KDE-global-then-local-ascent estimator.

    The globals have been fixed at their KDE mode and are *conditioned*; only
    each repeat's locals stay latent. Same shape as :func:`condition_mask`, with
    the group-level block flipped.
    """
    repeat = torch.cat((torch.zeros(local_dim), torch.ones(observation_dim)))
    return torch.cat((torch.ones(global_dim), repeat.repeat(per_group)))


def build_rows(shared, locals_, observations, assignment):
    """Per-subject tensors -> ``(rows, nodes)`` grouped node vectors.

    ``shared`` is one group-level vector, broadcast to every row; ``locals_``
    and ``observations`` are per subject and are gathered into slot order.
    """
    rows, per_group = assignment.shape
    index = assignment.reshape(-1)
    repeats = torch.cat((locals_[index], observations[index]), dim=-1)
    shared = shared.reshape(1, -1).expand(rows, -1)
    return torch.cat((shared, repeats.reshape(rows, -1)), dim=-1)


def extract_locals(rows, per_group, observation_dim, assignment,
                   local_dim=LOCAL_DIM, global_dim=GLOBAL_DIM):
    """``(rows, nodes)`` -> ``(subjects, local)`` in the test file's order."""
    width = local_dim + observation_dim
    repeats = rows[:, global_dim:].reshape(rows.shape[0], per_group, width)
    flat = repeats[:, :, :local_dim].reshape(-1, local_dim)
    restored = torch.empty_like(flat)
    restored[assignment.reshape(-1)] = flat
    return restored


def subjects_per_group(model):
    """Subjects in one composed factor: the Group's count, or 1 for a flat arm."""
    if getattr(model, "arch", None) != "grouped":
        return 1
    for block in model.schema.blocks:
        if block.kind == "group":
            return block.count
    raise ValueError("A grouped model has no Group block.")
