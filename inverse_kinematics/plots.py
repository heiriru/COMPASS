"""The two figures the BayesFlow notebook shows before any model is trained.

``plot_prior_parameters`` reproduces ``plot_params_kinematic`` from the notebook
exactly: same four panels, same accent colour, same shared axis limits.

``plot_example_simulations`` stands in for the notebook's static "Example
simulations of the inverse kinematics problem" image, whose source is not in the
published page -- only the rendered PNG is. It is redrawn here from the arm
geometry, and the helper it referenced (``misc.diffusion.kinematics_helper.
InverseKinematicsModel``) ships with the documentation build rather than with
BayesFlow, so it is reimplemented from its described behaviour: three segments
through four joints, the target end position marked, an exemplar configuration
drawn solid over an ensemble drawn faint.
"""

import numpy as np

from simulator import PARAMETER_LABELS, joint_positions

ACCENT = "#E7298A"
INK = "#22262B"
RAIL = "#9AA3AB"


def _style_axis(axis):
    axis.set_facecolor("white")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(RAIL)
    axis.tick_params(colors=INK, labelsize=8)


def plot_prior_parameters(parameters, output, plt):
    """The notebook's ``plot_params_kinematic``, called on the training draws.

    The notebook passes the samples as ``params2``, which is its *black,
    half-transparent* series; the pink series is reserved for a posterior it does
    not have yet at this point. Reproduced as-is rather than "improved" to pink,
    so the figure matches the page.
    """
    figure, axes = plt.subplots(
        1, 4, sharex=True, sharey=True, layout="constrained", figsize=(10, 2)
    )
    for index, (axis, label) in enumerate(zip(axes, PARAMETER_LABELS)):
        axis.hist(parameters[:, index], density=True, color="black", alpha=0.5)
        axis.set_xlabel(label)
        _style_axis(axis)
    axes[0].set_ylabel("Density")
    axes[0].set_ylim(0, 1.6)
    axes[0].set_xlim(-4, 4)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _draw_arm(axis, joints, color, linewidth, alpha, zorder=2, joint_size=0):
    axis.plot(
        joints[:, 0], joints[:, 1], color=color, linewidth=linewidth,
        alpha=alpha, solid_capstyle="round", zorder=zorder,
    )
    if joint_size:
        axis.plot(
            joints[:, 0], joints[:, 1], linestyle="none", marker="o",
            markersize=joint_size, color=color, alpha=alpha, zorder=zorder + 1,
        )


def _draw_rail(axis, span=1.6):
    axis.plot(
        [0, 0], [-span, span], color=RAIL, linewidth=1.2, linestyle=(0, (4, 3)),
        zorder=0,
    )


def plot_example_simulations(prior_parameters, consistent_parameters, target,
                             output, plt):
    """What a draw looks like, and why the inverse problem is hard.

    Top strip -- independent prior-predictive draws, one per panel. Overlaying
    them in a single axis is unreadable: the arms cross, and a configuration that
    folds back on itself cannot be told from two that happen to intersect. One
    axis per draw keeps each configuration legible, which is the whole job of
    this row.

    Bottom -- configurations that all reach the *same* end position to within a
    hundredth of a unit. This is the non-identifiability the notebook's opening
    image exists to show.
    """
    count = len(prior_parameters)
    figure = plt.figure(figsize=(9.5, 6.4), layout="constrained")
    grid = figure.add_gridspec(
        2, count, height_ratios=[1.0, 1.9], hspace=0.06,
    )

    prior_joints = joint_positions(prior_parameters)
    strip = [figure.add_subplot(grid[0, index]) for index in range(count)]
    for axis, joints in zip(strip, prior_joints):
        _draw_rail(axis, span=1.9)
        _draw_arm(axis, joints, ACCENT, 1.8, 0.95, joint_size=3.0)
        axis.plot(
            joints[-1, 0], joints[-1, 1], marker="o", markersize=5,
            color=INK, zorder=5,
        )
        axis.set_aspect("equal")
        axis.set_xlim(-2.3, 2.3)
        axis.set_ylim(-2.1, 2.1)
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_color(RAIL)
            side.set_linewidth(0.6)
    strip[0].set_ylabel("Prior draws", fontsize=9, color=INK)

    fan = figure.add_subplot(grid[1, :])
    consistent_joints = joint_positions(consistent_parameters)
    for joints in consistent_joints[1:]:
        _draw_arm(fan, joints, ACCENT, 1.2, 0.16)
    _draw_arm(fan, consistent_joints[0], ACCENT, 2.2, 1.0, zorder=4,
              joint_size=3.4)
    # `target` is in observation order (x1, x2); the plot is (x2, x1).
    fan.plot(
        target[1], target[0], marker="X", markersize=11, color=INK,
        markeredgecolor="white", markeredgewidth=1.0, zorder=6,
    )
    _draw_rail(fan)
    _style_axis(fan)
    fan.set_aspect("equal")
    # Tight to the drawn arms rather than to the strip's square framing: every
    # configuration starts on the rail at x2 = 0 and reaches right, so the
    # symmetric window the strip needs is mostly empty here.
    fan.set_xlim(-0.75, 2.15)
    fan.set_ylim(-1.25, 1.15)
    fan.set_xlabel("$x_2$", color=INK)
    fan.set_ylabel("$x_1$", color=INK)
    fan.set_title(
        f"{len(consistent_parameters)} configurations reaching one end position",
        fontsize=10, color=INK,
    )

    figure.suptitle(
        "Example simulations of the inverse kinematics problem",
        fontsize=12, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_observation_space(observables, output, plt):
    """Where the end effector lands under the prior.

    Not in the notebook. Included because the two figures above show the arm and
    the parameters but never the observation space the inference problem is
    posed in, and it costs nothing once the simulations exist. Kept clearly
    separate from the reproductions above.
    """
    figure, axis = plt.subplots(figsize=(4.4, 4.4), layout="constrained")
    axis.hexbin(
        observables[:, 1], observables[:, 0], gridsize=60, cmap="magma_r",
        mincnt=1, linewidths=0,
    )
    _draw_rail(axis)
    _style_axis(axis)
    axis.set_aspect("equal")
    axis.set_xlabel("$x_2$", color=INK)
    axis.set_ylabel("$x_1$", color=INK)
    axis.set_title(
        f"End positions of {len(observables):,} prior draws",
        fontsize=10, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_params_kinematic(params=None, params2=None, output=None, plt_module=None,
                          title=None):
    """The notebook's ``plot_params_kinematic``, behaviour-for-behaviour.

    Same four shared axes, same default ten bins, same colours and limits:
    ``params2`` is the black half-transparent reference series (their training
    draws, i.e. the prior) and ``params`` is the magenta posterior drawn over it.
    Kept separate from :func:`plot_prior_parameters` so the single-observation
    reproduction is not quietly restyled.
    """
    plt_module = plt_module or __import__("matplotlib.pyplot", fromlist=["x"])
    figure, axes = plt_module.subplots(
        1, 4, sharex=True, sharey=True, layout="constrained", figsize=(10, 2)
    )
    for index, (axis, name) in enumerate(zip(axes, PARAMETER_LABELS)):
        if params2 is not None:
            axis.hist(params2[:, index], density=True, color="black", alpha=0.5)
        if params is not None:
            axis.hist(params[:, index], density=True, color=ACCENT)
        axis.set_xlabel(name)
        _style_axis(axis)
    axes[0].set_ylabel("Density")
    axes[0].set_ylim(0, 1.6)
    axes[0].set_xlim(-4, 4)
    if title:
        figure.suptitle(title, fontsize=11, color=INK)
    if output is not None:
        figure.savefig(output, dpi=200, bbox_inches="tight")
        plt_module.close(figure)
    return output
