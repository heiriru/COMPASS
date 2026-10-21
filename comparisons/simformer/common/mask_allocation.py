"""Where a training condition-mask law actually sends its gradient.

The denoising-score-matching loss is evaluated on *latent* nodes only, so a mask
law's gradient allocation between "learning the posterior" (theta latent) and
"learning to generate data" (x latent) is fixed by two things and nothing else:

    share(role) = E[#latent nodes of that role] x (per-node difficulty of that role)

The first factor is pure arithmetic over the law and the role sizes, and is what
this module computes exactly. The second is a property of the *task* -- how hard
each node is to denoise -- and has to be measured (see
``Partial_Pooling/score_conditional_losses.py``).

The point of separating them is that the first factor turns out to be identical
for the Simformer law and for COMPASS's six-conditional menu, which means it
cannot be the explanation for why one needs reweighting and the other does not.
"""

from fractions import Fraction


def simformer_law(roles):
    """The paper's five-way uniform menu, as (name, conditioned roles, weight).

    Sec. 3.1 / A2.1: joint, posterior, likelihood, Bernoulli(0.3), Bernoulli(0.7),
    drawn uniformly. The two Bernoulli patterns condition each node independently,
    so they are expressed here by their expected conditioning rate rather than by
    a role set.
    """
    return [
        ("joint", {}, Fraction(1, 5)),
        ("posterior", {"x": 1}, Fraction(1, 5)),
        ("likelihood", {r: 1 for r in roles if r != "x"}, Fraction(1, 5)),
        ("bernoulli_0.3", {r: Fraction(3, 10) for r in roles}, Fraction(1, 5)),
        ("bernoulli_0.7", {r: Fraction(7, 10) for r in roles}, Fraction(1, 5)),
    ]


def bernoulli_law(roles, rate=Fraction(33, 100)):
    """COMPASS's historical default: every node conditioned independently."""
    return [("bernoulli", {r: rate for r in roles}, Fraction(1))]


def six_conditional_law(weights=None):
    """The six non-degenerate conditionals over (global, local, observation).

    ``weights`` are relative, in the order COMPASS uses:
    p(g,l|x), p(g|l,x), p(l|g,x), p(x|g,l), p(x,l|g), p(x,g|l).
    ``None`` is the uniform menu; ``[7,7,7,1,1,1]`` is the posterior-weighted one.
    """
    patterns = [
        ("p(g,l|x)", {"x": 1}),
        ("p(g|l,x)", {"l": 1, "x": 1}),
        ("p(l|g,x)", {"g": 1, "x": 1}),
        ("p(x|g,l)", {"g": 1, "l": 1}),
        ("p(x,l|g)", {"g": 1}),
        ("p(x,g|l)", {"l": 1}),
    ]
    if weights is None:
        weights = [1] * 6
    total = sum(Fraction(w) for w in weights)
    return [(name, cond, Fraction(w) / total)
            for (name, cond), w in zip(patterns, weights)]


def mix(law_a, law_b, fraction_b):
    """``fraction_b`` of draws come from ``law_b`` -- COMPASS's ``six_mixed``."""
    fraction_b = Fraction(fraction_b)
    return ([(n, c, w * (1 - fraction_b)) for n, c, w in law_a]
            + [(n, c, w * fraction_b) for n, c, w in law_b])


def expected_latent_nodes(law, sizes):
    """Expected number of latent nodes per role, per training draw."""
    out = {role: Fraction(0) for role in sizes}
    for _name, conditioned, weight in law:
        for role, size in sizes.items():
            rate = Fraction(conditioned.get(role, 0))
            out[role] += weight * size * (1 - rate)
    return out


def posterior_share(law, sizes, difficulty=None):
    """Fraction of the loss-bearing nodes that are parameters, not data.

    ``difficulty`` optionally scales each role by its measured per-node loss;
    with it left at 1 this is the pure node-count allocation.
    """
    difficulty = difficulty or {}
    latent = expected_latent_nodes(law, sizes)
    weighted = {r: n * Fraction(difficulty.get(r, 1)) for r, n in latent.items()}
    parameter = sum(v for r, v in weighted.items() if r != "x")
    total = sum(weighted.values())
    return float(parameter / total) if total else float("nan")


#: The role sizes of every benchmark that matters to this question.
#: (global, local, observation) -- flat tasks put everything in ``g``.
BENCHMARKS = {
    # The four tasks the paper scores C2ST on (A2.2).
    "gaussian_linear (paper)": {"g": 10, "l": 0, "x": 10},
    "gaussian_mixture (paper)": {"g": 2, "l": 0, "x": 2},
    "two_moons (paper)": {"g": 2, "l": 0, "x": 2},
    "slcp (paper)": {"g": 5, "l": 0, "x": 8},
    # Tasks they introduce for the all-conditional figure.
    "tree (paper)": {"g": 3, "l": 0, "x": 4},
    "hmm (paper)": {"g": 10, "l": 0, "x": 10},
    # COMPASS's own benchmarks.
    "inverse_kinematics": {"g": 4, "l": 0, "x": 2},
    "partial_pooling": {"g": 7, "l": 3, "x": 90},
}
