"""Metrics the Simformer paper reports: C2ST, expected coverage, average NLL.

C2ST is the paper's primary metric and does not exist anywhere else in this
repository -- COMPASS scores itself with SBC/coverage, standardised RMSE,
sliced-Wasserstein and endpoint error. Nothing in COMPASS is comparable to a
published C2ST number until this module is used.

The canonical implementation is sbibm's: an ``sklearn`` MLP classifier with two
hidden layers of ``10 * dim`` units, 5-fold cross-validation, accuracy averaged
over folds, on z-scored features. scikit-learn is **not** installed in
``.COMPASS``; ``c2st`` therefore uses sklearn when importable and otherwise a
torch reimplementation with the same architecture and protocol. The two are not
bit-identical -- optimiser defaults differ -- so a table must record which path
produced it (``c2st`` returns that in its result).
"""

import math

import torch


def _zscore(a, b):
    joint = torch.cat([a, b], dim=0)
    mean, std = joint.mean(0, keepdim=True), joint.std(0, keepdim=True)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return (a - mean) / std, (b - mean) / std


def _folds(n, k, generator=None):
    order = torch.randperm(n, generator=generator)
    return [order[i::k] for i in range(k)]


def _torch_c2st_fold(x_train, y_train, x_test, y_test, hidden, epochs, lr, generator):
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 2),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = x_train.shape[0]
    batch = min(128, n)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch):
            index = order[start:start + batch]
            optimiser.zero_grad()
            loss = loss_fn(model(x_train[index]), y_train[index])
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        predicted = model(x_test).argmax(-1)
    return (predicted == y_test).to(torch.float).mean().item()


def c2st(samples_a, samples_b, folds=5, seed=0, epochs=100, lr=1e-3,
         prefer_sklearn=True):
    """Classifier two-sample test accuracy between two sample sets.

    0.5 means the two samples are indistinguishable (a perfect posterior
    approximation); 1.0 means perfectly separable. Returns a dict with the
    score, the per-fold accuracies and the backend that produced them.

    The two sample sets should have the same size; sbibm uses 10,000 each.
    """
    samples_a = torch.as_tensor(samples_a, dtype=torch.float)
    samples_b = torch.as_tensor(samples_b, dtype=torch.float)
    if samples_a.shape[1] != samples_b.shape[1]:
        raise ValueError(
            f"Dimension mismatch: {samples_a.shape[1]} vs {samples_b.shape[1]}."
        )
    a, b = _zscore(samples_a, samples_b)
    features = torch.cat([a, b], dim=0)
    labels = torch.cat([torch.zeros(len(a)), torch.ones(len(b))]).long()
    hidden = 10 * samples_a.shape[1]

    backend = "torch"
    if prefer_sklearn:
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.neural_network import MLPClassifier
            backend = "sklearn"
        except ImportError:
            backend = "torch"

    if backend == "sklearn":
        classifier = MLPClassifier(
            activation="relu", hidden_layer_sizes=(hidden, hidden),
            max_iter=1000, solver="adam", random_state=seed,
        )
        scores = cross_val_score(
            classifier, features.numpy(), labels.numpy(), cv=folds,
            scoring="accuracy",
        )
        scores = [float(s) for s in scores]
    else:
        generator = torch.Generator().manual_seed(seed)
        scores = []
        for held_out in _folds(len(features), folds, generator):
            mask = torch.ones(len(features), dtype=torch.bool)
            mask[held_out] = False
            scores.append(_torch_c2st_fold(
                features[mask], labels[mask], features[~mask], labels[~mask],
                hidden, epochs, lr, generator,
            ))
    return {
        "c2st": float(sum(scores) / len(scores)),
        "folds": scores,
        "backend": backend,
        "n_a": int(len(samples_a)),
        "n_b": int(len(samples_b)),
    }


def expected_coverage(samples, truth, levels=None):
    """Expected-coverage curve (Hermans et al. 2022), as used in Fig. A9-A13.

    ``samples`` is ``(n_cases, n_draws, dim)`` and ``truth`` is
    ``(n_cases, dim)``. For each case the highest-density region is approximated
    by ranking draws on a Gaussian KDE-free proxy: the distance to the posterior
    mean in whitened coordinates. That is the cheap standard proxy and is exact
    for Gaussian posteriors only -- for the multimodal tasks (Two Moons, Tree,
    HMM) prefer SBC ranks, which is what ``inverse_kinematics/sbc_reference.py``
    already computes and what should be reused there.
    """
    if levels is None:
        levels = torch.linspace(0.05, 0.95, 19)
    levels = torch.as_tensor(levels, dtype=torch.float)
    samples = torch.as_tensor(samples, dtype=torch.float)
    truth = torch.as_tensor(truth, dtype=torch.float)
    mean = samples.mean(1, keepdim=True)
    std = samples.std(1, keepdim=True).clamp_min(1e-12)
    draw_distance = (((samples - mean) / std) ** 2).sum(-1)
    truth_distance = (((truth.unsqueeze(1) - mean) / std) ** 2).sum(-1).squeeze(1)
    covered = []
    for level in levels:
        radius = torch.quantile(draw_distance, level, dim=1)
        covered.append((truth_distance <= radius).to(torch.float).mean().item())
    return {"levels": levels.tolist(), "coverage": covered}


def average_nll(log_probs):
    """Average negative log-likelihood of the true value (Fig. A8).

    ``log_probs`` are ``log q(truth)`` under the approximation, one per joint
    draw. The paper averages over 5000 draws from the joint, for the posterior
    and for the likelihood separately. For COMPASS these come from
    ``ScoreBasedInferenceModel.log_prob`` (PF-ODE with exact divergence), which
    is the same discrepancy the paper flags: the SDE samples and the ODE
    density are not the same object at finite step counts.
    """
    log_probs = torch.as_tensor(log_probs, dtype=torch.float)
    finite = torch.isfinite(log_probs)
    if not finite.any():
        return {"nll": float("nan"), "n": 0, "n_nonfinite": int(len(log_probs))}
    return {
        "nll": float(-log_probs[finite].mean()),
        "n": int(finite.sum()),
        "n_nonfinite": int((~finite).sum()),
        "sem": float(log_probs[finite].std() / math.sqrt(int(finite.sum()))),
    }
