"""
Information criteria with explicit parameter and data counting.

Conventions (see validation/TODO.md, item A1):

* A comparison is made on ``N`` observation rows ``x_1..x_N``.
* A model has ``k_global`` parameters shared by all rows (hierarchy / population
  level, inferred once from all rows) and ``k_local`` parameters per row (fitted
  separately to every row). A non-hierarchical model applied row by row has
  ``k_global = 0`` and ``k_local = theta_dim``.
* The effective parameter count is ``K = k_global + N * k_local``: a global
  parameter is counted once, every row's local parameters once each.
* ``n_per_observation`` is the number of scalar measurements in one row
  (default: the number of observed dimensions). The total data count is
  ``n_data = N * n_per_observation``.

Criteria (``logL`` is the summed maximum log-likelihood over the rows):

    AIC  = -2 logL + 2 K
    AICc = AIC + 2 K (K + 1) / (n_data - K - 1)      (undefined for n_data <= K + 1)
    BIC  = -2 logL + k_global ln(n_data) + N k_local ln(n_per_observation)

BIC charges every parameter block the log of the data that inform it: a global
parameter sees all ``n_data`` scalars, a local parameter only its row's
``n_per_observation`` scalars.

The per-row decomposition spreads each penalty evenly over the rows,

    IC_i = -2 logL_i + penalty / N,

so that ``sum_i IC_i`` equals the total exactly. It is what per-observation model
probabilities and cumulative plots are built from.
"""

import math
import numpy as np

import torch

CRITERIA = ("aic", "aicc", "bic")


def effective_parameter_count(k_global, k_local, n_rows):
    """``K = k_global + n_rows * k_local``."""
    return int(k_global) + int(n_rows) * int(k_local)


def criterion_penalty(criterion, k_global, k_local, n_rows, n_per_observation):
    """Return ``(penalty, info)`` for one model; the criterion is ``-2 logL + penalty``.

    ``info`` holds ``K``, ``n_data`` and whether AICc was defined (when it is
    not, the plain AIC penalty is returned and ``aicc_defined`` is False).
    """
    criterion = canonical_criterion(criterion)
    k_global, k_local, n_rows = int(k_global), int(k_local), int(n_rows)
    n_per_observation = float(n_per_observation)
    if min(k_global, k_local) < 0 or n_rows < 1 or n_per_observation <= 0:
        raise ValueError(
            "Need k_global, k_local >= 0, n_rows >= 1 and n_per_observation > 0; got "
            f"{k_global}, {k_local}, {n_rows}, {n_per_observation}."
        )
    K = effective_parameter_count(k_global, k_local, n_rows)
    n_data = n_rows * n_per_observation
    info = {"K": K, "n_data": n_data, "aicc_defined": True}

    if criterion == "bic":
        penalty = k_global * math.log(n_data) + n_rows * k_local * math.log(n_per_observation)
        return penalty, info

    penalty = 2.0 * K
    if criterion == "aicc":
        denom = n_data - K - 1
        if denom <= 0:
            info["aicc_defined"] = False
        else:
            penalty += 2.0 * K * (K + 1) / denom
    return penalty, info


def information_criterion(log_likelihood_rows, criterion, k_global, k_local,
                          n_per_observation):
    """Criterion of one model from its per-row maximum log-likelihoods.

    Args:
        log_likelihood_rows: tensor ``(N,)`` of ``log p(x_i | theta_hat)``.
        criterion: ``"aic"``, ``"aicc"`` or ``"bic"`` (``"aic"`` here is the
            *plain* AIC; ``ModelTransfuser.compare`` maps its historical
            ``criterion="aic"`` to AICc before calling this).
        k_global, k_local, n_per_observation: see the module docstring.

    Returns:
        dict with ``total`` (scalar tensor), ``per_row`` (``(N,)`` tensor summing to
        ``total``), ``penalty``, ``K``, ``n_data``, ``aicc_defined``.
    """
    ll = torch.as_tensor(log_likelihood_rows, dtype=torch.float64).reshape(-1)
    n_rows = ll.numel()
    penalty, info = criterion_penalty(criterion, k_global, k_local, n_rows,
                                      n_per_observation)
    per_row = -2.0 * ll + penalty / n_rows
    out = dict(info)
    out.update(total=per_row.sum(), per_row=per_row, penalty=penalty)
    return out


def compare_criteria(log_likelihood_rows, criterion, k_globals, k_locals,
                     n_per_observation):
    """Criteria of several models evaluated on the same rows.

    AICc is only usable if it is defined for *every* model: comparing one model's
    AICc with another's plain-AIC fallback mixes two different penalties and, in
    the flat case, always favours the larger model (validation/a1, E-A1.4). If
    any model has ``n_data <= K + 1``, all models fall back to plain AIC.

    Returns a list of :func:`information_criterion` dicts (one per model) and the
    criterion actually used.
    """
    name = canonical_criterion(criterion)
    results = [information_criterion(ll, name, kg, kl, n_per_observation)
               for ll, kg, kl in zip(log_likelihood_rows, k_globals, k_locals)]
    if name == "aicc" and not all(r["aicc_defined"] for r in results):
        results = [information_criterion(ll, "aic", kg, kl, n_per_observation)
                   for ll, kg, kl in zip(log_likelihood_rows, k_globals, k_locals)]
        for r in results:
            r["aicc_defined"] = False
        return results, "aic"
    return results, name


def canonical_criterion(criterion):
    name = str(criterion).lower()
    if name not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    return name


def bic_local_penalty_warning(criterion, k_local, n_per_observation):
    """Message if BIC's per-row penalty is too weak to be consistent, else None.

    A superfluous local parameter raises a row's maximum log-likelihood by
    chi^2_1 / 2 on average, i.e. -2 logL drops by 1 per row, while BIC charges
    ln(n_per_observation). For n_per_observation <= e the penalty is below the
    expected gain, so BIC *prefers* the superfluous parameters more and more
    often as N grows (validation/a1_information_criteria, E-A1.4).
    """
    if canonical_criterion(criterion) != "bic" or int(k_local) == 0:
        return None
    if math.log(float(n_per_observation)) > 1.0:
        return None
    return ("BIC with n_per_observation = {:g}: each per-row (local) parameter is "
            "charged ln({:g}) = {:.2f} < 1, less than the expected gain of a "
            "superfluous parameter, so BIC is not consistent for local parameters "
            "here. Use AIC, or an evidence-based comparison.").format(
                float(n_per_observation), float(n_per_observation),
                math.log(float(n_per_observation)))


def akaike_weights(criterion_values):
    """``softmax(-IC / 2)`` over models (the first axis)."""
    values = torch.as_tensor(criterion_values, dtype=torch.float64)
    return torch.softmax(-0.5 * values, dim=0)


# ---------------------------------------------------------------------------
# WAIC (Watanabe 2010; Gelman, Hwang & Vehtari 2014; Vehtari, Gelman & Gabry 2017)
# ---------------------------------------------------------------------------
#
# From the pointwise log-likelihoods ``ll[i, s] = log p(x_i | theta_s)`` at S
# posterior draws (for row i, theta_s holds that draw's shared globals and row i's
# locals):
#
#     lppd_i   = log (1/S) sum_s p(x_i | theta_s)
#     p_waic_i = Var_s log p(x_i | theta_s)            (sample variance, ddof = 1)
#     elpd_i   = lppd_i - p_waic_i
#     WAIC     = -2 sum_i elpd_i                       (deviance scale, like AIC)
#
# WAIC needs no point estimate, no parameter count and no asymptotic argument:
# the effective number of parameters is p_waic. With locals in theta it is the
# *conditional* WAIC (prediction of a new measurement of the same unit).

# Vehtari, Gelman & Gabry (2017): WAIC is unreliable once a row's p_waic exceeds 0.4.
WAIC_P_WARNING = 0.4


def _waic_total(ll):
    n_draws = ll.shape[1]
    return float((-2.0 * (torch.logsumexp(ll, dim=1) - math.log(n_draws)
                          - ll.var(dim=1, unbiased=True))).sum())


def waic(log_likelihood, mc_groups=10):
    """WAIC of one model from its ``(N, S)`` pointwise log-likelihood matrix.

    Returns a dict with
        ``total``      WAIC (deviance scale, scalar tensor),
        ``per_row``    ``-2 elpd_i`` (sums to ``total``),
        ``elpd``       ``sum_i elpd_i`` and ``elpd_rows`` (``(N,)``),
        ``lppd``, ``p_waic`` (totals) and ``lppd_rows``, ``p_waic_rows``,
        ``se``         standard error of ``total`` over rows, ``2 sqrt(N Var_i elpd_i)``,
        ``n_p_high``   rows with ``p_waic_i > 0.4`` (WAIC unreliable for them),
        ``mc_se``      Monte-Carlo standard error of ``total`` from the finite number
                       of draws (delete-a-group jackknife over ``mc_groups`` groups of
                       draws; the draws must be independent),
        ``n_draws``.

    The Monte-Carlo error is not small: draws of the globals are shared by every
    row, so their sampling error adds up linearly over rows (validation/a4_waic).
    """
    ll = torch.as_tensor(log_likelihood, dtype=torch.float64)
    if ll.dim() != 2 or ll.shape[1] < 2:
        raise ValueError(f"log_likelihood must be (N rows, S >= 2 draws), got {tuple(ll.shape)}")
    if not torch.all(torch.isfinite(ll)):
        raise ValueError("log_likelihood contains non-finite values.")
    n_rows, n_draws = ll.shape
    lppd_rows = torch.logsumexp(ll, dim=1) - math.log(n_draws)
    p_rows = ll.var(dim=1, unbiased=True)
    elpd_rows = lppd_rows - p_rows
    se = 2.0 * math.sqrt(n_rows * float(elpd_rows.var(unbiased=True))) if n_rows > 1 else float("nan")
    groups = min(int(mc_groups), n_draws // 2)
    mc_se = float("nan")
    if groups >= 2:
        labels = torch.arange(n_draws) % groups
        partial = torch.tensor([_waic_total(ll[:, labels != k]) for k in range(groups)],
                               dtype=torch.float64)
        mc_se = math.sqrt((groups - 1) / groups * float(((partial - partial.mean()) ** 2).sum()))
    return dict(mc_se=mc_se, total=-2.0 * elpd_rows.sum(), per_row=-2.0 * elpd_rows,
                elpd=elpd_rows.sum(), elpd_rows=elpd_rows,
                lppd=lppd_rows.sum(), lppd_rows=lppd_rows,
                p_waic=p_rows.sum(), p_waic_rows=p_rows, se=se,
                n_p_high=int((p_rows > WAIC_P_WARNING).sum()), n_draws=n_draws)


def elpd_difference(elpd_rows_a, elpd_rows_b):
    """``elpd_a - elpd_b`` and its standard error ``sqrt(N Var_i(d_i))`` (paired over rows)."""
    d = (torch.as_tensor(elpd_rows_a, dtype=torch.float64)
         - torch.as_tensor(elpd_rows_b, dtype=torch.float64))
    se = math.sqrt(d.numel() * float(d.var(unbiased=True))) if d.numel() > 1 else float("nan")
    return float(d.sum()), se


def pseudo_bma_weights(elpd_rows, bootstrap=True, n_bootstrap=1000, seed=0):
    """Model weights from pointwise elpd values ``(M models, N rows)``.

    ``bootstrap=False`` is ``softmax(elpd)``, identical to the Akaike-type weights
    ``softmax(-WAIC / 2)``. ``bootstrap=True`` is pseudo-BMA+ (Yao, Vehtari, Simpson
    & Gelman 2018): the softmax averaged over Bayesian-bootstrap (Dirichlet(1))
    reweightings of the rows, which accounts for the uncertainty of the elpd
    estimate and keeps weights away from 0/1 when the evidence is thin.
    """
    elpd = torch.as_tensor(elpd_rows, dtype=torch.float64)
    if elpd.dim() != 2:
        raise ValueError("elpd_rows must be (M models, N rows).")
    if not bootstrap:
        return torch.softmax(elpd.sum(dim=1), dim=0)
    n_rows = elpd.shape[1]
    generator = torch.Generator().manual_seed(int(seed))
    gamma = -torch.log(torch.rand(int(n_bootstrap), n_rows, generator=generator,
                                  dtype=torch.float64))
    alpha = gamma / gamma.sum(dim=1, keepdim=True)              # Dirichlet(1, ..., 1)
    z = n_rows * alpha @ elpd.T                                  # (B, M)
    return torch.softmax(z, dim=1).mean(dim=0)


# ---------------------------------------------------------------------------
# PSIS-LOO (Vehtari, Gelman & Gabry 2017; Vehtari et al. 2024, "Pareto smoothed
# importance sampling", JMLR)
# ---------------------------------------------------------------------------
# Leave-one-row-out: elpd_loo,i = log p(x_i | x_-i), estimated from full-posterior draws
# with importance weights 1 / p(x_i | theta_s), whose upper tail is replaced by a fitted
# generalized Pareto distribution. With locals in theta this predicts a *new unit*
# (the left-out row's locals revert to their prior), unlike the conditional WAIC.
# The Pareto shape k diagnoses each row: k > 0.7 means the estimate is unreliable.

PSIS_K_WARNING = 0.7


def _gpd_fit(exceedances):
    """Zhang & Stephens (2009) generalized-Pareto fit with the weakly informative prior
    on k of Vehtari et al.; ``exceedances`` sorted ascending. Returns ``(k, sigma)``."""
    x = np.asarray(exceedances, dtype=np.float64)
    n = len(x)
    m = 30 + int(math.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= 3.0 * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    profile = n * (np.log(-(b / k)) - k - 1.0)
    weights = 1.0 / np.exp(profile - profile[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps
    b, weights = b[keep], weights[keep] / weights[keep].sum()
    b_post = float((b * weights).sum())
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    k_post = (n * k_post + 10 * 0.5) / (n + 10)
    return k_post, sigma


def _gpd_quantile(p, k, sigma):
    if abs(k) < np.finfo(float).eps:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psis_log_weights(log_ratios):
    """Pareto-smoothed normalized log weights for one row; returns ``(log_w, k)``."""
    lr = np.asarray(log_ratios, dtype=np.float64)
    n = len(lr)
    lr = lr - lr.max()
    tail_len = int(math.ceil(min(0.2 * n, 3.0 * math.sqrt(n))))
    order = np.argsort(lr)
    cutoff = max(lr[order[-tail_len - 1]], math.log(np.finfo(float).tiny))
    tail = np.where(lr > cutoff)[0]
    k = math.inf
    if len(tail) > 4:
        tail = tail[np.argsort(lr[tail])]
        exceed = np.exp(lr[tail]) - math.exp(cutoff)
        k, sigma = _gpd_fit(exceed)
        if math.isfinite(k):
            p = (np.arange(len(tail)) + 0.5) / len(tail)
            lr = lr.copy()
            lr[tail] = np.log(_gpd_quantile(p, k, sigma) + math.exp(cutoff))
            lr = np.minimum(lr, 0.0)                    # truncate at the largest raw ratio
    return lr - _logsumexp_np(lr), k


def _logsumexp_np(a):
    top = a.max()
    return top + math.log(np.exp(a - top).sum())


def _loo_rows(ll):
    lppd = np.array([_logsumexp_np(r) for r in ll]) - math.log(ll.shape[1])
    elpd, ks = np.empty(ll.shape[0]), np.empty(ll.shape[0])
    for i, row in enumerate(ll):
        log_w, ks[i] = psis_log_weights(-row)
        elpd[i] = _logsumexp_np(log_w + row)
    return elpd, lppd, ks


def psis_loo(log_likelihood, mc_groups=10):
    """PSIS-LOO of one model from its ``(N, S)`` pointwise log-likelihood matrix.

    Same keys as :func:`waic` where they apply (``total`` = LOOIC = ``-2 elpd_loo``,
    ``per_row``, ``elpd``, ``elpd_rows``, ``lppd``, ``se``, ``mc_se``, ``n_draws``),
    plus ``p_loo`` (``lppd - elpd_loo``), ``pareto_k`` (``(N,)``) and ``n_k_high``
    (rows with k > 0.7, whose estimate is unreliable).
    """
    ll = torch.as_tensor(log_likelihood, dtype=torch.float64)
    if ll.dim() != 2 or ll.shape[1] < 10:
        raise ValueError(f"log_likelihood must be (N rows, S >= 10 draws), got {tuple(ll.shape)}")
    if not torch.all(torch.isfinite(ll)):
        raise ValueError("log_likelihood contains non-finite values.")
    arr = ll.numpy()
    n_rows, n_draws = arr.shape
    elpd, lppd, ks = _loo_rows(arr)
    se = 2.0 * math.sqrt(n_rows * float(np.var(elpd, ddof=1))) if n_rows > 1 else float("nan")
    groups = min(int(mc_groups), n_draws // 20)
    mc_se = float("nan")
    if groups >= 2:
        labels = np.arange(n_draws) % groups
        partial = np.array([-2.0 * _loo_rows(arr[:, labels != g])[0].sum() for g in range(groups)])
        mc_se = math.sqrt((groups - 1) / groups * float(((partial - partial.mean()) ** 2).sum()))
    elpd_t = torch.as_tensor(elpd)
    return dict(total=-2.0 * elpd_t.sum(), per_row=-2.0 * elpd_t, elpd=elpd_t.sum(),
                elpd_rows=elpd_t, lppd=float(lppd.sum()), p_loo=float(lppd.sum() - elpd.sum()),
                p_loo_rows=torch.as_tensor(lppd - elpd), pareto_k=torch.as_tensor(ks),
                n_k_high=int((ks > PSIS_K_WARNING).sum()), se=se, mc_se=mc_se,
                n_draws=n_draws)
