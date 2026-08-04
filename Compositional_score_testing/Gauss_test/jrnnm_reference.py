"""Adapter for the exact JR-NMM simulator used by the paper.

The simulator lives in the R package ``sdbmsABC`` and is intentionally imported
only when Figure 4 data preparation is requested.  This keeps Figures 1--3 free
of the optional R/rpy2 dependency.
"""
from __future__ import annotations

import numpy as np
import torch


def simulate_exact_jrnnm(task, z: torch.Tensor) -> torch.Tensor:
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
    except ImportError as error:
        raise RuntimeError(
            "Exact Figure 4 preparation requires rpy2 and the R package sdbmsABC; "
            "see Gauss_test/README.md. Use --jrnnm-backend torch for the included "
            "Euler-Maruyama fallback."
        ) from error

    package = importr("sdbmsABC")
    r_chol = robjects.r["chol"]
    r_transpose = robjects.r["t"]
    theta = task.theta_from_z(z).detach().cpu().double().numpy()
    dt = 1.0 / 1024.0
    grid = robjects.FloatVector(list(np.arange(0.0, 10.0, dt)))
    matrix_one = package.exp_matJR(dt, 100.0, 50.0)
    summaries = []
    for c, mu, sigma in theta:
        matrix_two = r_transpose(r_chol(package.cov_matJR(
            dt, robjects.FloatVector([0.0, 0.0, 0.0, 0.01, float(sigma), 1.0]),
            100.0, 50.0,
        )))
        signal = np.asarray(package.Splitting_JRNMM_output_Cpp(
            dt,
            robjects.FloatVector(list(np.random.randn(6))),
            grid,
            matrix_one,
            matrix_two,
            float(mu),
            float(c),
            3.25,
            22.0,
            100.0,
            50.0,
            6.0,
            0.56,
            5.0,
        ))
        signal = signal[2 * 1024::8]
        signal = signal[:1024] - signal[:1024].mean()
        summaries.append(task._log_psd(torch.from_numpy(signal).float()[None])[0])
    return torch.stack(summaries)
