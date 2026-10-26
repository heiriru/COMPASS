"""Benchmark hook: run any experiment driver with the final method, or with one component of it
changed, without editing the driver. Python imports this module at start-up when its folder is on
PYTHONPATH (benchmarks/paper/queue.py puts it there for every job). With none of the variables
below set it does nothing.

Sampling (``MultiObsSampler.sample``, so every subclass and every driver):

    PAPER_SAMPLE_KW     JSON dict merged over the caller's keyword arguments, for calls whose
                        ``correction`` is in PAPER_APPLY_TO; ``null`` sets None. Example, the
                        library default refresh: {"jacobian_refresh": 9, "jacobian_refresh_low_noise": null}
    PAPER_APPLY_TO      comma list of corrections the override applies to
                        (default gauss_jacobian_FD,gauss_jacobian)
    PAPER_SAMPLER_SET   JSON {setter: [args]} called on the sampler before every sample(), e.g.
                        {"set_eigh_precision": ["float64", 2048], "set_schur_mode": ["cholesky"]}
    PAPER_NETCOV        1: use the network's covariance output in place of the score Jacobian
                        (the checkpoint must carry one); PAPER_NETCOV_LATENT / PAPER_NETCOV_OBSERVED
                        are comma lists of node indices

Training (``ScoreBasedInferenceModel.train``):

    PAPER_TRAIN_KW      JSON dict merged over the caller's keyword arguments, e.g.
                        {"time_loss_weight": null} or {"validation_noise_seed": null}
    PAPER_COV_TRAIN     joint | detached: attach a covariance output and train it by
                        second-order DSM (compass.covariance_output); PAPER_COV_RANK (default 8)

GPU:

    PAPER_GPU           the one GPU the queue holds: drivers that call autocvd themselves get this
                        device back instead of a search over all cards

Every applied override is printed once per process, prefixed [paper-hook].
"""
import json
import os
import sys

_KEYS = ("PAPER_SAMPLE_KW", "PAPER_SAMPLER_SET", "PAPER_NETCOV", "PAPER_TRAIN_KW", "PAPER_COV_TRAIN")

if os.environ.get("PAPER_GPU"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["PAPER_GPU"]
    os.environ["JACEXT_GPU"] = os.environ["PAPER_GPU"]
    try:
        import autocvd as _autocvd

        def _pinned(*args, **kwargs):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["PAPER_GPU"]
            return [int(os.environ["PAPER_GPU"])]

        _autocvd.autocvd = _pinned
    except ImportError:
        pass


def _say(message):
    print(f"[paper-hook] {message}", file=sys.stderr, flush=True)


def _install():
    import torch
    import compass  # noqa: F401
    sampler_module = sys.modules["compass.MultiObsSampler"]
    Sampler = sampler_module.MultiObsSampler
    from compass import ScoreBasedInferenceModel

    sample_kw = json.loads(os.environ.get("PAPER_SAMPLE_KW") or "{}")
    apply_to = set(os.environ.get("PAPER_APPLY_TO", "gauss_jacobian_FD,gauss_jacobian").split(","))
    setters = json.loads(os.environ.get("PAPER_SAMPLER_SET") or "{}")
    netcov = os.environ.get("PAPER_NETCOV") == "1"
    _say(f"compass from {compass.__file__}")

    def convert(value):
        if isinstance(value, str) and value in ("float32", "float64"):
            return getattr(torch, value)
        return value

    if sample_kw or setters or netcov:
        original_sample = Sampler.sample
        told = set()

        def sample(self, *args, **kwargs):
            if getattr(self, "_paper_hook_active", False):     # a subclass calling super().sample
                return original_sample(self, *args, **kwargs)
            correction = kwargs.get("correction", "gauss_jacobian_FD")
            for name, values in setters.items():
                getattr(self, name)(*[convert(v) for v in values])
            if sample_kw and correction in apply_to:
                kwargs.update(sample_kw)
            if netcov and correction.startswith("gauss_jacobian"):
                if getattr(self, "covariance_head", None) is None:
                    from compass.covariance_output import NetworkCovarianceHead
                    if getattr(self.SBIm, "covariance_output", None) is None:
                        raise RuntimeError("PAPER_NETCOV=1 but the network has no covariance output")
                    latent = [int(i) for i in os.environ["PAPER_NETCOV_LATENT"].split(",")]
                    observed = [int(i) for i in os.environ["PAPER_NETCOV_OBSERVED"].split(",")]
                    self.set_covariance_head(NetworkCovarianceHead(self.SBIm, latent, observed))
            key = (type(self).__name__, correction)
            if key not in told:
                told.add(key)
                _say(f"{type(self).__name__}.sample correction={correction}: setters {setters}, "
                     f"kwargs {sample_kw if correction in apply_to else {}}, netcov {netcov}")
            self._paper_hook_active = True
            try:
                return original_sample(self, *args, **kwargs)
            finally:
                self._paper_hook_active = False

        Sampler.sample = sample

    train_kw = json.loads(os.environ.get("PAPER_TRAIN_KW") or "{}")
    cov_mode = os.environ.get("PAPER_COV_TRAIN", "")
    if train_kw or cov_mode:
        original_train = ScoreBasedInferenceModel.train

        def train(self, *args, **kwargs):
            kwargs.update(train_kw)
            if cov_mode:
                from compass.covariance_output import CovarianceObjective, enable_covariance_output
                head = enable_covariance_output(self, rank=int(os.environ.get("PAPER_COV_RANK", "8")))
                kwargs["covariance_objective"] = CovarianceObjective(
                    head, weight=1.0, detach_trunk=cov_mode == "detached")
            _say(f"train: kwargs override {train_kw}, covariance output {cov_mode or 'off'}")
            return original_train(self, *args, **kwargs)

        ScoreBasedInferenceModel.train = train


if any(os.environ.get(k) for k in _KEYS):
    _install()
