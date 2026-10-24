"""Opt-in network output parameterisations (roadmap 2.10; experimental, removable).

COMPASS reads a score as ``score = raw / sigma_t`` (``output_scale_function``), so
the network ``raw`` output is an epsilon prediction: ``raw ~ -E[eps | x_t]``.
``parameterization="edm"`` instead builds the epsilon output from a denoiser in the
EDM form (Karras et al. 2022), written in the ``y = x / alpha`` coordinates with
``lambda = sigma / alpha``::

    D(y)  = c_skip y + c_out F(c_in y, t, c)
    c_skip = s_d^2 / (lambda^2 + s_d^2),   c_out = lambda s_d / sqrt(lambda^2 + s_d^2),
    c_in   = 1 / sqrt(lambda^2 + s_d^2)
    raw   = (D - y) / lambda = -lambda y / (lambda^2 + s_d^2) + s_d / sqrt(lambda^2 + s_d^2) F

``-lambda y / (lambda^2 + s_d^2)`` is exactly the raw output of an ``N(0, s_d^2)``
prior, so the network learns a residual on that baseline, and its latent inputs are
normalised to unit scale at every noise level instead of growing to ``sigma_max``.
Conditioned coordinates are neither rescaled nor transformed (their output is
masked everywhere). With ``time_loss_weight="edm"`` the epsilon loss becomes EDM's
uniform weighting of ``F``.

It is installed as a pair of forward hooks on the network, exactly like the
log-lambda time map: no parameters, no state-dict keys, no call site changes, so
every sampler, the PF-ODE and the Jacobian rules see the same transformed output.
The checkpoint records it.
"""

import torch

PARAMETERIZATIONS = ("eps", "edm")


def install(model, parameterization="eps", sigma_data=1.0):
    """Install ``parameterization`` on ``model`` (a ScoreBasedInferenceModel)."""
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(f"parameterization must be one of {PARAMETERIZATIONS}, "
                         f"got {parameterization!r}")
    for handle in getattr(model, "_parameterization_handles", ()):
        handle.remove()
    model._parameterization_handles = ()
    model.parameterization = parameterization
    model.parameterization_sigma_data = float(sigma_data)
    if parameterization == "eps":
        return model
    sde = model.sde
    s_d = float(sigma_data)
    state = {}

    def unpack(args, kwargs):
        x = kwargs["x"] if "x" in kwargs else args[0]
        t = kwargs["t"] if "t" in kwargs else args[1]
        c = kwargs["c"] if "c" in kwargs else args[2]
        return x, t, c

    def pre_hook(module, args, kwargs):
        x, t, c = unpack(args, kwargs)
        t_flat = torch.as_tensor(t, device=x.device).reshape(-1, 1).to(x.dtype)
        if t_flat.shape[0] == 1 and x.shape[0] > 1:
            t_flat = t_flat.expand(x.shape[0], 1)
        lam = sde.lambda_t(t_flat).to(x.dtype)
        alpha = sde.alpha_t(t_flat).to(x.dtype)
        latent = 1.0 - torch.as_tensor(c, device=x.device).to(x.dtype)
        y = x / alpha
        state["lam"], state["y"], state["latent"] = lam, y, latent
        c_in = torch.rsqrt(lam.square() + s_d ** 2)
        x_in = latent * (y * c_in) + (1.0 - latent) * x
        if "x" in kwargs:
            kwargs = dict(kwargs)
            kwargs["x"] = x_in
        else:
            args = (x_in,) + tuple(args[1:])
        return args, kwargs

    def post_hook(module, args, kwargs, output):
        attention = None
        if isinstance(output, tuple):
            output, attention = output[0], output[1:]
        lam, y, latent = state.pop("lam"), state.pop("y"), state.pop("latent")
        denominator = lam.square() + s_d ** 2
        raw = -lam * y / denominator + s_d * torch.rsqrt(denominator) * output
        output = latent * raw + (1.0 - latent) * output
        return (output,) + attention if attention is not None else output

    network = model.model
    # prepend: run before the log-lambda time map rewrites t.
    pre = network.register_forward_pre_hook(pre_hook, with_kwargs=True, prepend=True)
    post = network.register_forward_hook(post_hook, with_kwargs=True)
    model._parameterization_handles = (pre, post)
    return model


def payload(model):
    """Checkpoint keys; empty for the default so old checkpoints stay byte-identical."""
    mode = getattr(model, "parameterization", "eps")
    if mode == "eps":
        return {}
    return {"parameterization": mode,
            "parameterization_sigma_data": model.parameterization_sigma_data}
