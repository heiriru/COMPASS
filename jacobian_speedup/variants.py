"""Variant specs shared by the speed-up benches (see bench_toy.py)."""


def parse(spec):
    """'r1' | 'legacy' | 'r5' | 'r5s' | 'lo10@0.3' | 'hi20@3' | 'band0.05-20' | 'warm0.5', joined by '+'.

    r<k>        refresh period k in the band (default everywhere); trailing 's' adds secant updates
    lo<k>@<l>   period k below lambda l       hi<k>@<l>  period k above lambda l
    band<a>-<b> adaptive band in r = lambda^2 K; outside: never (or 'band<a>-<b>/<k>': every k)
    legacy      = r1 + lo10@0.3 (the toy benchmarks' reference schedule)
    prod        = r1 + lo20@0.3 (what pp.py's gauss_jacobian_FD, i.e. the 0.928 run, uses)
    warm<l>     Laplace warm start at lambda l
    """
    kwargs, warm = {"jacobian_refresh": 1}, None
    for part in spec.split("+"):
        if part == "legacy":
            kwargs.update(jacobian_refresh=1, jacobian_refresh_low_noise=10,
                          jacobian_refresh_lambda=0.3)
        elif part == "prod":
            kwargs.update(jacobian_refresh=1, jacobian_refresh_low_noise=20,
                          jacobian_refresh_lambda=0.3)
        elif part.startswith("r"):
            secant = part.endswith("s")
            kwargs["jacobian_refresh"] = int(part[1:-1] if secant else part[1:])
            kwargs["jacobian_secant"] = secant
        elif part.startswith("lo"):
            k, lam = part[2:].split("@")
            kwargs.update(jacobian_refresh_low_noise=int(k), jacobian_refresh_lambda=float(lam))
        elif part.startswith("hi"):
            k, lam = part[2:].split("@")
            kwargs.update(jacobian_refresh_high_noise=int(k), jacobian_refresh_high_lambda=float(lam))
        elif part.startswith("band"):
            body, _, outside = part[4:].partition("/")
            low, high = body.split("-")
            kwargs.update(jacobian_refresh_band=(float(low), float(high)),
                          jacobian_refresh_outside_band=int(outside) if outside else None)
        elif part.startswith("warm"):
            warm = float(part[4:])
        elif part == "secant":
            kwargs["jacobian_secant"] = True
        else:
            raise ValueError(f"unknown variant part {part!r}")
    return kwargs, warm
