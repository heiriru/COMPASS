# E04 — VESDE vs VPSDE

**Paper:** §4.1 and A3.1; Fig. A5 (VESDE) against Fig. A6 (VPSDE).

## What they did

Ran the whole benchmark under both SDEs with `σ_max = 15, σ_min = 1e-4` and
`β ∈ [0.01, 10]` on `t ∈ [1e-5, 1]`. Finding: broadly comparable, with VESDE
better on Two Moons and VPSDE slightly better on SLCP. Fig. A7 adds that Two
Moons under VPSDE is the one case where 50 sampling steps are *not* enough.

## What we do

No separate driver: this is e01 and e05 run with `--sde vpsde`. COMPASS
implements both (`src/compass/SDE.py`), and `common/model.py` maps the paper's
constants onto the constructor.

```bash
for SDE in vesde vpsde; do
  python e01_benchmark_posterior/run.py train --task two_moons \
      --arm paper_matched --budget 10000 --sde $SDE --device cuda
done
```

## Watch out for

- **COMPASS's `sigma` is the VESDE `σ_max`.** The library default is 25.0; the
  paper's 15.0 is what `common/model.py` passes. Do not leave it at the default
  "because it is the default" — the noise ceiling relative to the data scale is
  exactly what a VESDE gets wrong when mismatched, and `inverse_kinematics`
  already had to drop it to 8.0 for a problem of similar scale.
- **`eps`.** Their time interval starts at 1e-5; COMPASS's samplers default to
  `eps=1e-3`. `common/model.py::sampling_kwargs` pins the paper's value.
