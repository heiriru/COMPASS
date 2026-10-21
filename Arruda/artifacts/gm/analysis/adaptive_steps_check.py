"""Does BayesFlow's adaptive integrator reach t = 0 within max_steps, or does it hit the cap and
finish with one big final step? Wraps keras.ops.while_loop to report the loop's final counter and time."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
src = (HERE / "hier_improve.py").read_text().split("rows, panels = [], {}")[0]   # setup + score functions only
exec(compile(src, "hier_improve_setup", "exec"))
import keras

_original_while = keras.ops.while_loop
LOG = []


def logging_while(cond, body, loop_vars, *a, **k):
    out = _original_while(cond, body, loop_vars, *a, **k)
    if isinstance(out, (tuple, list)) and len(out) == 5:          # the adaptive loop's (i, state, time, step, last)
        LOG.append((int(ops.convert_to_numpy(out[0])), float(np.asarray(ops.convert_to_numpy(out[2])).ravel()[0]),
                    float(np.asarray(ops.convert_to_numpy(out[3])).ravel()[0])))
    return out


keras.ops.while_loop = logging_while
for local in ("learned", "exact"):
    network.score = learned_score if local == "learned" else exact_local_score
    for n in (8, 32, 128):
        x, *_ = R.gm_tall_problem(n, SEED, mixture, "hierarchical", G_TRUE)
        for max_steps in (1_000, 10_000):
            LOG.clear()
            approximator.compositional_sample(num_samples=1_000, conditions={"x": x[None]}, compute_prior_score=prior_fn,
                                              method="two_step_adaptive", steps="adaptive", mini_batch_size=10,
                                              max_steps=max_steps, compositional_bridge_d0=1.0, compositional_bridge_d1=1.0,
                                              seed=SEED)
            steps, t_end, last_dt = LOG[-1]
            print(f"{local:7s} n={n:3d} max_steps={max_steps:5d}: loop ran {steps} steps, stopped at t={t_end:.4f} "
                  f"(last dt {last_dt:+.5f}); {'HIT CAP -> one final jump of dt=' + format(-t_end, '.4f') if steps >= max_steps else 'reached the end'}",
                  flush=True)
