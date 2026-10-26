"""The paper benchmark: every job, in priority order (README.md has the design and the reasons).

A job is a dict: name, cwd (relative to the benchmark tree), cmd ("PY" = the COMPASS venv python),
env (PAPER_* overrides of benchmarks/paper/hook, COMPASS_* library switches, driver variables),
pythonpath (tree-relative), need (expected peak GPU MB), after (job names that must finish first).
"""
import json
from pathlib import Path

OUT = Path("/export/data/rheinric/paper_bench")
TREE = OUT / "tree"                       # detached worktree at the benchmarked commit + data copies
PY = "/export/home/rheinric/COMPASS/.COMPASS/bin/python"

JAC, GH, AD = "gauss_jacobian_FD", "gauss_hierarchical", "arruda_direct"

# ---------------------------------------------------------------------------- the final method
# Sampling: the library defaults (gauss_jacobian_FD, refresh 9 at every noise level, float32 eigh in
# 1,024-row chunks, compact state, elimination Schur, row projection + composed repair, coordinate
# clamp). The drivers pin older refresh schedules for their archived results, so the final method
# overrides exactly those two arguments back to the library default.
FINAL_KW = {"jacobian_refresh": 9, "jacobian_refresh_low_noise": None}
FINAL = {"PAPER_SAMPLE_KW": FINAL_KW}

# One component changed at a time, everything else final. name -> extra environment.
SAMPLER_VARIANTS = {
    "final": {},
    "r1": {"PAPER_SAMPLE_KW": {"jacobian_refresh": 1, "jacobian_refresh_low_noise": None}},
    "fp64": {"PAPER_SAMPLER_SET": {"set_eigh_precision": ["float64", 2048]}},
    "dense": {"COMPASS_STATE_LAYOUT": "dense", "COMPASS_SCHUR_MODE": "cholesky"},
    "noproj": {"COMPASS_INFORMATION_PROJECTION": "0"},
    "nocurv": {"COMPASS_CURVATURE_CLIP": "0"},
    "consistent": {"COMPASS_PD_STRATEGY": "row_projection_consistent"},
    "gate": {"COMPASS_PD_STRATEGY": "row_projection_consistent", "COMPASS_PD_CONSISTENT_BELOW": "0.5",
             "COMPASS_PD_CONSISTENT_GUARD": "fallback"},
    "gloeckler": {"COMPASS_PD_STRATEGY": "gloeckler"},
    "metric": {"COMPASS_CLAMP_MODE": "metric"},
}
# the PD / clamp / projection subset that is cheap enough to run on every benchmark
SAFEGUARD_VARIANTS = ("final", "noproj", "consistent", "gate", "gloeckler", "metric")


def env_of(variant, base=None):
    env = json.loads(json.dumps(FINAL))
    env.update(base or {})
    extra = dict(SAMPLER_VARIANTS[variant])
    if "PAPER_SAMPLE_KW" in extra:
        env["PAPER_SAMPLE_KW"] = {**env["PAPER_SAMPLE_KW"], **extra.pop("PAPER_SAMPLE_KW")}
    env.update(extra)
    return env


def job(name, cwd, cmd, env=None, need=2000, after=(), pythonpath=()):
    env = json.loads(json.dumps(env or {}))      # a private copy: jobs() serialises it in place
    return dict(name=name, cwd=cwd, cmd=["PY", "-u", *cmd], env=env, need=need,
                after=list(after), pythonpath=list(pythonpath))


# ---------------------------------------------------------------------------- B1: BayesFlow joint net
FAIR = "Arruda/partial_pooling_fair"
FAIR_ENV = {"KERAS_BACKEND": "torch", "FAIR_POSTERIORS": str(OUT / "fair" / "posteriors")}
FAIR_PATH = ("Arruda/.vendor", ".")


def fair(name, rule, net, n=100, env=None, need=3500):
    return job(f"fair_{name}", FAIR, ["fair.py", "infer", rule, net, str(n)],
               {**FAIR_ENV, **(env or {})}, need=need, pythonpath=FAIR_PATH)


def fair_jobs():
    out = []
    for net in ("joint_wide", "joint"):
        for n in (100, 25, 4):
            out.append(fair(f"final_{net}_n{n}", "paper-final", net, n, FINAL))
    # upstream's own compositional sampler on its global net (no COMPASS code involved); the
    # inference-prior n = 4 / 25 / 100 and wide n = 100 arms are copied from the archive
    for n in (25, 4):
        out.append(fair(f"bf_global_wide_n{n}", "bf", "global_wide", n))
    # Monte Carlo replicates of the final arm (inference seed only)
    for k in (1, 2):
        out.append(fair(f"final_joint_wide_n100_seed{k}", f"paper-final-seed{k}", "joint_wide", 100,
                        {**FINAL, "FAIR_SEED_OFFSET": str(1000 * k)}))
    for variant in SAMPLER_VARIANTS:
        if variant == "final":
            continue
        out.append(fair(f"{variant}_joint_wide_n100", f"paper-{variant}", "joint_wide", 100, env_of(variant)))
    for variant in ("r1", "noproj", "consistent", "metric"):
        out.append(fair(f"{variant}_joint_n100", f"paper-{variant}", "joint", 100, env_of(variant)))
    # solver settings: no churn, and 25 / 100 steps instead of 50
    out.append(fair("churn0_joint_wide_n100", "paper-churn0", "joint_wide", 100,
                    {**FINAL, "COMPASS_SAMPLER_OVERRIDE": {"churn_eta": 0.0}}))
    for steps in (25, 100):
        out.append(fair(f"t{steps}_joint_wide_n100", f"paper-t{steps}", "joint_wide", 100,
                        {"PAPER_SAMPLE_KW": {**FINAL_KW, "timesteps": steps}}))
    # even refresh periods (DPM2 makes two calls per step: an even period never refreshes at the
    # second-order stage) and a longer odd one
    for period in (5, 10, 19):
        out.append(fair(f"r{period}_joint_wide_n100", f"paper-r{period}", "joint_wide", 100,
                        {"PAPER_SAMPLE_KW": {"jacobian_refresh": period, "jacobian_refresh_low_noise": None}}))
    return out


# ---------------------------------------------------------------------------- B2: partial pooling, COMPASS net
EXP = "experiments_arruda_schur"
ENCODER = "artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt"
PP_NETCOV = {"PAPER_NETCOV": "1", "PAPER_NETCOV_LATENT": ",".join(map(str, range(10))),
             "PAPER_NETCOV_OBSERVED": ",".join(map(str, range(10, 26)))}


def pp_common(precondition="Rg", wide=1.5, mode="hierarchical"):
    out = ["--mode", mode]
    if precondition:
        out += ["--precondition", precondition]
    if wide != 1.0:
        out += ["--train-prior-scale", str(wide)]
    return out + ["--encoder", ENCODER]


def pp_infer(name, variant, tag, method="gauss_jacobian_FD_r9", env=None, datasets=30, subjects=100,
             checkpoint="final", after=(), need=4500, timesteps=50, **arch):
    cmd = ["pp.py", "infer", *pp_common(**arch), "--subjects", str(subjects), "--datasets", str(datasets),
           "--dataset-start", "0", "--draws", "100", "--timesteps", str(timesteps), "--method", method,
           "--variant", variant, "--checkpoint", checkpoint, "--tag", tag]
    return job(f"pp_{name}", EXP, cmd, env, need=need, after=after)


def pp_train(arm, seed, env=None, tsm=False, **arch):
    cmd = ["pp.py", "train", *pp_common(**arch), "--train-subset", "65536",
           "--variant", f"paper_{arm}_s{seed}", "--train-seed", str(seed)] + (["--tsm"] if tsm else [])
    return job(f"pp_train_{arm}_s{seed}", EXP, cmd, {"PAPER_COV_TRAIN": "joint", **(env or {})}, need=3000)


def pp_jobs():
    out = []
    # the final networks: covout_cov_s7 / _s8 (trained on this branch's recipe with the covariance
    # output), and a third seed trained here
    out.append(pp_train("final", 9))
    finals = {7: "covout_cov_s7", 8: "covout_cov_s8", 9: "paper_final_s9"}
    for seed, variant in finals.items():
        after = ["pp_train_final_s9"] if seed == 9 else []
        out.append(pp_infer(f"final_s{seed}_d100", variant, "paper_final", env=FINAL, datasets=100, after=after))
        out.append(pp_infer(f"netcov_s{seed}_d100", variant, "paper_netcov_r9", env={**FINAL, **PP_NETCOV},
                            datasets=100, after=after))
    # sampler ablations on the seed-7 final net (30 datasets: the paired comparison)
    net = finals[7]
    for variant in SAMPLER_VARIANTS:
        if variant != "final":
            out.append(pp_infer(f"{variant}_s7", net, f"paper_{variant}", env=env_of(variant)))
    out.append(pp_infer("netcov_r1_s7", net, "paper_netcov_r1", method="gauss_jacobian_FD_r1",
                        env={**PP_NETCOV}))
    out.append(pp_infer("gh_s7", net, "paper_gh", method=GH))
    out.append(pp_infer("arruda_s7", net, "paper_arruda", method=AD))
    out.append(pp_infer("fnpse_s7", net, "paper_fnpse", method="langevin_fnpse"))
    out.append(pp_infer("churn0_s7", net, "paper_churn0", env={**FINAL, "COMPASS_SAMPLER_OVERRIDE": {"churn_eta": 0.0}}))
    for steps in (25, 100):
        out.append(pp_infer(f"t{steps}_s7", net, f"paper_t{steps}", env=FINAL, timesteps=steps))
    out.append(pp_infer("bestval_s7", net, "paper_bestval", env=FINAL, checkpoint="best"))
    for n in (10, 25):
        out.append(pp_infer(f"n{n}_s7", net, f"paper_n{n}", env=FINAL, subjects=n))
    # training ablations (one component of the final recipe changed; two seeds for the three
    # training-procedure choices, one for the architecture ones whose effect is large)
    ablations = {
        "nocov": dict(env={"PAPER_COV_TRAIN": ""}),
        "noedm": dict(env={"PAPER_TRAIN_KW": {"time_loss_weight": None}}),
        "unseeded": dict(env={"PAPER_TRAIN_KW": {"validation_noise_seed": None}}),
        "tsm": dict(tsm=True),
        "norg": dict(arch={"precondition": ""}),
        "nowide": dict(arch={"wide": 1.0}),
        "full": dict(arch={"mode": "full"}),
    }
    for arm, spec in ablations.items():
        seeds = (7, 8) if arm in ("noedm", "unseeded", "tsm", "nocov") else (7,)
        arch = spec.get("arch", {})
        for seed in seeds:
            train = pp_train(arm, seed, env=spec.get("env"), tsm=spec.get("tsm", False), **arch)
            if arm == "nocov":
                train["env"]["PAPER_COV_TRAIN"] = ""
            out.append(train)
            out.append(pp_infer(f"{arm}_s{seed}", f"paper_{arm}_s{seed}", "paper_final", env=FINAL,
                                after=[train["name"]], **arch))
        if arm == "nocov":
            continue
    # seed-8 sampler cross-check of the two closest PD choices
    for variant in ("consistent", "gate"):
        out.append(pp_infer(f"{variant}_s8", finals[8], f"paper_{variant}", env=env_of(variant)))
    return out


# ---------------------------------------------------------------------------- B3: exact-reference problems
TOYS = "abs_hierarchy"
TOY_PATH = ("abs_hierarchy", "Gaussian_mixtures_compositional_score_modelling", "experiments_arruda_schur")
TOY_CASES = [("abs", "hierarchical", [JAC, GH, AD]), ("abs", "oracle_joint", [JAC, GH]),
             ("lean", "hierarchical", [JAC, GH]), ("lean", "oracle_joint", [JAC, GH]),
             ("square", "hierarchical", [JAC, GH]), ("curved", "hierarchical", [JAC, GH])]


def toy_jobs():
    out = []
    for model, mode, methods in TOY_CASES:
        for variant in SAFEGUARD_VARIANTS + ("r1",):
            ms = methods if variant == "final" else [JAC]
            out.append(job(f"toy_{model}_{mode}_{variant}", TOYS,
                           ["run.py", "--model", model, "--mode", mode, "--methods", *ms,
                            "--suffix", f"_paper_{variant}"], env_of(variant), need=1800, pythonpath=TOY_PATH))
    return out


IK_CASES = [("reach_sym", ["ik_reach.py", "run", "--variant", "sym"], {}, [JAC, AD]),
            ("reach_shift", ["ik_reach.py", "run", "--variant", "shift"], {}, [JAC, AD]),
            ("reach_sym_bernoulli", ["ik_reach.py", "run", "--variant", "sym", "--menu", "bernoulli"], {}, [JAC]),
            ("flat_global", ["ik_flat.py", "run", "--variant", "global"], {"IK_SD_H": "0.05"}, [JAC]),
            ("flat_fixed", ["ik_flat.py", "run", "--variant", "fixed"], {}, [JAC]),
            ("absx1", ["ik_absx1.py", "run"], {}, [JAC])]


def ik_jobs():
    out = []
    for name, cmd, extra, methods in IK_CASES:
        for variant in SAFEGUARD_VARIANTS + ("r1",):
            ms = methods if variant == "final" else [JAC]
            out.append(job(f"ik_{name}_{variant}", EXP, cmd + ["--methods", *ms, "--suffix", f"_paper_{variant}"],
                           {**env_of(variant), **extra}, need=1800))
    # the final recipe on IK reach: library training defaults + covariance output, final weights
    for v in ("sym", "shift"):
        train = job(f"ik_train_reach_{v}", EXP, ["ik_reach.py", "train", "--variant", v, "--model-suffix", "_paper"],
                    {"PAPER_COV_TRAIN": "joint", "PAPER_COV_RANK": "3"}, need=2500)
        out.append(train)
        for label, extra in (("final", {}), ("netcov", {"PAPER_NETCOV": "1", "PAPER_NETCOV_LATENT": "0,1,2",
                                                         "PAPER_NETCOV_OBSERVED": "3"})):
            methods = [JAC, AD] if label == "final" else [JAC]
            out.append(job(f"ik_reach_{v}_recipe_{label}", EXP,
                           ["ik_reach.py", "run", "--variant", v, "--model-suffix", "_paper", "--weights", "final",
                            "--methods", *methods, "--suffix", f"_paper_recipe_{label}"],
                           {**FINAL, **extra}, need=1800, after=[train["name"]]))
    return out


GM_LATENT_ARGS = ["--", "--observations", "8", "16", "32", "64", "--regimes", "asymmetric", "symmetric",
                  "--replicates", "1", "--draws", "2000"]


def gm_jobs():
    out = []
    # the latent-attention GM benchmark (experiments_arruda_schur/gm.py): trained hierarchical net and
    # the exact-score oracle, every rule; the safeguard variants on the trained net
    for mode, methods in (("hierarchical", [JAC, "arruda_gauss_direct", AD]), ("oracle", [JAC, AD])):
        for variant in (SAFEGUARD_VARIANTS if mode == "hierarchical" else ("final", "consistent", "metric")):
            ms = methods if variant == "final" else [JAC]
            out.append(job(f"gm_{mode}_{variant}", EXP, ["gm.py", "run", "--mode", mode, "--tag",
                                                          f"posteriors_paper_{variant}", *GM_LATENT_ARGS,
                                                          "--methods", *ms], env_of(variant), need=2200))
    # roadmap-2 GM (exact MC reference): the covariance-output nets (final recipe) with the Gaussian
    # prior interface and the mixture prior's score, Jacobian vs network covariance vs pilot vs F-NPSE
    gm_env = {"COMPASS_SRC": str(TREE / "src"), "ROADMAP_OUT": str(OUT / "gm"),
              "GM_DIR": str(TREE / "Gaussian_mixtures_compositional_score_modelling")}
    methods = ["gauss_jacobian_fd", "gauss_jacobian_mixture_prior_fd", "gauss_jacobian_netcov",
               "gauss_jacobian_mixture_prior_netcov", "gauss_hierarchical", "gauss_hierarchical_mixture_prior",
               "langevin_fnpse"]
    for arm, seeds, eps in (("cov", (7, 8), "1e-3"), ("cov_lo", (7, 8), "1e-5")):
        for seed in seeds:
            after = []
            if arm == "cov_lo":
                train = job(f"gm_train_{arm}_s{seed}", "experiments_roadmap2",
                            ["gm.py", "train", "--arm", arm, "--seed", str(seed)], gm_env, need=2500)
                out.append(train)
                after = [train["name"]]
            for refresh in ("1", "9"):
                ms = methods if refresh == "9" else ["gauss_jacobian_netcov", "gauss_jacobian_mixture_prior_netcov"]
                out.append(job(f"gm_{arm}_s{seed}_netcovr{refresh}", "experiments_roadmap2",
                               ["gm.py", "compose", "--arm", arm, "--seed", str(seed), "--sample-eps", eps,
                                "--observations", "1", "8", "16", "32", "64", "--replicates", "2",
                                "--draws", "2000", "--methods", *ms, "--tag-suffix", f"_paper_r{refresh}"],
                               {**gm_env, "GM_NETCOV_REFRESH": refresh}, need=5000, after=after))
    return out


def toy_gauss_jobs():
    env = {"COVOUT_TOY": str(OUT / "toy"), "NEED_MB": "3000"}
    return [job("gausstoy_cov_small", "covariance_output",
                ["toy.py", "run", "--arm", "cov", "--observations", "10", "100", "--datasets", "5",
                 "--methods", "jac_r9", "netcov_r9", "gh"], env, need=4000),
            job("gausstoy_cov_1000", "covariance_output",
                ["toy.py", "run", "--arm", "cov", "--observations", "1000", "--datasets", "3",
                 "--methods", "jac_r9", "netcov_r9"], env, need=6000, after=["gausstoy_cov_small"])]


def ar1_jobs():
    return [job("ar1_final_small", "Arruda/paper_benchmarks",
                ["ar1_grid.py", "--grids", "4", "16", "--methods", "paper_final", "--datasets", "10"], need=5000),
            job("ar1_final_64", "Arruda/paper_benchmarks",
                ["ar1_grid.py", "--grids", "64", "--methods", "paper_final", "--datasets", "3"], need=7000,
                after=["ar1_final_small"])]


def jobs():
    """Priority order: the final method everywhere first, then the sampler ablations, then the
    training ablations (the long ones), then the expensive scaling checks."""
    every = fair_jobs() + pp_jobs() + toy_jobs() + ik_jobs() + gm_jobs() + toy_gauss_jobs() + ar1_jobs()
    rank = lambda j: (0 if "final" in j["name"] and "train" not in j["name"] else  # noqa: E731
                      1 if j["name"].startswith(("fair_", "toy_", "ik_", "gm_")) and "train" not in j["name"] else
                      2 if "train" not in j["name"] else 1)
    for j in every:
        for key, value in j["env"].items():
            if not isinstance(value, str):
                j["env"][key] = json.dumps(value)
    return sorted(every, key=rank)
