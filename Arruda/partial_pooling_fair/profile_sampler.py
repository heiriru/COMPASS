import sys, time, functools, collections
ARGS = sys.argv[1:]; sys.argv = ["fair.py"]
sys.path.insert(0, "/export/home/rheinric/COMPASS/Arruda/partial_pooling_fair")
import fair, torch
from compass import MultiObsSampler as _M
M = sys.modules["compass.MultiObsSampler"].MultiObsSampler
D = int(ARGS[0]) if ARGS else 2
net, rule = "joint", (ARGS[1] if len(ARGS) > 1 else "ours-batched-r9")
test = fair.load("test"); wf = fair.composed(net)
model, std, common, mask = fair.joint_setup(net, wf)
Batched = fair.batched_sampler_class()
NAMES = ["_compositional_score", "_get_score", "_row_score_jacobian", "_row_score_jacobian_fd",
         "_jacobian_joint_precision", "_effective_global_factors", "_project_information",
         "_apply_pd_strategy", "_solve_composed_global", "_clamp_local_scores", "_metric_local_clamp",
         "_final_denoise", "_training_prior_terms", "_schur_factors", "_global_block_adaptation"]
stats = collections.defaultdict(lambda: [0, 0.0, 0.0])   # calls, seconds, max peak GiB during
depth = [0]
def wrap(name, f):
    @functools.wraps(f)
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.time(); depth[0] += 1
        out = f(*a, **k)
        torch.cuda.synchronize(); depth[0] -= 1
        s = stats[name]; s[0] += 1; s[1] += time.time() - t
        s[2] = max(s[2], torch.cuda.max_memory_allocated() / 2**30)
        return out
    return g
for n in NAMES:
    if hasattr(M, n):
        attr = M.__dict__[n]
        if isinstance(attr, staticmethod):
            setattr(M, n, staticmethod(wrap(n, attr.__func__)))
        else:
            setattr(M, n, wrap(n, attr))
eigh_orig = M._batched_eigh
eig_stats = collections.Counter()
def eigh_wrap(matrices, max_batch=2048):
    torch.cuda.synchronize(); t = time.time(); before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = eigh_orig(matrices, max_batch)
    torch.cuda.synchronize()
    key = (tuple(matrices.shape), str(matrices.dtype))
    eig_stats[key] += 1
    stats["_batched_eigh"][0] += 1; stats["_batched_eigh"][1] += time.time() - t
    stats["_batched_eigh"][2] = max(stats["_batched_eigh"][2], (torch.cuda.max_memory_allocated() - before) / 2**30)
    return out
M._batched_eigh = staticmethod(eigh_wrap)
# model forward
fwd = model.model.forward
def fwd_wrap(*a, **k):
    torch.cuda.synchronize(); t = time.time()
    out = fwd(*a, **k); torch.cuda.synchronize()
    s = stats["network forward"]; s[0] += 1; s[1] += time.time() - t
    return out
model.model.forward = fwd_wrap

conds = fair.summaries_of(wf, test["sim_data"][:D, :100])
sampler = Batched(model); fair.configure(sampler, rule)
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    fair.sample_batched(sampler, conds, mask, 100, common | fair.SETTINGS[rule])
torch.cuda.synchronize(); total = time.time() - t0
print(f"D={D}: total {total:.2f}s ({total/D:.2f} s/dataset), overall peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
print(f"{'function (inclusive)':32s} {'calls':>6s} {'seconds':>8s} {'share':>6s} {'peak GiB':>8s}")
for n, (c, s, p) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
    print(f"{n:32s} {c:6d} {s:8.2f} {s/total:6.1%} {p:8.2f}")
print("eigh calls by (shape, dtype):")
for k, v in eig_stats.most_common(): print("  ", k, v)
