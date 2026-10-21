# E02 — Structured attention masks — **BLOCKED**

**Paper:** §3.2, §4.1, Fig. 4a ("undirected graph" / "directed graph" arms),
Fig. A4 (the base masks), Appendix A1.1 (what the masks do and do not enforce).

## What they did

Encode the simulator's known conditional-independence structure in the
transformer's attention mask `M_E`, as an undirected (symmetric) or directed
(causal) graph, and repair the directed mask when conditioning changes the
dependencies (Webb et al. 2018 algorithm). This is what carries their headline
claim: averaged across tasks and observations, ~10× fewer simulations than NPE,
with the gain concentrated in the sparse tasks (Gaussian Linear is fully
factorised; SLCP has four i.i.d. observations).

## Why this cannot run today

COMPASS has no edge/adjacency mask. Both trunks pass only
`key_padding_mask = 1 - c` to `nn.MultiheadAttention`
(`src/compass/ConditionTransformer.py:167`,
`src/compass/TokenizedTransformer.py:205`) — a *per-key* mask, not a
*query × key* matrix. Supporting `M_E` means threading an `attn_mask` through
`TransformerBlock.forward`, the trunk, `ScoreBasedInferenceModel`, and the
checkpoint's architecture payload.

Note also that the two masking notions interact: COMPASS's key-padding
convention already removes every latent token from the key set, so an `M_E`
restricted further would compose with a mask the paper does not have.

## What is prepared

`common/tasks.py` carries each task's directed generative structure as
`Task.edges` (`{child node: [parent nodes]}` over the flat vector), transcribed
from Fig. A4. That is the input `M_E` would need. Nothing consumes it.

## Recommendation

Leave out of scope for a first pass. It is their strongest efficiency number,
but the gain is concentrated in tasks with artificial sparsity, and the work is
a trunk redesign plus dynamic mask repair. Better to declare it out of scope
than to half-implement it — a wrong `M_E` silently degrades accuracy rather
than failing.
