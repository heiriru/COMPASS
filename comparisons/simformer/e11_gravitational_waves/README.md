# E11 — Gravitational waves with an embedding net — **DEFERRED**

**Paper:** A3.2. Referenced from §3.1 ("specialized embedding networks ... can be
efficiently integrated here by condensing complex data into a single token") and
from the Limitations section, as the answer to "what if the data are
high-dimensional and only the posterior is wanted".

## What they did

Condense high-dimensional data into a single token via an embedding network
trained end to end, target specific conditionals rather than the full joint, and
thereby scale to gravitational-wave data. They are explicit about the cost: it
"loses direct control over dependencies and condition states for individual data
elements".

## Why it is deferred, not blocked

Nothing in COMPASS forbids it — an embedding net in front of the tokenizer is a
straightforward addition, and `Partial_Pooling/pretrain_encoder.py` and
`SetEncoder.py` are evidence the repository already thinks in these terms. But
the experiment needs gravitational-wave data and a waveform simulator, and it
tests a claim (scaling) that is orthogonal to everything else in this directory.

## Recommendation

Skip unless an actual high-dimensional application appears. If the *capability*
needs demonstrating, do it on a task already in the repository.
