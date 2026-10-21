# Results

Every number here is produced by `run_experiment.py` and rendered by
`make_tables.py` into `artifacts/tables.md`; the figures are `plot_model.py` and
`plot_methods.py`. Setup, parameters and caveats are in `README.md`.

One dataset per benchmark, `n = 8` observations, 3,000 posterior draws, 100
diffusion steps, one RTX 2080 Ti.

Coverage: all five arms, on both benchmarks, against all three score sources —
30 sampler runs in total.

---

## Headline

**COMPASS does construct the bimodality.** On the sign-ambiguous benchmark —
where every individual row is exactly symmetric about zero and says nothing at
all about the sign — all five arms recover *both* modes, at the right locations,
with the mass split within about two percentage points of the exact 0.500. None
of the composition rules collapses onto one branch, and none of them produces
the 2^n-moded product that treating the rows as independent would give. The
coordination is genuinely built by the composition.

**What separates the arms is not whether the modes survive but how much mass
leaks across the valley between them.** With the exact row score, the
ten-corrector DPM-2 arms put 2–4 draws of 3,000 into the middle fifth of the gap
against an exact expectation of 0.075; the corrector-free churn `eta = 8` arms
and Langevin + F-NPSE put 34–76 there, a factor of ten to a thousand too many,
and their modes come out 10–50% too wide to match. That ordering is the same one
`W1(g)/sigma` reports: 0.013–0.020 for the corrector arms against 0.043–0.075
for the churned and Langevin ones.

**The composed score and the draws disagree about F-NPSE, and the disagreement
is the interesting part.** F-NPSE's composed shared score is by far the worst of
the three rules at every noise level — 6.8% relative RMS at `lambda = 0.05`
rising to 690% at `lambda = 4`, against 0.009% for `gauss_jacobian` and 0.7% for
`gauss_hierarchical` at that same low end — and at `lambda = 0.2` it
over-amplifies the score by roughly 1.8x
(`05_sign_score_error.png`, right panel). Yet its *draws* are competitive with
the churned predictors. Annealed Langevin is corrector-only: it relaxes onto
whatever density its field defines at each level rather than transporting an
initial law through it, which forgives a badly scaled field in a way a
probability-flow predictor does not.

**On the control the story is different, and the difference is a prior
mis-specification, not a composition failure.** Every arm keeps both lobes and
none of them leaks a single draw into the gap, but every arm produces modes
2.5–4x too narrow (within-mode s.d. 0.09–0.15 against an exact 0.376) and pulls
the lobe split toward 50/50 (0.55–0.62 against an exact 0.643). That is what a too-flat prior in the `(1 - n)` prior-score term
does: `MultiObsSampler` takes a Gaussian prior, the control's is bimodal, and
the moment-matched stand-in `Normal(0, 2.06)` is five times wider than the true
lobes, so subtracting `(n - 1)` copies of its score over-sharpens the result.
The sign model, whose prior over `g` really is Gaussian, carries none of this —
which is why the two benchmarks' `W1` numbers should not be compared directly.

---

## Layer 1: the composed score, with the sampler removed

`05_<name>_score_error.png`. Relative RMS error of the composed shared score
against the closed-form diffused score, on states drawn from the exact diffused
posterior at each noise level.

Ordering, on the sign benchmark with the exact row score, at every `lambda`:

    gauss_jacobian  <  gauss_hierarchical  <<  F-NPSE

`gauss_jacobian` is essentially exact at small noise (9.4e-5 relative at
`lambda = 0.05`) and degrades to 0.12 by `lambda = 1` and 0.23 by `lambda = 2`;
`gauss_hierarchical` starts about seventy times worse (0.007) and converges to
within a factor of 1.5 of it by `lambda = 1`; F-NPSE is 0.068 at
`lambda = 0.05`, exceeds 1.0 — the error larger than the signal — by
`lambda = 0.5`, and reaches 6.9 at `lambda = 4`. Two arms that share a rule share a field exactly, so their
curves coincide: this layer sees the rule, never the sampler.

The vertical marker on those panels is the noise level at which a single row's
own two modes merge into one blob. Above it there are no two branches for a
composition rule to keep apart; below it, the rule has to reconstruct a
coordination that the individual rows do not contain. Both Gaussian rules do
their worst work in a band straddling that level.

---

## Layer 2: the draws

`03_<name>_marginals.png`, `04_<name>_joint.png`, `06_<name>_summary.png`, and
the tables in `artifacts/tables.md`.

### Learned versus exact row score

The three score sources answer a question a single learned run cannot, and the
answer does not split by composition rule — it splits by **whether the arm runs
Langevin correctors**.

*Corrector-free churned predictors track the network.* Going 552K -> 5.2M ->
exact row score, `W1/sigma` runs 0.112 -> 0.081 -> 0.075 for churned
`gauss_hierarchical` and 0.090 -> 0.056 -> 0.043 for churned `gauss_jacobian`.
Every other statistic converges too: at 5.2M parameters the gap-leakage counts
(74 and 40 draws) and the mode widths (0.297 and 0.256) are within Monte-Carlo
noise of the exact-score values (76/34, 0.300/0.260). These arms are limited by
the row score they are given, and improving it helps exactly as it should.

*Corrector-heavy arms are made worse by a learned score, and more capacity does
not fix it.* With the exact row score the ten-corrector arms are the two best
in the whole table (`W1/sigma` 0.020 for `gauss_hierarchical`, 0.013 for
`gauss_jacobian`). With the 552K network they are 0.185 and 0.093; with the 5.2M
network 0.190 and 0.127 — no better, and for `gauss_jacobian` worse. The failure
mode is legible in the other columns: their modes come out *too narrow*
(within-mode s.d. 0.163–0.177 against an exact 0.199 and 0.211–0.217 with the
exact row score) and the mass split swings 6–8 points off 0.500 (0.437 with the
small network, 0.582 with the large one).

That shape is what one would expect if a long Langevin chain is equilibrating to
the stationary distribution of a slightly over-confident field: a predictor
integrates a score error once, whereas ten corrector steps per level relax onto
whatever density that error implies. A degradation that survives a nine-fold
increase in network size, and disappears entirely when the row score is exact,
is a property of the arm rather than of the network.

This is the concrete payoff of running the closed-form row score. Without it,
"the corrector arms did badly" and "the network was too small" are the same
observation — and the naive fix, a bigger network, is the one that does not
work.

### Mode preservation

Mode counts are computed from a kernel density estimate at a bandwidth fixed to
the exact posterior's own component width, and compared against the exact
posterior seen through the same kernel — so the estimator's smoothing is not
charged to the sampler. All arms find both modes on both benchmarks with one
exception: two of the 5.2M-network sign arms report a third, spurious mode
(churned `gauss_hierarchical`, and Langevin + F-NPSE), a shoulder in the valley
that clears the 5%-of-peak prominence threshold. Both are corrector-free or
corrector-only arms with 65–74 draws in the gap, so that is leakage showing up
as structure, not a genuine extra branch.

### The joint

`04_sign_joint.png` is where the churned arms give themselves away. The exact
joint puts its mass in two blobs on the `(g, l_1)` diagonal; the churn `eta = 8`
arms and Langevin fill in a visible bridge along that diagonal between them,
while the corrector arms do not. The shared/local coupling summarises it:
mean `corr(g, l_j)` of 0.929–0.943 for the churned arms against 0.957–0.959 for
the corrector arms, and 0.962 exact.

### The control, briefly

Nothing separates the arms on the control the way it does on the sign model.
`W1(g)/sigma` sits between 0.19 and 0.31 for all fifteen runs; no arm leaks a
single draw into the gap; and the score source makes almost no difference —
churned `gauss_hierarchical` scores 0.228 / 0.217 / 0.266 with the 552K net, the
5.2M net and the exact row score, so the *exact* row score is nominally the
worst of the three. Improving the row score by a factor of ten (mean relative
error 0.230 -> 0.023) moves nothing.

Every arm instead gets the same two things wrong by about the same amount:
modes 2.5–4x too narrow (within-mode s.d. 0.09–0.15 against 0.376) and a lobe
split pulled toward 50/50 (0.55–0.62 against 0.643).

That is the signature of an error upstream of the composition rule. The
`(1 - n)` prior-score term is fed `Normal(0, 2.06)` where the truth is two lobes
of width 0.4, so seven copies of a much-too-flat prior score are being
subtracted; the result is over-sharpened by construction, whatever rule does the
composing and whatever score it composes. The control therefore grades the
sampler's prior handling, not its composition, and its numbers should not be
read as a rule comparison. Fixing it needs a non-Gaussian prior in
``MultiObsSampler``, not a different rule or a bigger network.

### Cost

The ranking is not free. With the exact row score, `gauss_jacobian` with ten
correctors is the most accurate arm and the most expensive: 3,804 score-network
evaluations and 26s, against 198 and 1.9s for churned `gauss_hierarchical`.
`gauss_hierarchical` with ten correctors takes a 55% larger `W1` (0.020 against
0.013) for a quarter of the cost. The churned corrector-free arms are the cheap
end (1.9–8s) and pay for it in valley leakage. With a *learned* score that trade reverses, because
the expensive arms are the ones the learned score hurts: churned
`gauss_jacobian` at 5.2M parameters is both the most accurate learned arm
(0.056) and cheaper than either corrector arm.

A practical note recorded in the code rather than discovered twice: with the
5.2M-parameter network, `gauss_jacobian` runs out of memory at 3,000 draws on an
11 GB card, because forward-mode duals through the transformer double the
activation footprint. `run_experiment.draw` takes the draws in chunks and halves
the chunk on an out-of-memory error; the draws are i.i.d., so this is
statistically identical to one call. It does inflate both cost columns for those
arms — each pass runs the full schedule over a quarter of the rows, so the call
count multiplies even though the total work does not, and `gauss_hierarchical`
really does re-estimate its pilot covariance per pass. The sampled statistics
are unaffected.

---

## Follow-up: where the corrector budget actually goes

`efficiency_sweep.py`, exact row score, same dataset and seed, re-measured on CPU
(so read it against its own baselines, not against the GPU table above — the
churn-8 baseline moves from 0.0435 to 0.0469 across devices, which is the scale
of the device/thread noise on a 3,000-draw statistic).

| arm | steps x snr | W1/sigma | gap | mode s.d. | calls | s |
|---|---|---|---|---|---|---|
| churn 8, 100 steps | — | 0.0469 | 33 | 0.256 | 594 | 18 |
| churn 8, 200 steps | — | 0.0328 | 37 | 0.255 | 1,194 | 31 |
| churn 8, 400 steps | — | 0.0314 | 36 | 0.257 | 2,394 | 73 |
| churn 2, 200 steps | — | 0.0917 | 102 | 0.317 | 1,194 | 29 |
| 2 correctors, snr 0.2 | 0.4 | 0.0339 | 33 | 0.254 | 1,236 | 30 |
| 10 correctors, snr 0.05 | 0.5 | 0.0305 | 16 | 0.233 | 3,804 | 94 |
| 3 correctors, snr 0.7 | 2.1 | 0.0248 | 26 | 0.248 | 1,557 | 37 |
| 4 correctors, snr 0.5 | 2.0 | 0.0274 | 9 | 0.218 | 1,878 | 46 |
| 5 correctors, snr 0.4 | 2.0 | 0.0116 | 16 | 0.224 | 2,199 | 55 |
| 10 correctors, snr 0.2 | 2.0 | 0.0114 | 2 | 0.214 | 3,804 | 93 |
| **10 correctors, snr 0.4** | 4.0 | **0.0094** | 5 | 0.215 | 3,804 | 93 |
| *exact* | | | 0.07 | 0.204 | | |

Three things follow, and two of them contradict the guess that prompted the
sweep.

**The predictor's residual error is not discretisation.** Going 100 -> 200 ->
400 steps moves `W1` from 0.047 to 0.033 to 0.031 and then stops, while the gap
leakage (33, 37, 36) and the mode width (0.256, 0.255, 0.257) do not move at
all. A corrector-free predictor converges to a law whose modes are 25% too wide,
and no amount of integrator refinement fixes that. Only relaxation does.

**Churn is helping, not leaking.** Weakening it from 8 to 2 makes everything
worse at identical cost (0.092, 102 draws in the gap, mode s.d. 0.317). The
churned predictor's over-dispersion is not caused by the churn.

**The corrector budget is a mixing-time budget, not an evaluation budget.**
Accuracy tracks `steps x snr` — the distance the Langevin chain travels — far
better than it tracks the number of network calls. Ten steps at a quarter of the
step size (`snr 0.05`, same 3,804 calls) scores 0.031, no better than a
predictor; five steps at twice the step size scores 0.0116, matching the
ten-step baseline at **58% of the cost**. And simply doubling `snr` from 0.2 to
0.4 at the stock ten steps buys an 18% accuracy improvement for *free*.

There is a ceiling: at `snr 0.7` with three steps the chain is taking steps too
large for an unadjusted Langevin discretisation and the gain reverses (0.025).
Somewhere around `snr 0.4-0.5` with at least five steps is the useful regime on
this problem. Caveat: this is one dataset and one seed at 3,000 draws, and the
inversion between `4 x 0.5` (0.027) and `5 x 0.4` (0.012) at the same product is
within the noise that implies — the robust readings are the two italicised
comparisons above, not a precise optimum.

### The same sweep with the 552K network -- the knob reverses sign

| arm | steps x snr | W1/sigma | gap | mode s.d. | calls | s |
|---|---|---|---|---|---|---|
| churn 8, 100 steps | — | 0.0901 | 6 | 0.226 | 594 | 24 |
| **churn 8, 200 steps** | — | **0.0664** | 9 | 0.223 | 1,194 | 40 |
| churn 8, 400 steps | — | 0.0775 | 3 | 0.212 | 2,394 | 81 |
| churn 2, 200 steps | — | 0.0499 | 53 | 0.290 | 1,194 | 40 |
| 2 correctors, snr 0.2 | 0.4 | 0.0754 | 9 | 0.216 | 1,236 | 41 |
| 10 correctors, snr 0.05 | 0.5 | 0.0863 | 3 | 0.195 | 3,804 | 128 |
| 3 correctors, snr 0.7 | 2.1 | 0.1225 | 0 | 0.162 | 1,557 | 52 |
| 4 correctors, snr 0.5 | 2.0 | 0.0887 | 0 | 0.165 | 1,878 | 63 |
| 5 correctors, snr 0.4 | 2.0 | 0.0888 | 1 | 0.167 | 2,199 | 74 |
| 10 correctors, snr 0.4 | 4.0 | 0.1199 | 0 | 0.160 | 3,804 | 128 |
| *exact* | | | 0.07 | 0.204 | | |

**Mixing time helps an exact field and hurts a learned one, monotonically.**
Raising `snr` from 0.2 to 0.4 at ten correctors takes the exact-score arm from
0.0114 to 0.0094 and the learned-score arm from 0.093 to 0.120. The mode width
column says why without ambiguity: *every* corrector arm driven by the learned
score converges to a width of 0.160-0.167 against an exact 0.204, and the only
one that escapes is the deliberately under-mixed `snr 0.05` (0.195) -- the arm
that relaxes least. The learned composed field is over-confident, and correctors
faithfully converge to its over-confidence; more mixing means more of it.

The predictor arms land at 0.212-0.226 instead, because a predictor never
equilibrates to anything.

**So there is no single best setting, and now there is a reason rather than an
observation.** The optimal corrector budget is a function of how biased the
field is, and it moves in opposite directions on either side of "exact". What
generalises is the *diagnostic*: run two corrector strengths and watch the mode
width. If it narrows as you add mixing, the field is over-confident and you
should be using a predictor; if it converges, you can spend on correctors.

**Practical recommendation.** With a near-exact score, ten correctors at
`snr 0.4` (free 18% over the stock `snr 0.2`) or five at `snr 0.4` (same
accuracy, 58% of the cost). With a learned score, the churned corrector-free
predictor at 200 steps: 0.066, best of its half, 1,194 calls, and it is also
respectable with the exact score (0.033 at a third of the corrector arm cost),
which makes it the right default when you do not know which regime you are in.

One warning about reading `W1` alone: churn 2 at 200 steps posts the lowest
learned-score `W1` in the table (0.050) while putting 53 draws in the gap and
over-widening the modes to 0.290. It is wrong in two directions that partly
cancel in a one-dimensional transport distance. The mode width and gap columns
are what separate it from the arm above it.

---

## Follow-up: two free recipe changes, and which one worked

`training_variants.py`, 552K network throughout, 2 x 2 over data normalization
and training-time sampling. Neither change adds a parameter or a single
evaluation at inference. Normalized runs are graded at `lambda / 1.5`, the
noise level corresponding to the original problem's, and reported in the
original columns; `W1 / sigma` and relative RMS are both scale free.

### Row-score fidelity (relative RMS against the closed form)

| variant | lam 0.05 | lam 0.1 | lam 0.2 | lam 0.5 | lam 1 | lam 2 | mean |
|---|---|---|---|---|---|---|---|
| baseline (mixture) | 0.440 | 0.207 | 0.152 | 0.082 | 0.069 | 0.048 | 0.166 |
| **log_sigma** | **0.088** | **0.068** | 0.057 | 0.043 | 0.038 | 0.029 | **0.054** |
| normalized | 0.434 | 0.106 | 0.064 | 0.041 | 0.030 | 0.023 | 0.116 |
| normalized + log_sigma | 0.259 | 0.091 | 0.058 | 0.042 | 0.039 | 0.022 | 0.085 |
| *h128d6, mixture (9x the parameters)* | *0.132* | *0.094* | *0.045* | *0.030* | *0.028* | *0.021* | *0.058* |

**Training-time sampling is the fix, and it is a large one.** Dropping the
uniform-in-`t` half of the mixture takes the mean error from 0.166 to 0.054 --
a factor of three -- and the smallest noise level, where the whole problem was,
from 0.440 to 0.088, a factor of five. The 552K network trained this way is
**better than the 5.2M network trained the old way** (0.054 against 0.058), at
a ninth of the parameters. The diagnosis was right: half the training budget was
being spent uniformly in `t`, which puts almost nothing near small `lambda`,
which is exactly where the sampler needs the score and where the error lived.

**Normalization helps, but not where it matters.** It improves the mean (0.166
-> 0.116) entirely through the middle of the range, and does essentially nothing
at the smallest noise level (0.434 against 0.440). Standardizing was never going
to fix a budget-allocation problem -- and this data was already O(1) (marginal
s.d. 1.50 / 1.58 / 1.60), so there was no scale pathology to cure.

**The two do not compose.** Together they give 0.085, materially worse than
`log_sigma` alone at 0.054, and worse at every noise level below 0.5. Whatever
normalization buys in the middle of the range, it gives back at the bottom once
the time sampling has already been concentrated there. On this problem the
recommendation is `log_sigma` *without* normalization.

### The prediction this was set up to test

The main run showed the corrector arm and the predictor arm reacting to
row-score error in opposite directions, which implies that *fixing the field
should help correctors disproportionately*. It does:

| variant | churn 8, 200 steps | 10 correctors | mode s.d. (correctors) |
|---|---|---|---|
| baseline | 0.0664 | 0.0932 | 0.164 |
| **log_sigma** | 0.0343 | **0.0204** | **0.191** |
| normalized | **0.0183** | 0.0779 | 0.172 |
| normalized + log_sigma | 0.0330 | 0.0191 | 0.191 |
| *exact row score* | *0.0328* | *0.0114* | *0.214* |
| *exact target* | | | *0.199* |

With the baseline recipe the corrector arm was the *worse* of the two (0.093
against 0.066) and its modes came out 18% too narrow. With `log_sigma` it
becomes the *better* of the two (0.020 against 0.034) and the mode width
recovers from 0.164 to 0.191 against an exact 0.199. The over-concentration was
never a property of the corrector chain; it was the chain faithfully
equilibrating to an over-confident field, and it disappears when the field is
fixed.

That is the practical headline of the whole follow-up: **`log_sigma` plus ten
correctors reaches `W1/sigma` = 0.0204**, against 0.056 for the best learned arm
in the main run (which needed the 5.2M network), and against an exact-row-score
ceiling of 0.0114 on the same arm. The gap to the analytic ceiling narrows from
about 5x to about 1.8x, using the small network, with no extra inference cost.

One caveat on the one number that does not fit the story: `normalized` with the
predictor posts 0.0183, nominally the best `W1` in the table, on a network whose
row score is three times worse than `log_sigma`'s. Its mode width (0.226) is
further from the truth than the corrector arms', so this looks like the usual
compensation of two errors in a one-dimensional transport distance rather than a
better posterior, and at 3,000 draws on one dataset it is not separable from
0.0191/0.0204 anyway. I would not build on it without repeat seeds.

---

## What this says about the design note's question

> Can COMPASS preserve bimodality? versus can COMPASS *construct* the correct
> bimodality from ambiguous rows?

Construct: yes, robustly, under every rule tested. The ferromagnetic
coordination that turns eight independently ambiguous rows into two coherent
branches is recovered by the composition, not lost in it, and the mass splits
between the branches correctly.

Preserve: also yes, but the control cannot answer it cleanly, for two reasons
that are worth keeping in mind before designing more control experiments. First,
with a well-separated prior and informative data the posterior is unimodal, so
the control's bimodality is a property of the *dataset* and the dataset has to
be chosen; the sign model's is a property of the likelihood and every dataset
has it. Second, the sampler's Gaussian prior term is mis-specified for any
genuinely bimodal prior, and that mis-specification dominates the control's
errors. The sign-ambiguous construction avoids both problems, which is exactly
the argument the design note made for it.

## Where a rule could still be caught out

Nothing here found a mode collapse, and the reason is visible in
`02_sign_emergence.png` panel D: for the largest `|y_j|` in this dataset the row
posterior's own two modes do not merge until `lambda = 1.77`, well inside the
schedule, so the sampler spends many steps in a regime where both branches are
explicitly represented in every row score.

The obvious way to push harder is more rows, and it does not work the way one
would guess. On this dataset the valley goes 0.1, 0.4, 5.8, 11.5 nats at
`n = 1, 2, 4, 8` — and then *back down* to 11.1 at `n = 16`
(`02_sign_ladder.csv`). The reason is worth recording: an observation with
`|y_j|` near zero is almost sign-indifferent, so it adds no coordination and
dilutes the rest. Extending this dataset to 16 rows brings in several such rows,
and the coherent-configuration mass falls from 0.96 to 0.27 even though the
marginal stays cleanly two-moded. Deepening the valley therefore means either
tighter pooling (`tau/sigma_y` below 1 puts it past 30 nats — panel C) or more
rows *conditioned on being informative*, not simply more rows.

Tighter pooling is the cleaner knob and is one command away
(`sign_model.TAU`); it is the natural next run.
