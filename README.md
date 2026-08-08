# When Simulators Disagree

An audit of **sim-to-sim validation** — the practice of training a locomotion
policy in one physics backend and checking it in another, which stands in for
hardware testing when no robot is available.

We measure how far the three MuJoCo backends in common use actually drift
apart (the C reference, MJX-JAX, and MuJoCo Warp), why they drift, and whether
a trained humanoid policy notices.

## Findings

**The backends are not interchangeable under contact.** Without contact all
three agree to float32 rounding and the two GPU backends are identical. With
contact, MJX-JAX is the outlier in **34 of 56** robot × condition tests across
four bipeds (Benjamini-Hochberg, 5%), by factors of 2 to 82.

**The cause is algorithmic, not numerical.** Two independent controls each
rule out one explanation while the gap survives:

| Control | Effect on contact-free gap | Effect on contact gap |
| --- | --- | --- |
| float64 instead of float32 | removed, 9 orders of magnitude | unchanged (ratio 1.05) |
| refining `dt` 8× | — | Warp → 0 at O(dt); JAX stops at a floor |

Warp and the C reference are the same algorithm discretised differently.
MJX-JAX is not.

**A trained policy absorbs it anyway.** Measured against Warp and against the
C reference itself, performance drops 2.7% (pooled *t*(191) = 2.88) — smaller
than the spread from retraining with a different seed (0.53×).

**Why:** contact mismatch is ~40× more damaging than inertial mismatch *at
equal measured divergence*. Where two simulators differ matters more than how
much. The engine gap lands in the range a policy tolerates, which makes
sim-to-sim agreement a **weak test**.

**A protocol warning.** Op3 appeared immune to the whole effect (1/14
conditions). It was not the robot — it was the shipped solver settings.
Matched to the other bipeds, it goes to **13/14**. A sim-to-sim check can pass
simply because the defaults are too coarse to resolve the disagreement.

## The paper

`paper/ieee/` — IEEE conference format, ready for Overleaf.
`paper/draft.md` — the long-form working draft, with reasoning the paper omits.

```bash
cd paper/ieee
python3 check_tex.py           # structure: refs, cites, figures, environments
python3 verify_tex_numbers.py  # every table re-read from results/*.json
```

Both must pass before submitting. The second one exists because transcribing
numbers into tables by hand is where errors enter a paper, and nothing else in
the pipeline would catch one.

## Layout

```
sim2sim/     experiment code (one module per experiment)
tools/       analysis, statistics, plotting, verification
results/     raw JSON — every number in the paper traces to a file here
figures/     the six paper figures; each regenerates from results/ alone
checkpoints/ four trained PPO policies, for reproducing the evaluations
paper/       draft.md, and ieee/ for submission
```

## Reproducing

Environment: MuJoCo 3.11.0, mujoco-mjx 3.11.0, JAX 0.10.2 (CUDA 12),
brax 0.14.2, mujoco_playground 0.2.0, warp-lang 1.15.0, Python 3.11.

`sim2sim/_jax_compat.py` reinstates `jax.device_put_replicated`, which brax
0.14.2 still calls but JAX 0.10 removed.

Full command list is in `paper/draft.md` §12. The short version:

```bash
python -m sim2sim.groundtruth --engines c jax warp    # closed-form baseline
python -m sim2sim.divergence --duration 1.0 --seeds 10 --engines c jax warp
python3 tools/analyze_conditions.py --csv figures/condition_level.csv
```

## Things worth knowing before you extend this

Four protocol errors produced confident wrong answers here. Each ran without
error and returned a plausible number. They are documented in the paper, and
each now has a guard:

- **Fixed step count instead of fixed duration** gave a 109× timestep effect
  that is truly null (1.50 ± 0.49). Conditions are specified by physical
  duration now.
- **Means over heavy-tailed divergence** — mean/median reaches 10.4 on some
  robots. Ratios are medians with bootstrap intervals.
- **A null model with mismatched units** reported a 3.17× effect on data
  containing none. Systematic and per-episode tests need separate nulls.
- **Silent constraint overflow** — `njmax` too small for Op3's box feet meant
  the solver dropped constraints and returned wrong physics while printing a
  warning nothing was reading. The rollout wrapper raises now.

Evaluations on GPU are **not** bit-reproducible: repeating an identical run
gives a s.d. of 1.86% of the mean at 128 episodes. Any improvement smaller
than that is indistinguishable from a rerun.

## Status

Experiments complete. The paper is drafted and its numbers verified. Open
items: first Overleaf build (layout not yet checked), citation venue
confirmation in `paper/ieee/refs.bib`, and trimming to the target page limit —
`main.tex` opens with an ordered trimming guide.
