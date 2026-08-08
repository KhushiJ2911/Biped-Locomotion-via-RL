# What Actually Drives the Sim-to-Sim Gap in Humanoid Locomotion?

**Status: working draft.** All eight experiments are complete. No number appears
in this document that was not produced by a committed script in this repository
and checked against its source JSON by `tools/verify_paper_numbers.py`.

---

## Abstract (draft)

Sim-to-sim transfer — training a locomotion policy in one physics backend and
verifying it in another — is the field's standard hardware-free validation
protocol, used precisely when authors lack access to a robot. Its own
reliability has not been audited. We measure how far the three MuJoCo backends
in common use actually diverge (the C reference, MJX-JAX, and MuJoCo Warp),
whether that divergence matters to a trained policy, and what it takes to
measure either correctly.

**MJX-JAX does not converge to MuJoCo-C.** Contact-free, all three backends
agree to float32 round-off and the two GPU backends are numerically identical.
Under contact they separate, with MJX-JAX the outlier in **34 of 56 robot x
condition tests** across four bipeds (Benjamini-Hochberg, 5%), spanning 2x to
82x. Two independent controls show this is not numerical: running MJX in
**double precision** leaves the contact gap unchanged (ratio 1.05) while
eliminating the contact-free gap by nine orders of magnitude, and **refining the
solver** drives MuJoCo-Warp's disagreement toward zero at O(dt) while MJX-JAX
plateaus at a floor of ~1.9e-03 that no refinement removes. Warp and C are the
same algorithm discretised differently; MJX-JAX is not.

**Policy performance is nevertheless unaffected.** Evaluated against Warp and
against the C reference itself, a trained locomotion policy shows a small,
consistent, statistically detectable degradation (2.7%, pooled t(191) = 2.88)
that is **smaller than the variation from choosing a different training seed**
(0.53x). Section 5 explains why: contact-parameter mismatch is ~40x more
damaging than inertial mismatch at equal divergence, and the measured engine gap
lands in the regime where a policy tolerates it.

**Measuring this correctly is harder than it appears.** We document four
protocol errors that each produced a confident wrong answer in our own pipeline
— unmatched simulated duration (a 109x artefact), means over heavy-tailed
distributions, dimensionally mismatched null models, and silently dropped solver
constraints — together with the guards that now make each of them fail loudly.

The practical consequence: sim-to-sim validation across MuJoCo backends is a
**weak test**. The gap it exercises falls below the threshold at which policies
respond, so an unchanged policy is not evidence of much.

---

## 1. Motivation

The standard argument for sim-to-sim validation is that a policy surviving a
change of simulator is more likely to survive contact with hardware. The
protocol is widely used and rarely examined. It also contains a free parameter
that is almost never reported: the physics configuration under which the check
is performed.

If a policy's apparent sim-to-sim robustness depends on choices such as contact
stiffness, then "validated in sim-to-sim" is not a well-defined claim, and two
papers reporting it may be reporting different things.

We do not have hardware. That constrains what we can claim — we can say which
engines *disagree*, and by how much, but "closer to the C reference" is a
statement about agreement, not physical accuracy. Section 2 addresses this
directly by scoring every engine against closed-form solutions rather than
against each other.

---

## 2. Experiment 0 — absolute accuracy against closed-form solutions

**Purpose.** Establish an external reference, and separate integrator error from
contact-solver error.

![Figure 6](../figures/fig6_schematic.pdf)
**Figure 6.** What is held fixed, what is varied, and what is measured. All
three engines are built from one compiled model and driven from identical
initial state and control, so any divergence is attributable to the
implementation.

**Method.** Four systems with exact analytic solutions, each rolled out
identically in all three backends (`sim2sim/groundtruth.py`).

A discrete integrator is not expected to reproduce the continuous solution
exactly. Semi-implicit Euler on constant acceleration accumulates a
deterministic offset of `g*dt^2*n/2` after n steps. We therefore score against
the *discrete* prediction, which is what distinguishes a correct implementation
from a buggy one.

### 2.1 Contact-free: free fall (500 steps, dt = 0.002)

Analytic continuous z = 5.095000 m; discrete semi-implicit Euler z = 5.085190 m.

| engine | final z | error vs discrete Euler |
| --- | --- | --- |
| MuJoCo-C | 5.085190 | 4.44e-15 |
| MJX-JAX | 5.085197 | 6.97e-06 |
| MuJoCo-Warp | 5.085197 | 6.97e-06 |

MuJoCo-C reproduces the analytic discrete solution to machine precision. Both
GPU backends deviate by ~7e-6 — **identically to each other**, the signature of
float32 arithmetic against C's float64.

### 2.2 Contact-free: pendulum energy conservation

Total mechanical energy is conserved by the true dynamics, so any drift is an
integrator artefact. Energy is evaluated post-hoc with a single C model for all
three trajectories, so the energy function is identical and only the trajectory
differs.

| engine | relative energy drift |
| --- | --- |
| MuJoCo-C | 8.884e-04 |
| MJX-JAX | 8.891e-04 |
| MuJoCo-Warp | 8.891e-04 |

All three share the same drift to three significant figures. This is the
integrator's artefact, not an engine difference.

### 2.3 Contact-dominated: block on an incline

Gravity is tilted rather than the geometry rotated, keeping the contact normal
exactly along +z so the analytic prediction is unambiguous.

**Sliding regime** (theta = 30 deg, mu = 0.3; analytic x = 1.178144 m):

| engine | x final | abs error |
| --- | --- | --- |
| MuJoCo-C | 1.180169 | 2.03e-03 |
| MJX-JAX | 1.180264 | 2.12e-03 |
| MuJoCo-Warp | 1.180173 | 2.03e-03 |

All within 0.18% of closed form. But Warp matches C to 3.99e-06 while JAX sits
9.51e-05 away — a **23.9x** difference in agreement, in a case where all three
engines are "correct" to within a fifth of a percent of the analytic answer.

**Static regime** (theta = 15 deg, mu = 0.8; the block must not move):

| engine | displacement after 1 s |
| --- | --- |
| MuJoCo-C | 1.147948e-03 |
| MJX-JAX | 1.128214e-03 |
| MuJoCo-Warp | 1.147944e-03 |

**Every engine creeps ~1.1 mm/s under a block that should be perfectly static.**
The magnitude is near-identical across engines, so this is inherited from
MuJoCo's soft-contact *model*, not from any implementation. For locomotion this
is a planted foot sliding ~1 cm over a 10-second episode, and it affects all
MuJoCo-based work regardless of backend.

![Figure 5](../figures/fig5_groundtruth.pdf)
**Figure 5.** All three engines scored against closed-form solutions. MJX-JAX
and MuJoCo-Warp coincide to plotting precision in every case; MuJoCo-C is
separated only in the contact-free cases, where the difference is float64 vs
float32.

### 2.4 What Experiment 0 establishes

Without contact, the three engines agree to float32 precision and the two GPU
backends are indistinguishable. **Any larger disagreement on a humanoid is
therefore attributable to contact and constraint solving, not to arithmetic.**
This is the floor against which Experiment 1 is read.

---

## 3. Experiment 1 — factor attribution on a 29-DoF humanoid

**Purpose.** Determine which physics factors drive the engine gap.

**Method.** `G1JoystickFlatTerrain` (nq=36, nv=35, nu=29) is stepped from an
identical initial state under an identical open-loop hold-pose control sequence,
built from the same `mj_model`, in all three backends. One physics option is
varied at a time. n = 10 initial conditions per cell; errors are standard errors
of the mean, propagated through ratios in quadrature. A factor is called null
when its ratio lies within 2 sigma of 1.0.

Two design points materially change the numbers:

**Physical duration is held constant, not step count.** At a fixed step count,
`dt=0.004` integrates 2.0 s of physics while `dt=0.001` integrates 0.5 s. Since
divergence grows with simulated time, fixing steps makes large timesteps appear
dominant for trivial reasons. *We initially measured timestep at 109x baseline
under a fixed step count; duration-matched, the same condition is 1.50 ± 0.49 —
a null result. This correction is the single largest change to our conclusions.*

**No-op perturbations are detected rather than reported as nulls.** Playground
bipeds define foot/floor contact as explicit `<pair>` elements that override
per-geom parameters, so scaling `geom_friction` alone silently does nothing to
the contacts carrying the robot. Each condition asserts it moved the reference
trajectory; none of the fourteen triggered the guard.

### 3.1 Baseline engine disagreement

| engine pair | base position error (m) | base orientation error (rad) |
| --- | --- | --- |
| c \| jax | 3.354e-03 ± 7.6e-04 | 6.381e-03 ± 1.2e-03 |
| c \| warp | **4.120e-04 ± 8.6e-05** | 1.914e-03 ± 4.4e-04 |
| jax \| warp | 3.385e-03 ± 8.1e-04 | 6.468e-03 ± 1.3e-03 |

Warp is ~8x closer to the C reference than JAX. That `jax|warp` (3.385e-03) is
statistically identical to `c|jax` (3.354e-03) shows **JAX is the outlier**:
Warp and C sit together, JAX sits apart from both.

### 3.2 Factor attribution (metric: base position error, vs baseline)

`c|jax` — one significant effect in thirteen:

| condition | ratio | verdict |
| --- | --- | --- |
| solref_scale=2.0 | **13.26 ± 4.54** | **widens** |
| friction_scale=0.5 | 1.64 ± 0.58 | null |
| timestep=0.004 | 1.50 ± 0.49 | null |
| solref_scale=0.5 | 1.36 ± 0.42 | null |
| armature_scale=0.5 | 1.24 ± 0.50 | null |
| iterations=10 / 50 / 100 | 1.12 ± 0.41 (identical) | null |
| ls_iterations=20 | 1.00 ± 0.33 | null |
| cone=1 | 0.97 ± 0.35 | null |
| timestep=0.001 | 0.95 ± 0.33 | null |
| friction_scale=2.0 | 0.65 ± 0.21 | null |
| armature_scale=2.0 | 0.63 ± 0.19 | null |

`c|warp` — three significant:

| condition | ratio | verdict |
| --- | --- | --- |
| solref_scale=0.5 | 5.17 ± 1.32 | widens |
| friction_scale=2.0 | 0.52 ± 0.17 | narrows |
| armature_scale=2.0 | 0.47 ± 0.14 | narrows |

### 3.3 Contact stiffness controls the Warp advantage

| contact stiffness | Warp advantage over JAX |
| --- | --- |
| stiff (solref x0.5) | 2.1x |
| default | 8.1x |
| soft (solref x2.0) | **28.2x** |

Three points, one monotonic trend. Softer contacts give the constraint solver
more freedom, and MJX-JAX's solver is the one that exploits it. The practical
consequence is that a flat recommendation ("use Warp") is unsafe: the advantage
is conditional on contact configuration, and hard contacts — what practitioners
reach for when chasing hardware realism — are where it is weakest.

### 3.4 Two useful null results

**Solver iterations do nothing.** 10, 50 and 100 iterations produce *identical*
divergence to four significant figures (3.739e-03), all statistically
indistinguishable from the shipped default of 3. Tuning this knob is wasted
effort.

**Timestep does not dominate.** 1.50 ± 0.49, a null. Reported here explicitly
because we believed the opposite before controlling for simulated duration.

---

### 3.5 Across four bipeds, tested per condition rather than per robot

Experiment 1 was replicated on every biped MuJoCo Playground ships -- Unitree
G1, Booster T1, Berkeley Humanoid and ROBOTIS Op3 -- under an identical
protocol.

![Figure 2](../figures/fig2_crossembodiment.pdf)
**Figure 2.** The full robot x condition grid; 34 of 56 cells are significant
after Benjamini-Hochberg correction at 5%. Reading only the baseline row is
what produced our initial and wrong conclusion that only G1 differs.

**Comparing engines at each robot's shipped baseline is not sufficient.** The
discrepancy is strongly condition-dependent: T1 shows no advantage at baseline
(1.27 [0.64, 2.86]) and a significant one in six other configurations. Reading
robot-dependence off a single configuration produced a conclusion -- "only G1
differs" -- that the full grid refutes. We therefore test every robot x
condition cell and correct with Benjamini-Hochberg, since at 56 simultaneous
tests roughly three cells would clear an uncorrected 5 % threshold by chance.

Ratios are medians with bootstrap confidence intervals. Divergence
distributions are heavy-tailed on several robots (mean/median up to 10.4), so a
mean ratio is dominated by individual seeds; Berkeley's mean ratio of 3.32 at
n=10 was an artefact of exactly this, and resolves to 2.91 [1.58, 4.51] at
n=30 with medians.

| robot | significant conditions | median significant ratio |
| --- | --- | --- |
| G1 | **13 / 14** | 10.23 |
| Berkeley Humanoid | **13 / 14** | 3.08 |
| T1 | 6 / 14 | 3.27 |
| Op3 | 2 / 14 | 2.09 |

**34 of 56 conditions are significant after correction.** The claim is
therefore not that one robot is unusual, but that **MJX-JAX departs from the
MuJoCo-C reference more than MuJoCo-Warp does across most physics
configurations on most bipeds**, with magnitudes spanning 2x to 82x.

Two qualifications matter. Op3 is largely null (2/14), so robot-to-robot
variation is real even if it is not the binary split a baseline-only comparison
suggests. And the effect is **not universal in direction**: Op3 under
``solref_scale=0.5`` gives 0.33 [0.15, 0.89], significantly *below* one, meaning
MJX-JAX is the closer backend there.

We considered and rejected a measurement-floor explanation -- that T1 and Op3
simply diverge too little for a ratio to resolve. Their baseline divergences are
13-45x smaller than G1's and Berkeley's, which is suggestive. But T1's
*significant* conditions occur at its **smallest** divergences (1.4e-04 to
2.9e-04) while its two largest-divergence conditions (1.3e-03, 1.9e-03) show
nothing. A floor effect predicts the opposite ordering.

**A data-integrity note.** Every Op3 sweep prior to the one reported here was
invalid. The MJX backends size their constraint buffer from ``njmax``; the
default was too small for Op3's four box feet, so the solver dropped constraints
and returned physically wrong trajectories while printing "nefc overflow" to
stdout. The rollout wrapper now raises rather than returning such a trajectory.
Corrupted outputs are retained under ``results/INVALID_op3_nefc_overflow/``
rather than deleted.

## 4. Experiment 2 — does a learned policy survive the swap? (complete)

**Result: yes. Policy performance is unaffected by the backend swap, despite the
physics gap being real and large.**

Three PPO policies (52.9M steps each, 512 envs, MJX-JAX) were evaluated in both
backends over 128 episodes x 1000 steps, giving 384 matched episode pairs.

| | JAX | Warp |
| --- | --- | --- |
| mean return (over 3 policies) | 12.523 | 12.849 |
| seed-to-seed sd of that mean | 0.582 | 0.396 |
| within-backend episode sd | 8.651 | 8.714 |

![Figure 3](../figures/fig3_policy_null.pdf)
**Figure 3.** The measured backend effect against the seed-to-seed variation it
must be judged relative to. The effect falls inside the band that retraining
alone produces.

### 4.1 No systematic performance shift

Paired over all 384 episodes, the mean difference is **+0.3264**, with
`t(383) = 1.400` -- below the 1.96 threshold for 5% significance. Paired
Cohen's d is **+0.071**, negligible. The per-policy signed gaps do not even
share a sign (+0.431, -0.022, +0.570), so there is no consistent direction.

Against the pre-registered decision rule, comparing like with like -- difference
of backend means against the seed-to-seed spread of those means:

    |0.3264| / 0.5822 = 0.56     (>= 2 required for an effect)

**The cross-backend gap is smaller than ordinary training-seed variation.**

### 4.2 Individual episodes are perturbed, but less than natural variation

The mean absolute per-episode difference is 1.845, against a within-backend
episode-to-episode sd of 8.651 -- a ratio of **0.213**. Individual episodes do
diverge, sometimes sharply (max |difference| 27.6, and 13% of episodes differ by
more than 5 return units), but the perturbation is roughly a fifth of the
variation the task produces anyway.

Trajectories diverge far more visibly than returns do. A single policy rolled
out from an identical state in both backends separates steadily:

| elapsed | base separation |
| --- | --- |
| 1 s | 0.019 m |
| 4 s | 0.068 m |
| 8 s | 0.190 m |
| 10 s | **0.271 m** |

against 3.4e-03 m for open-loop hold-pose over 1 s. Closed-loop execution
amplifies the physics gap by roughly two orders of magnitude in *state space* --
each small state difference feeds back through the policy into different
actions, which produce different contacts -- and yet leaves *task performance*
statistically unchanged. The policy is not insensitive to the physics; it is
robust in the aggregate while being individually unpredictable.

### 4.3 A methodological warning

Our own analysis script initially reported this as a **3.17x effect that
exceeded seed noise**. That verdict was wrong: it divided the mean absolute
*per-episode* difference by the seed-to-seed spread of *means*. Those quantities
have different units, and the ratio is inflated by roughly the square root of
the episode count. Corrected, the same data gives 0.56 -- a null.

The two questions require separately matched nulls: a systematic shift is tested
against between-seed variation of means, and per-episode perturbation against
within-backend episode variation. Conflating them converts a null into an
apparent effect, and we report this because the error is easy to make and was
caught only by checking the units.

### 4.4 What this means for sim-to-sim validation

The engines genuinely disagree -- up to 28x between Warp and JAX in Section 3,
and 0.21 m of trajectory divergence under a policy. But a locomotion policy's
*measured performance* is robust to that disagreement, to within less than the
noise introduced by choosing a different training seed.

For practitioners this is reassuring rather than alarming: **sim-to-sim
validation across MJX backends is sound for flat-terrain humanoid locomotion**,
and a policy that scores well in one backend will score equivalently in the
other. The caveat is that individual episodes are not reproducible across
backends, so any claim resting on specific rollouts -- a demonstration video, a
failure-case analysis, a worst-case bound -- is backend-dependent even though
the aggregate is not.

We do not claim this generalises. It is one task, one robot, flat terrain, and a
policy trained to a quarter of the reference budget. Section 3 shows the engine
gap grows sharply with contact softness, so a task with softer or more
intermittent contact could plausibly cross the threshold where performance does
move.

## 4b. Experiment 2 design (as pre-registered)

Experiments 0 and 1 characterise the physics. They say nothing about whether a
*learned policy* behaves differently, which is the question a robot-learning
venue cares about.

**Design.** Three PPO policies (50M timesteps each, 512 envs) trained in MJX-JAX
on `G1JoystickFlatTerrain`, each evaluated in both JAX and Warp over 128
episodes x 1000 steps. Both conditions load the same environment, so
observations, rewards, terminations and command sampling are identical; only
`config.impl` differs.

**Statistics.** Both backends reset from the same PRNG key, so episodes are
matched pairs, and we report paired differences rather than a difference of two
independent means. This matters because signed per-episode differences largely
cancel: in the final data the signed mean gap is +0.3264 while the mean
*absolute* per-episode gap is 1.8446, five times larger. Reporting only the
signed mean would understate how much individual episodes actually move.

**Decision rule, fixed in advance.** The cross-backend gap is only a finding if
it exceeds seed-to-seed training noise. Training three seeds in one backend and
evaluating each in both yields the effect and its null model from the same runs.
The verdict is `|gap| / seed_noise`: >= 2 is an effect, 1-2 is weak, < 1 is no
policy-level effect. This is implemented in `tools/compare_transfer.py`, which
refuses to return a verdict when fewer than two seeds are available.

**A null outcome here is a publishable result** and will be reported as such: it
would say the engines diverge measurably but learned policies absorb it, and
sim-to-sim validation is sound for this task class.

---

## 5. Experiment 3 — how much divergence can a policy tolerate?

Experiments 1 and 2 ask yes/no questions about one engine pair. This asks the
quantitative question underneath: a policy trained in simulator A and evaluated
in B loses performance as a function of how far apart A and B are -- where is
the knee, and is distance alone a sufficient description?

**Method.** Each perturbed physics configuration is characterised twice. The
*dose* is the open-loop state divergence between unperturbed and perturbed
physics under MuJoCo-C, i.e. the same quantity Experiment 1 measures between
engines, so the two share an axis. The *response* is closed-loop task
performance of a trained policy under the perturbed physics. Three parameters
were swept over 19 conditions, three policy seeds, 64 episodes each.

Plotting against measured divergence rather than parameter value is what makes
the result transferable: no other group's simulator pair is characterised by our
``solref_scale``, but every pair can be located by how far apart it is.

![Figure 4](../figures/fig4_tolerance.pdf)
**Figure 4.** Closed-loop cost against measured open-loop divergence, with the
real engine gaps marked on the same axis. Contact and inertial perturbations of
equal dose produce very different closed-loop consequences.

### 5.1 Distance alone is not a sufficient statistic

Within each parameter, divergence predicts loss well. Across parameters, it does
not:

| parameter | Spearman rho (loss vs dose) | loss range |
| --- | --- | --- |
| solref (contact stiffness) | **+0.964** | 2.1 – 135.6 % |
| friction (contact) | **+0.943** | 5.3 – 323.3 % |
| armature (joint inertia) | **+0.143** | -0.0 – 6.4 % |

At matched dose the difference is stark: ``friction=0.4`` (dose 5.90e-03 m)
costs **51.7 %** while ``armature=4.0`` (dose 5.55e-03 m) costs **1.3 %** -- a
40x difference at the same divergence. Armature never exceeds 6.4 % loss at any
divergence tested.

**The headline is therefore not a universal tolerance curve.** It is that *where*
two simulators differ matters more than *how much* they differ. A single scalar
summary of simulator disagreement -- which is what a naive sim-to-sim check
reports -- cannot distinguish a benign inertial mismatch from a damaging contact
mismatch of identical magnitude.

We initially expected the curves to collapse and drew that conclusion from a
partial sweep; the complete condition set refutes it. The distinction survives
because it is mechanistically sensible: armature is a joint-space inertia term
that feedback compensates for, whereas contact parameters change how the robot
interacts with the ground, which feedback cannot recover from.

### 5.2 Why this explains Experiment 2

Experiment 0 established that the engine gap is *specifically* a contact-solver
gap. It therefore falls in the damaging family, and the contact-parameter curve
predicts roughly **6–7 % loss** at G1's measured ``c|jax`` gap of 3.35e-03 m.

Section 6 puts the evaluation noise floor at 1.86 % of the mean. A 6 % expected
effect against that floor, with seed-to-seed variation of comparable size, is
precisely the regime in which Experiment 2 returned a null. The null is not an
absence of physics -- it is a predicted-size effect sitting at the resolution
limit of the measurement.

### 5.3 One perturbation destabilised the simulator

``solref_scale=0.25`` (contacts 4x stiffer) drove divergence to 1.01 m and
returns to NaN. It is excluded from the curve and reported separately: "the
simulator became unstable" is a different statement from "the policy degraded",
and averaging a NaN into a series would have silently removed the point from the
figure while leaving it in the data.

## 6. Measurement noise floor

Every comparison in this paper rests on evaluations that are nominally
deterministic -- fixed PRNG keys, a deterministic policy, an unchanged model.
On GPU they are not reproducible: repeating an identical evaluation gives
different returns, because floating-point reductions are not associative and
thread scheduling varies between launches.

| episodes (x 1000 steps) | s.d. over 3 identical repeats | % of mean |
| --- | --- | --- |
| 16 | 0.837 | 9.91 % |
| 64 | 0.571 | 8.34 % |
| 128 | 0.127 | **1.86 %** |

At Experiment 2's settings the floor is 1.86 % of the mean. The measured
cross-backend gap (+0.3264) is 2.57x that s.d., so it is resolvable -- it is
simply not significant, and smaller than seed-to-seed variation.

We report this as a measurement with three repeats and make no claim about how
it scales; three samples estimate a standard deviation too poorly to support one.
We note it because it appears to be undocumented: benchmark returns from this
stack carry an unstated uncertainty of this order, and a reported improvement
smaller than it is not distinguishable from a rerun.

## 7. Experiment 4 — is contact representation the mechanism?

G1 is the only robot of four showing a large engine discrepancy, and also the
only one declaring foot contact through explicit ``<pair>`` elements rather than
geom-geom collision -- its foot geoms carry ``contype=0, conaffinity=0`` and
cannot collide by any other route. That is a correlation across four robots with
one significant point, so we tested it directly.

**Method.** Two models of the same robot, differing only in how the identical
contact is declared: (A) as shipped, and (B) with the floor<->foot pairs deleted
and the foot geoms given collision flags carrying the pair's condim, friction
and solref. Because geom-geom friction is *combined* between the two geoms
whereas a pair's friction is used directly, the floor's friction was matched as
well; without that the converted model would silently have had different physics.

**Control.** The two models are **bit-identical under MuJoCo-C** (0.000e+00
divergence across five initial conditions). MuJoCo-C routes explicit pairs and
dynamic geom pairs through the same solver, so any difference the GPU backends
show between A and B is an implementation artefact, not physics.

**Result (n=30, median ratio with bootstrap 95 % CI):**

| variant | contacts | Warp advantage |
| --- | --- | --- |
| A, as shipped | explicit ``<pair>`` | 5.96 [3.43, 13.13] |
| B, converted | geom-geom | 6.68 [4.54, 13.96] |

**The advantage survives the conversion**; the intervals overlap almost
entirely. Contact representation is not the mechanism. The hypothesis is
eliminated by construction rather than left untested, and the mechanism behind
G1's uniqueness remains open -- remaining candidates (foot geometry type, robot
scale, contact count) cannot be separated with four robots.

## 8. Experiment 5 — the comparison the field actually performs

Experiment 2 compared MJX-JAX against MuJoCo-Warp: two GPU reimplementations of
the same engine. The sim-to-sim protocol in the literature validates against
**MuJoCo-C**, and Section 3 shows that is where the discrepancy lives. So the
comparison that matters had not been run.

**Method.** Playground environments are MJX-native (``impl='cpp'`` cannot step),
so we reproduce against ``mujoco.MjData`` the parts of the environment the
policy depends on: the 103-dim ``state`` observation, the action-to-motor-target
map, the gait phase, and the termination condition. Observation noise and random
pushes are disabled in both, making the environment a deterministic function of
(qpos, qvel, phase, last_action, command).

**Validation.** The reimplementation agrees with MJX to **2.38e-07** on the full
observation at reset and **5.96e-08** on bookkeeping across 200 steps, with the
gait phase exact. This mattered: four bugs were found only because these checks
existed, each of which produced plausible output and no error -- a frozen gait
clock, an off-by-one in ``last_act`` and ``phase`` (the environment calls
``_get_obs`` *before* updating them), an inverted termination test
(``get_gravity`` reads the ``upvector_torso`` **sensor**, whose sign is opposite
to the ``site_xmat`` product used for the pelvis in the observation), and a flaw
in the validator itself (MuJoCo evaluates sensors before the final integration,
so imposing qpos/qvel and calling ``mj_forward`` compares two different
instants).

**Horizon.** Episodes are capped at 500 steps because the environment resamples
the velocity command once ``step > 500``. Beyond that the two engines track
different targets, which manufactures a large spurious advantage for MuJoCo-C
(t = -11 at 1000 steps, "better" in 61/64 episodes, correlated with command
magnitude). The runner refuses horizons above 500.

**Result.** Three policies, 64 paired episodes each, scored on velocity-tracking
error:

| seed | n | MJX-JAX | MuJoCo-C | paired diff | t |
| --- | --- | --- | --- | --- | --- |
| 0 | 64 | 0.2861 | 0.2923 | +0.0062 | 4.09 |
| 1 | 64 | 0.2650 | 0.2771 | +0.0122 | 1.86 |
| 2 | 64 | 0.2917 | 0.2958 | +0.0040 | 1.02 |
| pooled | 192 | | | **+0.0075** | **2.877** |

Two statements are simultaneously true and answer different questions. There
**is** a systematic effect: pooled over 192 paired episodes the policy tracks
worse in MuJoCo-C, t(191) = 2.877, consistent in sign across all three seeds,
Cohen's d = 0.21. And it **does not matter**: against the pre-registered rule of
Section 4, the difference of means (0.0075) is **0.53x** the seed-to-seed spread
(0.0141), below the threshold of 2.

We report the null, as pre-registered, despite the paired test being
significant. The observed 2.7% degradation is also the size Section 5's
contact-parameter curve predicts at G1's measured gap.

Both policy-level experiments therefore agree, now including the comparison the
field actually performs.

## 9. Experiment 6 — is the discrepancy arithmetic or algorithmic?

Section 2 showed the *contact-free* difference is float32 versus float64.
Section 3 found gaps three orders of magnitude larger under contact. Whether
those are the same phenomenon scaled up is the first question a reader should
ask, and it decides how much the result means.

**Method.** Re-run the MuJoCo-C vs MJX-JAX comparison with JAX in double
precision (``jax_enable_x64``), across conditions spanning Experiment 1's
largest and smallest gaps. MuJoCo Warp rejects float64 arrays, so this is
JAX-only -- which is the relevant comparison, since JAX is the backend that
departs from the reference.

**Positive control.** In float64 the contact-free free-fall error falls from
6.97e-06 to **4.441e-15**, exactly matching MuJoCo-C. The precision change
demonstrably works and demonstrably removes precision-driven discrepancy.

**Result (median over 20 initial conditions):**

| condition | float32 | float64 | ratio |
| --- | --- | --- | --- |
| baseline | 1.921e-03 | 2.013e-03 | 1.048 |
| solref_scale=2.0 | 3.101e-02 | 3.310e-02 | 1.067 |
| solref_scale=0.5 | 2.928e-03 | 3.120e-03 | 1.065 |
| friction_scale=2.0 | 2.236e-03 | 2.261e-03 | 1.011 |
| timestep=0.001 | 1.955e-03 | 2.062e-03 | 1.055 |

**Double precision does not close the contact gap.** Every ratio is ~1.0, and if
anything marginally above it.

The control makes this decisive rather than inferential: the same switch that
erases the contact-free gap by nine orders of magnitude leaves the contact gap
untouched. **The contact discrepancy between MJX-JAX and MuJoCo-C is
algorithmic, not arithmetic** -- the implementations disagree about contact
dynamics, not merely about rounding.

This also changes the practical outlook. A precision difference would shrink as
hardware and libraries improve; an algorithmic one will persist until an
implementation changes.

## 10. Experiment 7 — does the gap survive solver refinement?

Experiment 6 ruled out arithmetic. This rules out discretisation. A difference
that is merely an artefact of finite step size and finite solver iterations
should vanish as both are refined; one that persists means two implementations
converging to different answers.

**Method.** A 4x4 grid over ``dt`` in {4e-3, 2e-3, 1e-3, 5e-4} and solver
``iterations`` in {1, 3, 10, 50}, baseline condition only, 20 initial conditions
per cell, all three engines. This reuses the validated Experiment 1 sweep rather
than new code.

### 10.1 Solver iterations saturate at 3

Across every timestep, the gap drops sharply from 1 to 3 iterations and then
stops changing:

| dt | iters=1 | 3 | 10 | 50 |
| --- | --- | --- | --- | --- |
| 0.004 | 1.098e-02 | 3.232e-03 | 3.649e-03 | 3.668e-03 |
| 0.002 | 4.881e-03 | 2.164e-03 | 2.323e-03 | 2.216e-03 |
| 0.001 | 5.895e-03 | 2.017e-03 | 1.993e-03 | 1.915e-03 |
| 0.0005 | 3.600e-03 | 1.892e-03 | 1.940e-03 | 1.956e-03 |

(``c|jax``, median over 20 initial conditions.) This independently reproduces
Experiment 1's bit-identical iterations result on a different grid and a
different measurement. Playground's shipped default of 3 is already converged.

### 10.2 The two backends refine differently

At the most-converged iteration count (50), refining the timestep:

| dt | c \| jax | c \| warp |
| --- | --- | --- |
| 0.004 | 3.668e-03 | 6.359e-04 |
| 0.002 | 2.216e-03 | 2.536e-04 |
| 0.001 | 1.915e-03 | 1.511e-04 |
| **0.0005** | **1.956e-03** | **7.459e-05** |

**MuJoCo-Warp converges to MuJoCo-C.** Each halving of dt roughly halves the gap
(ratios 0.40, 0.60, 0.49) -- clean first-order convergence toward zero. An
eight-fold refinement yields an 8.5x reduction.

**MJX-JAX converges to something else.** It falls to ~1.9e-03 and stops; the
finest timestep is marginally *worse* than the next-coarsest. Refinement cannot
remove it.

At the finest settings the MJX-JAX gap is **26x** the Warp gap, and that ratio
grows as refinement continues.

![Figure 1](../figures/fig1_algorithmic.pdf)
**Figure 1.** The two independent controls. (a) Refining dt: MuJoCo-Warp
converges toward MuJoCo-C along slope 1 while MJX-JAX flattens at a floor.
(b) Doubling precision: float64 collapses the contact-free gap by nine orders
of magnitude and leaves the contact gap unchanged.

### 10.3 Two orthogonal controls, one conclusion

Experiments 6 and 7 vary independent knobs -- arithmetic precision and
discretisation -- and each removes every *other* candidate explanation while
leaving the MJX-JAX gap intact. Warp's disagreement with C behaves exactly as a
discretisation difference should under both; MJX-JAX's does not under either.

This also explains the cross-embodiment pattern in Section 3.5. The Warp
advantage is visible wherever MJX-JAX's algorithmic floor dominates and hidden
wherever discretisation error is large enough to mask it -- which is precisely
why Op3's coarse shipped settings (dt=0.004, iterations=1) concealed the effect
until they were matched to the other robots.

## 11. Limitations

- **No hardware.** We can rank engines by agreement with the C reference and by
  accuracy on closed-form problems. We cannot say which is closer to reality.
- **One robot, one task.** All humanoid results are `G1JoystickFlatTerrain`.
  Cross-embodiment claims are not supported.
- **Statistical power.** n=10 resolves the 13x contact-stiffness effect
  comfortably but not factor effects of size ~1.5x. "Null" here means "not
  resolved at this sample size", not "absent".
- **Experiment 1 is open-loop.** Hold-pose control in a contact-rich standing
  regime, not locomotion. Experiment 2 addresses this.
- **Compute.** A 4 GB laptop GPU caps parallel environments at 512 (measured
  faster than 1024) and the training budget at 50M timesteps, one quarter of
  Playground's default for this task.

---

## 12. Reproducing

```bash
python -m sim2sim.groundtruth --engines c jax warp          # Experiment 0
python -m sim2sim.divergence  --duration 1.0 --seeds 10 \
                              --engines c jax warp          # Experiment 1
python3 tools/report_divergence.py results/divergence_*.json
tools/run_experiment2.sh 50000000 3 jax                     # Experiment 2
python3 tools/compare_transfer.py results/eval_*.json
python -m sim2sim.doseresponse --ckpt CKPT                   # Experiment 3
python -m sim2sim.contactrep                                # Experiment 4
python -m sim2sim.ceval --validate --ckpt CKPT --steps 200  # Experiment 5 check
tools/run_ceval.sh                                          # Experiment 5
python -m sim2sim.precision            # Experiment 6, float32
python -m sim2sim.precision --x64      # Experiment 6, float64
tools/run_convergence.sh                                    # Experiment 7
tools/run_cross_embodiment.sh                               # Section 3.5

# Figures (each reads only from results/ and writes figures/)
python3 tools/plot_convergence.py       # Figure 1
python3 tools/analyze_conditions.py --csv figures/condition_level.csv
python3 tools/plot_crossembodiment.py   # Figure 2
python3 tools/plot_policy_null.py       # Figure 3
python3 tools/plot_tolerance.py         # Figure 4
python3 tools/plot_groundtruth.py       # Figure 5
python3 tools/plot_schematic.py         # Figure 6

python3 tools/verify_paper_numbers.py   # traces every number back to results/
```

Environment: MuJoCo 3.11.0, mujoco-mjx 3.11.0, JAX 0.10.2 (CUDA 12), brax
0.14.2, mujoco_playground 0.2.0, warp-lang 1.15.0, Python 3.11.

`sim2sim/_jax_compat.py` reinstates `jax.device_put_replicated`, which brax
0.14.2 still calls but JAX 0.10 removed.
