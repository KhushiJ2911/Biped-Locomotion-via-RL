# Auditing sim-to-sim evaluation for learned humanoid locomotion

Sim-to-sim transfer is the field's default hardware-free validation protocol:
train a policy in one simulator, check it still works in another, and treat that
as evidence it would survive contact with a real robot. It is used precisely
when authors lack hardware access.

The protocol has a free parameter that essentially nobody reports: **the physics
configuration**. If a policy's apparent sim-to-sim robustness depends on choices
like integration timestep or contact stiffness, then "validated in sim-to-sim"
is not a well-defined claim.

This repository measures that, in two layers.

## Experiment 1 — where does the engine gap come from?

`sim2sim/divergence.py`

Steps multiple physics backends from an identical initial state with an
identical open-loop control sequence, built from the *same* `mj_model`, and
measures how far their trajectories drift apart. Sweeping one physics option at
a time attributes the gap to individual factors.

Backends compared pairwise:

| pair | why it matters |
| --- | --- |
| `c \| jax` | the classic sim-to-sim comparison in the literature |
| `c \| warp` | what Playground users actually get — it now defaults to `warp` |
| `jax \| warp` | whether the two GPU backends agree with each other at all |

Two methodological points that materially change the numbers:

- **Physical duration is held constant, not step count.** At a fixed step count,
  `dt=0.004` integrates 2.0 s of physics while `dt=0.001` integrates 0.5 s.
  Since divergence grows with simulated time, fixing steps makes large
  timesteps look dominant for trivial reasons. `--duration` derives the step
  count per condition instead.
- **No-op perturbations are detected, not silently reported as null results.**
  Playground bipeds define foot/floor contacts as explicit `<pair>` elements
  that override per-geom parameters, so scaling `geom_friction` alone does
  nothing to the contacts carrying the robot. Each condition asserts it
  actually moved the reference trajectory; a condition that did not is marked
  `INVALID`, not "no effect".

Orientation error uses the `atan2` form of the quaternion geodesic rather than
`2*arccos(|dot|)`, which loses nearly all precision in exactly the
near-agreement regime being measured.

## Experiment 2 — does a *policy* survive the change?

`sim2sim/evaluate.py`

Experiment 1 is a property of the physics, not of a policy. This one takes a
single trained policy and runs it closed-loop under two different backends,
reporting return, episode length and termination rate.

The comparison is tightly controlled: both conditions load the same Playground
environment, so observations, rewards, terminations and command sampling are
identical. Only `config.impl` differs, so any performance gap is attributable to
the physics backend alone.

## Reproducing

```bash
# Experiment 1: three-way factor sweep
python -m sim2sim.divergence --env G1JoystickFlatTerrain --duration 1.0 --seeds 10
python3 tools/report_divergence.py results/divergence_G1JoystickFlatTerrain.json

# Train, then evaluate the same policy across backends
python -m sim2sim.train    --env G1JoystickFlatTerrain --impl jax --naconmax 8192
python -m sim2sim.evaluate --ckpt checkpoints/G1JoystickFlatTerrain_impl-jax_seed0
```

### Hardware notes

Developed on a 4 GB laptop GPU, which constrains two things:

- Playground's default `naconmax=65536` is sized for large GPUs and is the
  dominant VRAM cost. `--naconmax 8192` is what makes this fit.
- `num_evals` defaults to 20, and each eval forces a full epoch. A short run
  therefore costs `num_evals * epoch_size` steps regardless of `--timesteps` —
  asking for 60k with 20 evals actually runs 3.3M. Use `--num-evals` for
  benchmarks.

`sim2sim/_jax_compat.py` reinstates `jax.device_put_replicated`, which brax
0.14.2 still calls but JAX 0.10 removed.

## Status

Infrastructure is complete and verified. Results are not yet collected —
no numbers are reported here until they come from a duration-matched sweep
with at least 3 seeds.
