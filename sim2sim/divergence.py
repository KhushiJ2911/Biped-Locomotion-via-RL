"""Experiment 1: cross-engine (MuJoCo-C vs MJX) state divergence for bipeds.

Both engines are stepped from an identical initial state with an identical
control sequence, on a model built from the same ``mj_model``. Any divergence
is therefore attributable to the engine implementation. Sweeping ``PhysicsOpts``
tells us which physics option dominates that gap.

Usage:
    python -m sim2sim.divergence --env G1JoystickFlatTerrain --steps 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

from sim2sim.engines import (
    PhysicsOpts,
    divergence_metrics,
    rollout_mjc,
    rollout_mjx,
)

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def home_state(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Initial state: the model's 'home' keyframe if it has one, else qpos0."""
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        if name in ("home", "stand", "standing", "init"):
            return model.key_qpos[i].copy(), np.zeros(model.nv)
    if model.nkey > 0:
        return model.key_qpos[0].copy(), np.zeros(model.nv)
    data = mujoco.MjData(model)
    return data.qpos.copy(), np.zeros(model.nv)


def hold_pose_ctrl(model: mujoco.MjModel, qpos0: np.ndarray, steps: int) -> np.ndarray:
    """Constant control that commands the initial pose.

    Playground bipeds use position actuators, so holding the home pose keeps the
    robot in a contact-rich standing regime -- the regime where engines disagree
    most and where locomotion policies actually live.
    """
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        trn = model.actuator_trnid[a, 0]
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            adr = model.jnt_qposadr[trn]
            ctrl[a] = qpos0[adr]
    return np.tile(ctrl, (steps, 1))


def perturb_state(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    seed: int,
    joint_sd: float = 0.02,
    height_sd: float = 0.005,
) -> np.ndarray:
    """Small random perturbation of the initial pose, for variance estimates.

    seed=0 returns the unperturbed home pose so the n=1 result stays comparable.
    """
    if seed == 0:
        return qpos0.copy()
    rng = np.random.default_rng(seed)
    q = qpos0.copy()
    free_base = model.nq > model.nu + 6
    start = 7 if free_base else 0
    q[start:] += rng.normal(0.0, joint_sd, size=model.nq - start)
    if free_base:
        q[2] += abs(rng.normal(0.0, height_sd))
    return q


def run_condition(
    base_model: mujoco.MjModel,
    opts: PhysicsOpts,
    duration: float,
    ref_qpos: np.ndarray | None = None,
    seed: int = 0,
    engines: tuple[str, ...] = ("c", "jax", "warp"),
    njmax: int | None = None,
    naconmax: int | None = None,
) -> dict:
    """Roll out both engines under one physics configuration.

    ``duration`` is in *physical seconds*, and the step count is derived from
    each condition's own timestep. Fixing the step count instead would confound
    the timestep sweep with simulated time: at a fixed 500 steps, dt=0.004
    integrates 2.0 s of physics while dt=0.001 integrates only 0.5 s. Since
    engine divergence grows with simulated time, that alone would make large
    timesteps look dominant regardless of their true per-second effect.

    If ``ref_qpos`` (the baseline MuJoCo-C trajectory) is given, we assert the
    perturbation actually moved the trajectory. A perturbation that silently
    does nothing -- e.g. scaling geom params when explicit <pair> elements
    override them -- would otherwise look like "this factor doesn't matter".
    """
    model = opts.apply(base_model)
    steps = int(round(duration / model.opt.timestep))
    qpos0, qvel0 = home_state(model)
    qpos0 = perturb_state(model, qpos0, seed)
    # Control still targets the *unperturbed* home pose, so each seed is a
    # different recovery transient from the same commanded posture.
    ctrl = hold_pose_ctrl(model, home_state(model)[0], steps)

    free_base = model.nq > model.nu + 6

    traj, wall = {}, {}
    for eng in engines:
        t0 = time.time()
        if eng == "c":
            traj[eng] = rollout_mjc(model, qpos0, qvel0, ctrl)
        else:
            traj[eng] = rollout_mjx(model, qpos0, qvel0, ctrl, impl=eng,
                                    njmax=njmax, naconmax=naconmax)
        wall[eng] = round(time.time() - t0, 2)

    # Every unordered engine pair. With engines = (c, jax, warp) this gives
    # c|jax (the classic sim-to-sim comparison), c|warp (what Playground users
    # are actually exposed to, since it now defaults to warp), and jax|warp
    # (whether the two GPU backends agree with each other at all).
    pairs = {}
    for i, a in enumerate(engines):
        for b in engines[i + 1 :]:
            (qa, va), (qb, vb) = traj[a], traj[b]
            pairs[f"{a}|{b}"] = divergence_metrics(qa, qb, va, vb, free_base=free_base)

    # Guard: did this perturbation actually do anything? Only meaningful at
    # seed 0 -- for seed>0 the initial state itself differs from ref_qpos, so
    # a nonzero diff would prove nothing and the check would never fire.
    effect = None
    if ref_qpos is not None and opts.label() != "baseline" and seed == 0 and "c" in traj:
        effect = float(np.abs(traj["c"][0][-1] - ref_qpos[-1]).max())
        if effect == 0.0:
            print(
                f"    !! WARNING: '{opts.label()}' left the MuJoCo-C trajectory "
                f"bit-identical to baseline -- the perturbation is a NO-OP. "
                f"Treat this condition as invalid, not as a null result.",
                flush=True,
            )

    return {
        "label": opts.label(),
        "opts": {k: v for k, v in vars(opts).items() if v is not None},
        "steps": steps,
        "dt": float(model.opt.timestep),
        "sim_time_s": steps * model.opt.timestep,
        "perturbation_effect_on_mjc": effect,
        "wall_s": wall,
        "pairs": {
            p: {
                "final": {k: float(v[-1]) for k, v in m.items()},
                "max": {k: float(np.max(v)) for k, v in m.items()},
                "curves": {k: v.tolist() for k, v in m.items()},
            }
            for p, m in pairs.items()
        },
        "_traj": {
            f"qpos_{e}": q for e, (q, _) in traj.items()
        } | {f"qvel_{e}": v for e, (_, v) in traj.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="physical seconds to simulate; step count is derived per condition "
        "so the timestep sweep is not confounded with simulated time",
    )
    ap.add_argument("--seeds", type=int, default=1, help="initial conditions per cell")
    ap.add_argument("--quick", action="store_true", help="baseline condition only")
    ap.add_argument(
        "--engines",
        nargs="+",
        default=["c", "jax", "warp"],
        help="engines to compare pairwise: 'c' (MuJoCo-C reference), 'jax' "
        "(MJX-JAX), 'warp' (MuJoCo Warp, Playground's current default)",
    )
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    # Robots ship with different solver defaults -- Op3 uses dt=0.004 and
    # iterations=1 where every other biped uses 0.002 and 3. Comparing its
    # engine gap to theirs without matching those first confounds the robot
    # with its integration settings.
    ap.add_argument("--base-timestep", type=float, default=None,
                    help="override the baseline model's timestep before sweeping")
    ap.add_argument("--base-iterations", type=int, default=None,
                    help="override the baseline model's solver iterations")
    ap.add_argument("--njmax", type=int, default=None,
                    help="constraint-buffer size for the MJX backends. Too small "
                         "and the solver silently DROPS constraints (it prints "
                         "'nefc overflow' to stdout but still returns a "
                         "trajectory), which corrupts the physics. Op3 needs "
                         ">=76; other bipeds fit the default.")
    ap.add_argument("--naconmax", type=int, default=None)
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filename, to keep variants apart")
    args = ap.parse_args()
    engines = tuple(args.engines)

    from sim2sim import _wandb

    wb = _wandb.init(
        enabled=not args.no_wandb,
        name=f"divergence_{args.env}",
        project=args.wandb_project,
        group=args.env,
        job_type="divergence",
        config={
            "env": args.env,
            "duration_s": args.duration,
            "seeds": args.seeds,
            "engines": list(engines),
        },
    )

    from mujoco_playground import registry

    print(f"[load] {args.env}")
    env = registry.load(args.env)
    base = env.mj_model
    if args.base_timestep is not None:
        base.opt.timestep = args.base_timestep
    if args.base_iterations is not None:
        base.opt.iterations = args.base_iterations
    print(
        f"[model] nq={base.nq} nv={base.nv} nu={base.nu} "
        f"dt={base.opt.timestep} solver={base.opt.solver} "
        f"iters={base.opt.iterations} cone={base.opt.cone}"
    )

    # Factors to attribute. Each isolates one physics option against baseline.
    conditions = [PhysicsOpts()]  # baseline = Playground's shipped settings
    if not args.quick:
        conditions += [
            PhysicsOpts(iterations=10),
            PhysicsOpts(iterations=50),
            PhysicsOpts(iterations=100),          # MuJoCo's own default
            PhysicsOpts(ls_iterations=20),
            PhysicsOpts(cone=1),                   # elliptic friction cone
            PhysicsOpts(timestep=0.001),
            PhysicsOpts(timestep=0.004),
            PhysicsOpts(friction_scale=0.5),
            PhysicsOpts(friction_scale=2.0),
            PhysicsOpts(solref_scale=0.5),         # stiffer contacts
            PhysicsOpts(solref_scale=2.0),         # softer contacts
            PhysicsOpts(armature_scale=0.5),
            PhysicsOpts(armature_scale=2.0),
        ]

    # Baseline MuJoCo-C trajectory, used to verify each perturbation bites.
    q0, v0 = home_state(base)
    base_steps = int(round(args.duration / base.opt.timestep))
    print(f"[plan] duration={args.duration}s -> {base_steps} baseline steps")
    ref_qpos, _ = rollout_mjc(base, q0, v0, hold_pose_ctrl(base, q0, base_steps))

    out = {
        "env": args.env,
        "duration_s": args.duration,
        "seeds": args.seeds,
        "conditions": [],
    }
    for i, opts in enumerate(conditions, 1):
        print(f"[{i}/{len(conditions)}] {opts.label()} ...", flush=True)
        per_seed = [
            run_condition(
                base, opts, args.duration, ref_qpos=ref_qpos, seed=s,
                engines=engines, njmax=args.njmax, naconmax=args.naconmax
            )
            for s in range(args.seeds)
        ]
        rec = per_seed[0]
        for pair in rec["pairs"]:
            finals = {
                k: [p["pairs"][pair]["final"][k] for p in per_seed]
                for k in rec["pairs"][pair]["final"]
            }
            rec["pairs"][pair]["seed_finals"] = finals
            rec["pairs"][pair]["mean"] = {
                k: float(np.mean(v)) for k, v in finals.items()
            }
            # Standard error, not standard deviation: we are reporting how well
            # the condition mean is pinned down, which is what the comparison
            # between conditions actually rests on.
            rec["pairs"][pair]["sem"] = {
                k: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                for k, v in finals.items()
            }
        out["conditions"].append(rec)
        for pair in rec["pairs"]:
            # One flat summary key per (condition, pair, metric) so runs across
            # robots can be compared directly in the W&B UI.
            _wandb.summary(
                wb,
                {
                    f"{opts.label()}/{pair}/{k}": v
                    for k, v in rec["pairs"][pair]["mean"].items()
                }
                | {
                    f"{opts.label()}/{pair}/{k}_sem": v
                    for k, v in rec["pairs"][pair]["sem"].items()
                },
            )
        for pair in rec["pairs"]:
            m, se = rec["pairs"][pair]["mean"], rec["pairs"][pair]["sem"]
            print(
                f"    {pair:10s} "
                f"base_pos={m['base_pos_err_m']:.3e}+-{se['base_pos_err_m']:.1e}  "
                f"base_rot={m['base_rot_err_rad']:.3e}+-{se['base_rot_err_rad']:.1e}  "
                f"(n={args.seeds})",
                flush=True,
            )

    RESULTS.mkdir(exist_ok=True)

    # Raw trajectories, so metrics can be revised without re-simulating.
    traj = {}
    for rec in out["conditions"]:
        for k, v in rec.pop("_traj").items():
            traj[f"{rec['label']}|{k}"] = v
    npz = RESULTS / f"traj_{args.env}{'_' + args.tag if args.tag else ''}.npz"
    np.savez_compressed(npz, **traj)

    suffix = f"_{args.tag}" if args.tag else ""
    path = RESULTS / f"divergence_{args.env}{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {path}\n[saved] {npz}")
    _wandb.finish(wb)


if __name__ == "__main__":
    main()
