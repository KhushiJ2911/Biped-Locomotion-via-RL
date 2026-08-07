"""Experiment 3: how much simulator divergence can a policy actually tolerate?

Experiments 1 and 2 answer "do these engines differ" and "does it change policy
performance". Both are yes/no questions about one particular pair of engines.
This asks the quantitative question underneath them:

    A policy trained in simulator A and evaluated in simulator B loses
    performance as a function of how far apart A and B are. Where is the knee?

The design point is the x-axis. Plotting performance against a *parameter value*
(``solref_scale = 2.0``) says nothing transferable -- nobody else's simulator
pair is characterised by our solref setting. Plotting it against the *measured
state divergence* that the parameter induces gives a curve any two simulators
can be located on, including the MJX backend gaps from Experiment 1.

So each dose is characterised twice:

  dose (x)      open-loop state divergence between unperturbed and perturbed
                physics, same engine, same control sequence -- the same
                quantity Experiment 1 measures between engines, so the two are
                directly comparable and can share an axis.

  response (y)  closed-loop task performance of a trained policy evaluated
                under the perturbed physics.

Sweeping several *different* parameters is deliberate. If curves from solref,
friction and armature collapse onto a single line when plotted against measured
divergence, then tolerance is a property of *how far apart the simulators are*
rather than of which knob differs -- a far stronger and more useful claim than
any single-parameter result.

    python -m sim2sim.doseresponse --ckpt checkpoints/..._seed0
"""
from __future__ import annotations

import argparse
import json
import pathlib

import mujoco
import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Scale factors per parameter. Ranges deliberately extend well past Experiment
# 1's +/-2x: the point is to find where performance breaks, which requires
# perturbations large enough to break it.
DEFAULT_DOSES = {
    "solref_scale": [0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 8.0],
    "friction_scale": [0.2, 0.4, 0.6, 1.5, 2.5, 4.0],
    "armature_scale": [0.4, 0.6, 0.8, 1.5, 2.5, 4.0],
}


def perturb_mj_model(model: mujoco.MjModel, param: str, scale: float) -> mujoco.MjModel:
    """Apply one scaled perturbation to a copy of an MjModel.

    Explicit <pair> elements override per-geom contact parameters, so both
    arrays must be scaled or the perturbation silently misses the contacts that
    actually carry the robot.
    """
    import copy

    m = copy.deepcopy(model)
    if param == "solref_scale":
        m.geom_solref[:] = m.geom_solref * scale
        if m.npair:
            m.pair_solref[:] = m.pair_solref * scale
    elif param == "friction_scale":
        m.geom_friction[:] = m.geom_friction * scale
        if m.npair:
            m.pair_friction[:] = m.pair_friction * scale
    elif param == "armature_scale":
        m.dof_armature[:] = m.dof_armature * scale
    else:
        raise ValueError(f"unknown parameter {param!r}")
    return m


def perturb_mjx_model(mx, mj_model: mujoco.MjModel, param: str, scale: float):
    """Same perturbation applied to an mjx.Model (a frozen dataclass)."""
    if param == "solref_scale":
        upd = {"geom_solref": mx.geom_solref * scale}
        if mj_model.npair:
            upd["pair_solref"] = mx.pair_solref * scale
    elif param == "friction_scale":
        upd = {"geom_friction": mx.geom_friction * scale}
        if mj_model.npair:
            upd["pair_friction"] = mx.pair_friction * scale
    elif param == "armature_scale":
        upd = {"dof_armature": mx.dof_armature * scale}
    else:
        raise ValueError(f"unknown parameter {param!r}")
    return mx.replace(**upd)


def measure_dose(base_model: mujoco.MjModel, param: str, scale: float,
                 duration: float, seeds: int) -> dict:
    """Open-loop state divergence between unperturbed and perturbed physics.

    Uses MuJoCo-C for both sides so the engine is held fixed and only the
    physics configuration differs -- the dose is a property of the parameter
    change alone, not of any engine discrepancy.
    """
    from sim2sim.divergence import home_state, hold_pose_ctrl, perturb_state
    from sim2sim.engines import divergence_metrics, rollout_mjc

    pert = perturb_mj_model(base_model, param, scale)
    free_base = base_model.nq > base_model.nu + 6
    steps = int(round(duration / base_model.opt.timestep))

    pos, rot = [], []
    for s in range(seeds):
        q0, v0 = home_state(base_model)
        q0 = perturb_state(base_model, q0, s)
        ctrl = hold_pose_ctrl(base_model, home_state(base_model)[0], steps)
        qa, va = rollout_mjc(base_model, q0, v0, ctrl)
        qb, vb = rollout_mjc(pert, q0, v0, ctrl)
        m = divergence_metrics(qa, qb, va, vb, free_base=free_base)
        pos.append(float(m["base_pos_err_m"][-1]))
        rot.append(float(m["base_rot_err_rad"][-1]))

    return {
        "base_pos_err_m": float(np.mean(pos)),
        "base_pos_sem": float(np.std(pos, ddof=1) / np.sqrt(len(pos))) if len(pos) > 1 else 0.0,
        "base_rot_err_rad": float(np.mean(rot)),
        "base_rot_sem": float(np.std(rot, ddof=1) / np.sqrt(len(rot))) if len(rot) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dose-seeds", type=int, default=5,
                    help="initial conditions for the open-loop dose measurement")
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--naconmax", type=int, default=8192)
    ap.add_argument("--params", nargs="+", default=list(DEFAULT_DOSES))
    ap.add_argument("--impl", default="jax")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    from sim2sim import _jax_compat, _wandb

    _jax_compat.install()

    import jax
    from brax.io import model as brax_model
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    from sim2sim.evaluate import build_inference_fn, rollout_batch, summarise

    tag = pathlib.Path(args.ckpt).name
    wb = _wandb.init(
        enabled=not args.no_wandb,
        name=f"dose_{tag}",
        project=args.wandb_project,
        group=args.env,
        job_type="doseresponse",
        config={
            "env": args.env, "ckpt": args.ckpt, "episodes": args.episodes,
            "max_steps": args.max_steps, "impl": args.impl,
            "params": list(args.params), "duration_s": args.duration,
        },
    )

    cfg = registry.get_default_config(args.env)
    cfg.impl, cfg.naconmax = args.impl, args.naconmax
    env = registry.load(args.env, config=cfg)
    base_mj = env.mj_model
    base_mjx = env.mjx_model

    params_obj = brax_model.load_params(args.ckpt)
    ppo_params = locomotion_params.brax_ppo_config(args.env)
    policy_fn = build_inference_fn(env, ppo_params, params_obj)

    out = {"env": args.env, "ckpt": args.ckpt, "impl": args.impl,
           "episodes": args.episodes, "max_steps": args.max_steps, "points": []}

    def evaluate_current() -> dict:
        stats = rollout_batch(env, policy_fn, args.episodes, args.max_steps, args.seed)
        return summarise(stats, args.max_steps)

    # Reference: unperturbed physics, i.e. dose = 0 by construction.
    print("[ref] unperturbed ...", flush=True)
    ref = evaluate_current()
    print(f"    return={ref['return_mean']:.3f}+-{ref['return_sem']:.3f} "
          f"len={ref['length_mean']:.1f} term={ref['termination_rate']:.3f}", flush=True)
    out["reference"] = ref
    _wandb.summary(wb, {f"reference/{k}": v for k, v in ref.items()})

    for param in args.params:
        for scale in DEFAULT_DOSES[param]:
            dose = measure_dose(base_mj, param, scale, args.duration, args.dose_seeds)

            # Swap the perturbed physics into the live env. The env holds its
            # mjx model on a private attribute; everything else (observations,
            # rewards, terminations) is untouched, so only physics differs.
            env._mjx_model = perturb_mjx_model(base_mjx, base_mj, param, scale)

            # Guard: a perturbation that leaves the open-loop trajectory
            # untouched is invalid, not a null. Experiment 1 hit exactly this
            # with geom-vs-pair friction.
            if dose["base_pos_err_m"] == 0.0:
                print(f"    !! WARNING {param}={scale} is a NO-OP -- skipping", flush=True)
                env._mjx_model = base_mjx
                continue

            perf = evaluate_current()
            env._mjx_model = base_mjx  # restore before the next dose

            rec = {"param": param, "scale": scale, "dose": dose, "performance": perf,
                   "return_drop": ref["return_mean"] - perf["return_mean"],
                   "return_drop_frac": (ref["return_mean"] - perf["return_mean"])
                   / abs(ref["return_mean"]) if ref["return_mean"] else float("nan")}
            out["points"].append(rec)
            print(f"  {param}={scale:<5} dose={dose['base_pos_err_m']:.3e} m  "
                  f"return={perf['return_mean']:7.3f}  "
                  f"drop={rec['return_drop']:+7.3f} ({100*rec['return_drop_frac']:+.1f}%)  "
                  f"term={perf['termination_rate']:.3f}", flush=True)
            _wandb.log(wb, {
                "dose/base_pos_err_m": dose["base_pos_err_m"],
                "dose/base_rot_err_rad": dose["base_rot_err_rad"],
                "response/return_mean": perf["return_mean"],
                "response/return_drop": rec["return_drop"],
                "response/return_drop_frac": rec["return_drop_frac"],
                "response/termination_rate": perf["termination_rate"],
                "response/length_mean": perf["length_mean"],
                f"param/{param}": scale,
            })

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"doseresponse_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {path}")
    _wandb.finish(wb)


if __name__ == "__main__":
    main()
