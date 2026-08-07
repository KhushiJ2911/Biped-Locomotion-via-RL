"""Train a locomotion policy in MJX with brax PPO, for the sim2sim study.

The trained policy is what turns the divergence study from "a collapsing squat"
into "walking, with contacts making and breaking" -- the regime the sim-to-real
gap actually lives in.

Requires a working CUDA JAX. On CPU this is ~200 env-steps/s vs ~1e5 on GPU,
i.e. days instead of minutes; the script refuses to start a full run on CPU
unless --allow-cpu is passed.

    python -m sim2sim.train --env G1JoystickFlatTerrain
"""
from __future__ import annotations

import argparse
import functools
import json
import pathlib
import time

import jax

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
CKPT = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--timesteps", type=int, default=None, help="override num_timesteps")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument(
        "--impl",
        default=None,
        choices=["jax", "warp"],
        help="MJX backend to train in. Playground defaults to 'warp'; 'jax' has a "
        "smaller memory footprint. Which backend a policy is trained in is itself "
        "a factor in the transfer study, so it is recorded in the checkpoint name.",
    )
    ap.add_argument(
        "--naconmax",
        type=int,
        default=None,
        help="contact buffer size. Playground's default (65536) is sized for large "
        "GPUs and is the dominant VRAM cost on a 4GB card.",
    )
    ap.add_argument("--njmax", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument(
        "--num-evals",
        type=int,
        default=None,
        help="Playground defaults to 20. Each eval forces a full epoch, so a short "
        "run costs num_evals * epoch_size steps regardless of --timesteps: asking "
        "for 60k with 20 evals actually runs 3.3M. Lower this for benchmarks.",
    )
    args = ap.parse_args()

    backend = jax.default_backend()
    print(f"[jax] backend={backend} devices={jax.devices()}")
    if backend == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "Refusing to train on CPU: MJX is ~500x slower without CUDA and a full\n"
            "run would take days. Fix the GPU first:\n"
            "    sudo mokutil --import /var/lib/shim-signed/mok/MOK.der\n"
            "    reboot -> blue MOK Manager screen -> Enroll MOK -> Continue -> Yes\n"
            "then reinstall JAX with CUDA:\n"
            "    pip install -U 'jax[cuda12]'\n"
            "Pass --allow-cpu to override (smoke tests only)."
        )

    from sim2sim import _jax_compat

    if _jax_compat.install():
        print("[compat] reinstated jax.device_put_replicated for brax")

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    env_cfg = registry.get_default_config(args.env)
    if args.impl is not None:
        env_cfg.impl = args.impl
    if args.naconmax is not None:
        env_cfg.naconmax = args.naconmax
    if args.njmax is not None:
        env_cfg.njmax = args.njmax
    print(
        f"[env] impl={env_cfg.impl} naconmax={env_cfg.naconmax} njmax={env_cfg.njmax}"
    )

    env = registry.load(args.env, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(args.env)

    if args.timesteps is not None:
        ppo_params.num_timesteps = args.timesteps
    if args.num_envs is not None:
        ppo_params.num_envs = args.num_envs
    if args.num_evals is not None:
        ppo_params.num_evals = args.num_evals
    print(
        f"[cfg] timesteps={ppo_params.num_timesteps:,} "
        f"num_envs={ppo_params.num_envs} batch={ppo_params.batch_size}"
    )

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

    from sim2sim import _wandb

    run_name = f"{args.env}_impl-{env_cfg.impl}_seed{args.seed}"
    wb = _wandb.init(
        enabled=not args.no_wandb,
        name=run_name,
        project=args.wandb_project,
        group=args.env,
        job_type="train",
        config={
            "env": args.env,
            "impl": str(env_cfg.impl),
            "seed": args.seed,
            "num_timesteps": ppo_params.num_timesteps,
            "num_envs": ppo_params.num_envs,
            "batch_size": ppo_params.batch_size,
            "episode_length": ppo_params.episode_length,
            "naconmax": env_cfg.naconmax,
            "njmax": env_cfg.njmax,
        },
    )

    progress_log = []
    t_start = time.time()

    def progress(step: int, metrics: dict) -> None:
        rec = {
            "step": int(step),
            "wall_s": round(time.time() - t_start, 1),
            "reward": float(metrics.get("eval/episode_reward", float("nan"))),
            "reward_std": float(metrics.get("eval/episode_reward_std", float("nan"))),
        }
        progress_log.append(rec)
        # Forward every brax metric, not just reward: the per-term reward
        # breakdown is what tells us *why* a policy plateaued.
        _wandb.log(
            wb,
            {"train/wall_s": rec["wall_s"], **{k: float(v) for k, v in metrics.items()}},
            step=int(step),
        )
        print(
            f"[{rec['wall_s']:8.1f}s] step={rec['step']:>10,}  "
            f"reward={rec['reward']:8.2f} +- {rec['reward_std']:.2f}",
            flush=True,
        )

    train_fn = functools.partial(
        ppo.train,
        **{k: v for k, v in dict(ppo_params).items() if k != "network_factory"},
        network_factory=network_factory,
        progress_fn=progress,
        seed=args.seed,
    )

    print(f"[train] {args.env} ...")
    _, params, _ = train_fn(
        environment=env,
        eval_env=registry.load(args.env, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    CKPT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    from brax.io import model as brax_model

    tag = f"{args.env}_impl-{env_cfg.impl}_seed{args.seed}"
    out = CKPT / tag
    brax_model.save_params(str(out), params)
    (RESULTS / f"train_{tag}.json").write_text(json.dumps(progress_log, indent=2))

    if progress_log:
        final = progress_log[-1]
        _wandb.summary(
            wb,
            {
                "final_reward": final["reward"],
                "final_reward_std": final["reward_std"],
                "total_wall_s": final["wall_s"],
                "checkpoint": str(out),
            },
        )
    _wandb.finish(wb)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
