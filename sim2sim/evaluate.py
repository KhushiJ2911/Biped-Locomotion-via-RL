"""Experiment 2: does a learned policy survive a change of physics backend?

Experiment 1 (``divergence.py``) measures how far two engines drift apart under
an *open-loop* control sequence. That is a property of the physics, not of a
policy. This script asks the question a robot-learning venue actually cares
about: take one trained policy, run it closed-loop in two different physics
backends, and measure how much *task performance* changes.

The comparison is tightly controlled. Both conditions load the same Playground
environment, so the observation function, reward terms, termination conditions
and command sampling are byte-identical; only ``config.impl`` differs. Any
performance gap is therefore attributable to the physics backend alone.

This matters because MuJoCo Playground now defaults to ``impl='warp'``, while
the field's sim-to-sim validation convention is to check policies against
MuJoCo/MJX-JAX. If the two backends disagree, that convention is validating
against the wrong reference.

    python -m sim2sim.evaluate --ckpt checkpoints/G1..._impl-jax_seed0 --episodes 64
"""
from __future__ import annotations

import argparse
import functools
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def build_inference_fn(env, ppo_params, params):
    """Reconstruct the policy network and bind the trained parameters."""
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    network = factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    # Deterministic: we are measuring the policy, not its exploration noise.
    return make_policy(params, deterministic=True)


def rollout_batch(env, policy_fn, n_episodes: int, max_steps: int, seed: int):
    """Run ``n_episodes`` in parallel, and report per-episode statistics.

    Episodes are stepped in lockstep for a fixed horizon. Once an episode
    terminates we stop accumulating its reward, so a policy cannot inflate its
    return by being reset and collecting again inside the same rollout.
    """
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    rngs = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    state = reset(rngs)

    total_r = jnp.zeros(n_episodes)
    length = jnp.zeros(n_episodes)
    alive = jnp.ones(n_episodes)  # 1 while the episode has never terminated

    for _ in range(max_steps):
        act, _ = policy_fn(state.obs, jax.random.PRNGKey(0))
        state = step(state, act)
        total_r = total_r + state.reward * alive
        length = length + alive
        # state.done is 1.0 on the terminating step; freeze that episode after it.
        alive = alive * (1.0 - state.done)
        if float(jnp.sum(alive)) == 0.0:
            break

    return {
        "return": np.asarray(total_r),
        "length": np.asarray(length),
        "terminated": np.asarray(1.0 - alive),  # 1 if it fell/ended early
    }


def summarise(stats: dict, max_steps: int) -> dict:
    ret, ln, term = stats["return"], stats["length"], stats["terminated"]
    return {
        "return_mean": float(np.mean(ret)),
        "return_std": float(np.std(ret)),
        "return_sem": float(np.std(ret) / np.sqrt(len(ret))),
        "length_mean": float(np.mean(ln)),
        "termination_rate": float(np.mean(term)),
        "survived_full_horizon": float(np.mean(ln >= max_steps)),
        "n_episodes": int(len(ret)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--ckpt", required=True, help="path passed to brax load_params")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--naconmax", type=int, default=None)
    ap.add_argument(
        "--impls",
        nargs="+",
        default=["jax", "warp"],
        help="physics backends to evaluate the same policy in",
    )
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    from sim2sim import _jax_compat, _wandb

    _jax_compat.install()

    wb = _wandb.init(
        enabled=not args.no_wandb,
        name=f"eval_{pathlib.Path(args.ckpt).name}",
        project=args.wandb_project,
        group=args.env,
        job_type="eval",
        config={
            "env": args.env,
            "ckpt": args.ckpt,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "impls": list(args.impls),
        },
    )

    from brax.io import model as brax_model
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    params = brax_model.load_params(args.ckpt)
    ppo_params = locomotion_params.brax_ppo_config(args.env)

    out = {
        "env": args.env,
        "ckpt": args.ckpt,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "backends": {},
    }

    for impl in args.impls:
        cfg = registry.get_default_config(args.env)
        cfg.impl = impl
        if args.naconmax is not None:
            cfg.naconmax = args.naconmax
        env = registry.load(args.env, config=cfg)

        policy_fn = build_inference_fn(env, ppo_params, params)
        print(f"[eval] impl={impl} ...", flush=True)
        stats = rollout_batch(env, policy_fn, args.episodes, args.max_steps, args.seed)
        summary = summarise(stats, args.max_steps)
        out["backends"][impl] = summary
        out.setdefault("raw", {})[impl] = {
            k: v.tolist() for k, v in stats.items()
        }
        _wandb.summary(wb, {f"{impl}/{k}": v for k, v in summary.items()})
        print(
            f"    return={summary['return_mean']:.2f}+-{summary['return_sem']:.2f}  "
            f"len={summary['length_mean']:.1f}  "
            f"term_rate={summary['termination_rate']:.3f}",
            flush=True,
        )

    # Cross-backend gap. Both backends reset from the same PRNG key, so episode
    # i in one backend and episode i in the other start from an identical state
    # and are a matched pair. The paired comparison is what has power here: the
    # difference of means can sit at ~0 while every individual episode diverges
    # substantially, because per-episode differences cancel. Reporting only the
    # mean gap would report "no effect" in exactly that case.
    if len(args.impls) >= 2:
        ref = args.impls[0]
        ref_ret = np.asarray(out["raw"][ref]["return"])
        for impl in args.impls[1:]:
            ret = np.asarray(out["raw"][impl]["return"])
            diff = ret - ref_ret
            n = len(diff)
            sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n > 1 else 0.0
            rec = {
                "mean_gap": float(np.mean(diff)),
                "mean_gap_sem": float(sem),
                # Mean |difference| does not cancel, so it detects a symmetric
                # spread that the signed mean is blind to.
                "mean_abs_gap": float(np.mean(np.abs(diff))),
                "max_abs_gap": float(np.max(np.abs(diff))),
                # Cohen's d for paired samples; |d| < 0.2 is a negligible shift.
                "paired_effect_size": float(np.mean(diff) / sd) if sd else 0.0,
                "n_pairs": int(n),
                "identical": bool(np.array_equal(ret, ref_ret)),
            }
            out.setdefault("gap_vs_" + ref, {})[impl] = rec
            _wandb.summary(wb, {f"gap_{impl}_vs_{ref}/{k}": v for k, v in rec.items()})
            print(
                f"[gap] {impl} vs {ref}: "
                f"mean={rec['mean_gap']:+.4f}+-{sem:.4f}  "
                f"mean|d|={rec['mean_abs_gap']:.4f}  "
                f"max|d|={rec['max_abs_gap']:.4f}  "
                f"effect={rec['paired_effect_size']:+.3f}",
                flush=True,
            )
            if rec["identical"]:
                print(
                    "    !! WARNING: byte-identical returns across backends. "
                    "Verify config.impl is actually taking effect.",
                    flush=True,
                )

    RESULTS.mkdir(exist_ok=True)
    tag = pathlib.Path(args.ckpt).name
    path = RESULTS / f"eval_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {path}")
    _wandb.finish(wb)


if __name__ == "__main__":
    main()
