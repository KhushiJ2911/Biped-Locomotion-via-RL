#!/usr/bin/env python3
"""Measure the run-to-run noise floor of policy evaluation.

Everything in the evaluation pipeline is nominally deterministic: fixed PRNG
keys, a deterministic policy, an unchanged model. Repeating an evaluation
should therefore reproduce it exactly. On GPU it does not -- floating-point
reductions are not associative and thread scheduling varies between launches,
so identical inputs give slightly different returns.

That spread is the floor beneath every comparison in this repository. A
cross-backend or cross-configuration difference smaller than it cannot be
resolved no matter how many episodes are averaged, because the measurement
does not reproduce itself at that resolution.

Reported per episode-count so the floor can be quoted at the settings each
experiment actually used.

    python3 tools/measure_noise_floor.py --ckpt checkpoints/..._seed0 \\
        --episodes 128 --max-steps 1000 --repeats 3
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[128],
                    help="one or more episode counts to characterise")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--impl", default="jax")
    ap.add_argument("--naconmax", type=int, default=8192)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    from sim2sim import _jax_compat, _wandb

    _jax_compat.install()
    from brax.io import model as brax_model
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    from sim2sim.evaluate import build_inference_fn, rollout_batch, summarise

    tag = pathlib.Path(args.ckpt).name
    wb = _wandb.init(
        enabled=not args.no_wandb,
        name=f"noisefloor_{tag}",
        group=args.env,
        job_type="noisefloor",
        config={"env": args.env, "ckpt": args.ckpt, "impl": args.impl,
                "max_steps": args.max_steps, "repeats": args.repeats,
                "episode_counts": args.episodes},
    )

    cfg = registry.get_default_config(args.env)
    cfg.impl, cfg.naconmax = args.impl, args.naconmax
    env = registry.load(args.env, config=cfg)
    params = brax_model.load_params(args.ckpt)
    policy = build_inference_fn(env, locomotion_params.brax_ppo_config(args.env), params)

    out = {"env": args.env, "ckpt": args.ckpt, "impl": args.impl,
           "max_steps": args.max_steps, "repeats": args.repeats, "by_episodes": {}}

    for n_ep in args.episodes:
        rets, lens = [], []
        print(f"\n=== {n_ep} episodes x {args.max_steps} steps, "
              f"{args.repeats} identical repeats ===", flush=True)
        for r in range(args.repeats):
            s = summarise(rollout_batch(env, policy, n_ep, args.max_steps, 0),
                          args.max_steps)
            rets.append(s["return_mean"])
            lens.append(s["length_mean"])
            print(f"  repeat {r}: return={s['return_mean']:.6f}  "
                  f"len={s['length_mean']:.3f}", flush=True)

        v = np.array(rets)
        rec = {
            "returns": rets,
            "lengths": lens,
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "range": float(v.max() - v.min()),
            "rel_range_pct": float(100 * (v.max() - v.min()) / abs(v.mean()))
            if v.mean() else float("nan"),
        }
        out["by_episodes"][str(n_ep)] = rec
        print(f"  -> sd={rec['sd']:.6f}  range={rec['range']:.6f}  "
              f"({rec['rel_range_pct']:.2f}% of mean)", flush=True)
        _wandb.summary(wb, {f"ep{n_ep}/{k}": val for k, val in rec.items()
                            if isinstance(val, float)})

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"noisefloor_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {path}")
    print("\nAny effect smaller than the sd above is unresolvable at that "
          "episode count, regardless of statistical treatment.")
    _wandb.finish(wb)


if __name__ == "__main__":
    main()
