"""Render the tolerance curve as a video: one policy, increasing physics mismatch.

The dose-response table says a policy loses 6 % at 3e-03 m of divergence and
fails outright past ~1e-2 m. This shows it. The same policy, from the same
initial state, is rolled out under progressively mismatched physics and the
clips are placed side by side: intact gait on the left, collapse on the right.

Each panel is labelled with its *measured* divergence rather than the parameter
value, so the video and the figure share an axis.

    python -m sim2sim.render_dose --ckpt checkpoints/..._seed0 \\
        --param solref_scale --scales 1.0 1.5 2.0 3.0
"""
from __future__ import annotations

import argparse
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--param", default="solref_scale",
                    choices=["solref_scale", "friction_scale", "armature_scale"])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--episode-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=380)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--cam-distance", type=float, default=7.0)
    ap.add_argument("--naconmax", type=int, default=8192)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.gpu:
        os.environ.pop("JAX_PLATFORMS", None)

    import jax

    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    from sim2sim import _jax_compat

    _jax_compat.install()
    from brax.io import model as brax_model

    from sim2sim.doseresponse import measure_dose, perturb_mjx_model
    from sim2sim.evaluate import build_inference_fn
    from sim2sim.render import label, render_traj, stack_side_by_side, write_video

    cfg = registry.get_default_config(args.env)
    cfg.impl, cfg.naconmax = "jax", args.naconmax
    env = registry.load(args.env, config=cfg)
    base_mj, base_mjx = env.mj_model, env.mjx_model
    track = base_mj.body(1).name

    params = brax_model.load_params(args.ckpt)
    policy = build_inference_fn(env, locomotion_params.brax_ppo_config(args.env), params)

    clips = []
    for scale in args.scales:
        if scale == 1.0:
            env._mjx_model = base_mjx
            dose = 0.0
        else:
            env._mjx_model = perturb_mjx_model(base_mjx, base_mj, args.param, scale)
            dose = measure_dose(base_mj, args.param, scale, 1.0, 3)["base_pos_err_m"]

        reset, step = jax.jit(env.reset), jax.jit(env.step)
        state = reset(jax.random.PRNGKey(args.seed))
        qs = [np.asarray(state.data.qpos)]
        fell_at = None
        for t in range(args.episode_steps):
            act, _ = policy(state.obs, jax.random.PRNGKey(0))
            state = step(state, act)
            qs.append(np.asarray(state.data.qpos))
            if float(state.done) > 0 and fell_at is None:
                fell_at = t
                break
        print(f"[rollout] {args.param}={scale:<5g} dose={dose:.2e} m  "
              f"survived {len(qs) - 1}/{args.episode_steps} steps"
              f"{'' if fell_at is None else f' (fell at {fell_at})'}", flush=True)

        frames = render_traj(base_mj, np.array(qs), args.width, args.height,
                             args.stride, track_body=track, cam_distance=args.cam_distance)
        tag = "reference" if scale == 1.0 else f"{dose:.0e} m"
        clips.append(label(frames, tag))

    env._mjx_model = base_mjx

    # Clips end at different lengths because the perturbed policies fall
    # earlier. Freezing the shorter ones on their final frame keeps the panels
    # in sync, so a robot that has fallen stays visibly down instead of the
    # video simply truncating to the worst case and hiding the difference.
    longest = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (longest - len(c)) for c in clips]

    out = pathlib.Path(args.out or (RESULTS / f"anim_dose_{args.param}.mp4"))
    write_video(stack_side_by_side(clips), out, args.fps)


if __name__ == "__main__":
    main()
