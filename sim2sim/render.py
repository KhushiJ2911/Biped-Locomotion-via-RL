"""Render rollouts to video -- including the same motion under two engines.

Two modes:

  trajectory mode  render qpos trajectories already saved by the divergence
                   sweep. No policy or GPU work needed.

  policy mode      roll a trained checkpoint out under each backend and render
                   the result.

The side-by-side output is the point. A table showing "3.4e-03 rad of base
orientation error" is hard to feel; two humanoids started from an identical
state visibly drifting apart is not.

    python -m sim2sim.render --traj results/traj_G1JoystickFlatTerrain.npz \\
        --condition baseline --engines c jax warp --out results/video.mp4

    python -m sim2sim.render --ckpt checkpoints/..._impl-jax_seed0 \\
        --impls jax warp --out results/policy.mp4
"""
from __future__ import annotations

import argparse
import os
import pathlib

# MuJoCo picks its GL backend at import time. EGL and OSMesa are both broken on
# this machine; glfw works against the running X display.
os.environ.setdefault("MUJOCO_GL", "glfw")

# Rendering saved trajectories needs no JAX compute, but importing
# mujoco_playground initialises JAX, which will grab GPU memory and can starve a
# concurrent training run on a 4GB card. Pin JAX to CPU unless the caller has
# already chosen a platform; policy mode overrides this explicitly below.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def render_traj(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    width: int,
    height: int,
    stride: int,
    camera: str | int = -1,
    track_body: str | None = None,
    cam_distance: float = 3.5,
) -> list[np.ndarray]:
    """Render a qpos trajectory to a list of RGB frames."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    if isinstance(camera, str) and camera:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id >= 0:
            cam.fixedcamid, cam.type = cam_id, mujoco.mjtCamera.mjCAMERA_FIXED

    # A free-floating robot walks out of a static frame within a second, so the
    # camera follows the base body unless a fixed camera was requested.
    body_id = -1
    if track_body:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)

    # For the fixed camera, anchor on the robot's *initial* position and hold it
    # there. MuJoCo's default camera sits too close and crops the model, and a
    # camera that re-centres every frame would subtract out exactly the
    # base-position divergence a fixed camera is meant to reveal.
    if body_id < 0:
        data.qpos[:] = qpos[0]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.subtree_com[0] if model.nbody > 1 else qpos[0][:3]
        cam.distance, cam.elevation, cam.azimuth = cam_distance, -15.0, 130.0

    frames = []
    for i in range(0, len(qpos), stride):
        data.qpos[:] = qpos[i]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        if body_id >= 0:
            cam.lookat[:] = data.xpos[body_id]
            cam.distance, cam.elevation, cam.azimuth = cam_distance, -15.0, 130.0
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    renderer.close()
    return frames


def label(frames: list[np.ndarray], text: str) -> list[np.ndarray]:
    """Burn a text label into the top-left of each frame.

    Uses PIL when present; silently returns the frames unchanged otherwise,
    since a missing font is not worth failing a render over.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frames
    out = []
    for f in frames:
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 8 * len(text) + 10, 22], fill=(0, 0, 0))
        d.text((5, 5), text, fill=(255, 255, 255))
        out.append(np.asarray(img))
    return out


def stack_side_by_side(clips: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Horizontally concatenate clips, truncating to the shortest."""
    n = min(len(c) for c in clips)
    return [np.concatenate([c[i] for c in clips], axis=1) for i in range(n)]


def write_video(frames: list[np.ndarray], path: pathlib.Path, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 stops ffmpeg silently resizing odd dimensions, which
    # would otherwise misalign a side-by-side composition.
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)
    print(f"[saved] {path}  ({len(frames)} frames @ {fps}fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--traj", default=None, help="npz from the divergence sweep")
    ap.add_argument("--condition", default="baseline", help="condition inside the npz")
    ap.add_argument("--engines", nargs="+", default=["c", "jax", "warp"])
    ap.add_argument("--ckpt", default=None, help="checkpoint for policy mode")
    ap.add_argument("--impls", nargs="+", default=["jax", "warp"])
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="reset seed in policy mode")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth step")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--cam-distance", type=float, default=3.5,
                    help="camera distance (m). A walking policy travels several\nmetres, so a fixed camera needs more distance than a standing one.")
    ap.add_argument("--naconmax", type=int, default=8192)
    ap.add_argument(
        "--no-track",
        action="store_true",
        help="fixed world camera. Use this for divergence videos: a camera that "
        "follows each robot's own base subtracts out base-position error, which "
        "is precisely the quantity being demonstrated.",
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="allow JAX on the GPU (policy mode). Off by default so rendering "
        "cannot starve a concurrent training run.",
    )
    args = ap.parse_args()

    if args.gpu:
        os.environ.pop("JAX_PLATFORMS", None)
    elif args.ckpt:
        print("[render] JAX pinned to CPU; pass --gpu if no training is running")

    if not args.traj and not args.ckpt:
        raise SystemExit("need either --traj (trajectory mode) or --ckpt (policy mode)")

    from mujoco_playground import registry

    cfg = registry.get_default_config(args.env)
    cfg.naconmax = args.naconmax
    env = registry.load(args.env, config=cfg)
    model = env.mj_model
    track = None if args.no_track else model.body(1).name

    clips = []
    traj_out = {}
    if args.traj:
        blob = np.load(args.traj)
        available = sorted({k.split("|")[0] for k in blob.files})
        if args.condition not in available:
            raise SystemExit(
                f"condition {args.condition!r} not in {args.traj}.\n"
                f"available: {available}"
            )
        # Sweeps predating the three-way rewrite stored MuJoCo-C as "mjc" and
        # MJX-JAX as "mjx". Accept both spellings so old runs stay renderable.
        aliases = {"c": ["c", "mjc"], "jax": ["jax", "mjx"], "warp": ["warp"]}
        for eng in args.engines:
            key = next(
                (f"{args.condition}|qpos_{a}" for a in aliases.get(eng, [eng])
                 if f"{args.condition}|qpos_{a}" in blob.files),
                None,
            )
            if key is None:
                print(f"[skip] no trajectory for engine {eng!r} in this npz")
                continue
            print(f"[render] {key}")
            fr = render_traj(model, blob[key], args.width, args.height,
                             args.stride, track_body=track,
                             cam_distance=args.cam_distance)
            clips.append(label(fr, eng.upper()))
        default_name = f"sim2sim_{args.env}_{args.condition}.mp4"
    else:
        import jax

        from sim2sim import _jax_compat

        _jax_compat.install()
        from brax.io import model as brax_model
        from mujoco_playground.config import locomotion_params

        from sim2sim.evaluate import build_inference_fn

        params = brax_model.load_params(args.ckpt)
        ppo_params = locomotion_params.brax_ppo_config(args.env)

        for impl in args.impls:
            c = registry.get_default_config(args.env)
            c.impl, c.naconmax = impl, args.naconmax
            e = registry.load(args.env, config=c)
            policy = build_inference_fn(e, ppo_params, params)

            reset, step = jax.jit(e.reset), jax.jit(e.step)
            state = reset(jax.random.PRNGKey(args.seed))
            qs = [np.asarray(state.data.qpos)]
            print(f"[rollout] impl={impl}")
            for _ in range(args.episode_steps):
                act, _ = policy(state.obs, jax.random.PRNGKey(0))
                state = step(state, act)
                qs.append(np.asarray(state.data.qpos))
                if float(state.done) > 0:
                    break
            fr = render_traj(model, np.array(qs), args.width, args.height,
                             args.stride, track_body=track,
                             cam_distance=args.cam_distance)
            traj_out[impl] = np.array(qs)
            clips.append(label(fr, impl.upper()))
        default_name = f"policy_{pathlib.Path(args.ckpt).name}.mp4"
        if traj_out:
            npz = RESULTS / f"policytraj_{pathlib.Path(args.ckpt).name}.npz"
            np.savez_compressed(npz, **{f'qpos_{k}': v for k, v in traj_out.items()})
            print(f'[saved] {npz}')

    if not clips:
        raise SystemExit("nothing rendered")

    frames = stack_side_by_side(clips) if len(clips) > 1 else clips[0]
    write_video(frames, pathlib.Path(args.out or (RESULTS / default_name)), args.fps)


if __name__ == "__main__":
    main()
