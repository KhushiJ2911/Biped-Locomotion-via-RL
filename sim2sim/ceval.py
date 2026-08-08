"""Run a Playground-trained policy closed-loop in MuJoCo-C.

Experiment 2 compared MJX-JAX against MuJoCo-Warp -- two GPU reimplementations
of the same engine. The sim-to-sim protocol the field actually performs
validates against **MuJoCo-C**, the reference implementation, and Experiment 1
shows that is precisely where the discrepancy lives (34/56 conditions). So the
comparison that matters has not yet been run.

Playground environments are MJX-native: ``impl='cpp'`` cannot step, so the
policy cannot simply be pointed at the C engine. This module reproduces the
parts of ``G1JoystickFlatTerrain`` the policy actually depends on -- the 103-dim
``state`` observation, the action-to-motor-target mapping, the gait phase, and
the termination condition -- against ``mujoco.MjData``.

Two configuration choices make the comparison clean rather than merely
plausible:

  observation noise is disabled  (``noise_config.level = 0``)
  random pushes are disabled     (``push_config.enable = False``)

Both are stochastic perturbations that would otherwise differ between the two
implementations for reasons unrelated to physics. With them off the environment
is a deterministic function of (qpos, qvel, phase, last_action, command), so any
divergence is attributable to the engine.

**Nothing here should be trusted until ``validate_observation`` passes.** A
silent mismatch in one of 103 entries would produce a plausible-looking
performance number that measures a reimplementation bug rather than an engine
difference. That check is the point of this module as much as the evaluation is.

    python -m sim2sim.ceval --validate --ckpt checkpoints/..._seed0
"""
from __future__ import annotations

import argparse
import json
import pathlib

import mujoco
import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Slices of the 103-dim observation that this module assembles itself, as
# opposed to reading straight out of MuJoCo's sensor array. Only these can be
# compared against MJX along a trajectory -- see validate_trajectory.
#   0:3 linvel | 3:6 gyro | 6:9 gravity   <- sensor-derived, excluded
#   9:12 command | 12:41 joint pos | 41:70 joint vel | 70:99 last_act
#   99:103 phase
ASSEMBLED_BLOCKS = [(9, 12), (12, 41), (41, 70), (70, 99), (99, 103)]


def sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Read a named sensor out of MjData, mirroring mjx_env.get_sensor_data."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(f"sensor {name!r} not found")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return np.array(data.sensordata[adr : adr + dim])


class CPolicyEnv:
    """G1 joystick environment stepped by MuJoCo-C.

    Only what the *policy* consumes is reproduced. The critic's privileged
    observation and the reward terms are deliberately omitted: the policy reads
    ``state`` alone (``policy_obs_key='state'``), so reproducing the rest would
    add surface area for bugs without affecting what is measured.
    """

    def __init__(self, env, action_scale: float = 0.5, n_substeps: int = 10):
        self.model = env.mj_model
        self.data = mujoco.MjData(self.model)
        self.action_scale = action_scale
        self.n_substeps = n_substeps

        self._default_pose = np.array(env._default_pose)
        self._pelvis_imu_site_id = env._pelvis_imu_site_id
        # phase_dt is NOT a config value: reset() samples a gait frequency
        # U(1.25, 1.5) Hz per episode and derives phase_dt from it. It must
        # therefore be copied from the episode being compared against, or the
        # gait clock silently freezes and the policy sees a constant phase.
        self._phase_dt = None
        self.env = env

        self.nu = self.model.nu
        self.reset()

    def reset(self, qpos=None, qvel=None, phase=None, command=None, phase_dt=None):
        mujoco.mj_resetData(self.model, self.data)
        if qpos is None:
            key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
            if key < 0:
                key = 0
            qpos = self.model.key_qpos[key].copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0 if qvel is None else qvel
        mujoco.mj_forward(self.model, self.data)
        self.phase = np.array([0.0, np.pi]) if phase is None else np.array(phase)
        self.command = np.zeros(3) if command is None else np.array(command)
        self.last_act = np.zeros(self.nu)
        if phase_dt is not None:
            self._phase_dt = np.array(phase_dt)
        return self.observe()

    def observe(self) -> np.ndarray:
        """The 103-dim policy observation, noise-free."""
        d, m = self.data, self.model
        linvel = sensor(m, d, "local_linvel_pelvis")
        gyro = sensor(m, d, "gyro_pelvis")
        # gravity in the IMU frame: R^T @ (0,0,-1) is the third row of R negated
        R = d.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        joint_angles = d.qpos[7:]
        joint_vel = d.qvel[6:]
        phase = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
        return np.concatenate([
            linvel, gyro, gravity, self.command,
            joint_angles - self._default_pose, joint_vel, self.last_act, phase,
        ]).astype(np.float64)

    def step(self, action: np.ndarray):
        if self._phase_dt is None:
            raise ValueError(
                "phase_dt is unset. reset() samples a gait frequency U(1.25, 1.5) "
                "Hz per episode in the MJX env, so it must be copied in via "
                "reset(phase_dt=...). Without it the gait clock never advances "
                "and the policy silently sees a frozen phase."
            )
        motor_targets = self._default_pose + np.asarray(action) * self.action_scale
        self.data.ctrl[:] = motor_targets
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # ORDERING IS LOad-BEARING. In the MJX environment _get_obs runs before
        # info["last_act"] and info["phase"] are updated, so the observation at
        # step t carries the PREVIOUS action and the PRE-advance phase. Updating
        # either before observing shifts two 29- and 4-dim blocks by one step --
        # which reads as a plausible observation and silently measures a
        # reimplementation bug instead of an engine difference.
        obs = self.observe()
        done = self.terminated()

        self.last_act = np.asarray(action).copy()
        self.phase = np.mod(self.phase + self._phase_dt + np.pi, 2 * np.pi) - np.pi
        return obs, done

    # Self-collision sensors the environment also terminates on.
    _CONTACT_TERM_SENSORS = (
        "right_foot_left_foot_found",
        "left_foot_right_shin_found",
        "right_foot_left_shin_found",
    )

    def terminated(self) -> bool:
        """Fallen over, or an illegal self-collision.

        The environment tests ``get_gravity(data, "torso")[-1] < 0``, and
        get_gravity reads the **"upvector_torso" sensor** -- not a site_xmat
        product. The two differ by sign: upright gives the sensor +0.995 but
        ``site_xmat.T @ (0,0,-1)`` gives -0.995. Deriving it from site_xmat
        therefore terminates every upright robot on its first step. The
        observation's *pelvis* gravity really is the site_xmat product, so the
        same word means two different quantities a few lines apart.
        """
        up_torso = sensor(self.model, self.data, "upvector_torso")
        if up_torso[-1] < 0.0:
            return True
        for name in self._CONTACT_TERM_SENSORS:
            try:
                if sensor(self.model, self.data, name)[0] > 0:
                    return True
            except KeyError:
                pass  # not every model declares all three
        return False


def validate_observation(env, cenv: CPolicyEnv, n: int = 8, seed: int = 0) -> dict:
    """Compare the C observation against the MJX environment's, entry by entry.

    Both are evaluated at the *same* qpos/qvel/phase/command/last_act, so a
    correct reimplementation must agree to float32 round-off. Anything larger
    means a term is wrong, and every number produced downstream would be
    measuring that bug rather than an engine difference.
    """
    import jax

    rng = np.random.default_rng(seed)
    worst = 0.0
    worst_idx = -1
    for trial in range(n):
        state = jax.jit(env.reset)(jax.random.PRNGKey(trial))
        qpos = np.asarray(state.data.qpos)
        qvel = np.asarray(state.data.qvel)
        mjx_obs = np.asarray(state.obs["state"])

        cenv.reset(qpos=qpos, qvel=qvel,
                   phase=np.asarray(state.info["phase"]),
                   command=np.asarray(state.info["command"]),
                   phase_dt=np.asarray(state.info["phase_dt"]))
        cenv.last_act = np.asarray(state.info["last_act"])
        c_obs = cenv.observe()

        if c_obs.shape != mjx_obs.shape:
            return {"ok": False, "reason": f"shape {c_obs.shape} vs {mjx_obs.shape}"}
        diff = np.abs(c_obs - mjx_obs)
        if diff.max() > worst:
            worst, worst_idx = float(diff.max()), int(diff.argmax())
    return {"ok": worst < 1e-4, "max_abs_diff": worst, "worst_index": worst_idx,
            "n_trials": n}


def validate_trajectory(env, cenv: CPolicyEnv, policy_fn, steps: int = 100,
                        seed: int = 0) -> dict:
    """Validate the C observation *along a trajectory*, not just at reset.

    ``validate_observation`` only checks the initial state, where phase is
    [0, pi] and last_act is zero in both implementations. That is exactly how a
    frozen gait clock passed unnoticed: the observation was correct at t=0 and
    wrong at every step after.

    Here the MJX environment drives the episode, and at each step its qpos/qvel
    and bookkeeping are copied into the C environment, which then recomputes the
    observation. Because the state is *imposed* rather than integrated, physics
    divergence cannot accumulate -- any mismatch is a reimplementation bug in
    the observation, the phase update, or the action mapping.
    """
    import jax

    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    cenv.reset(qpos=np.asarray(state.data.qpos), qvel=np.asarray(state.data.qvel),
               phase=np.asarray(state.info["phase"]),
               command=np.asarray(state.info["command"]),
               phase_dt=np.asarray(state.info["phase_dt"]))
    step_fn = jax.jit(env.step)

    worst, worst_t, worst_i = 0.0, -1, -1
    phase_worst = 0.0
    for t in range(steps):
        act, _ = policy_fn(state.obs, jax.random.PRNGKey(0))
        state = step_fn(state, act)

        # Impose the MJX physics state, observe with the PRE-update bookkeeping
        # (matching _get_obs's position in the env's step), then advance.
        cenv.data.qpos[:] = np.asarray(state.data.qpos)
        cenv.data.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(cenv.model, cenv.data)

        # Only the blocks this module assembles are comparable here. MuJoCo
        # evaluates sensors BEFORE the final integration, so after mj_step
        # data.sensordata describes the pre-integration state while qpos/qvel
        # describe the post-integration one. CPolicyEnv.step reads sensordata
        # straight after mj_step and therefore mirrors MJX correctly -- but this
        # validator imposes qpos/qvel and calls mj_forward, which recomputes the
        # sensors at the *new* state. Comparing linvel/gyro/gravity here would
        # compare two different instants and fail for a reason unrelated to the
        # code. Those blocks are covered by validate_observation, which
        # evaluates both sides at a single well-defined state (2.4e-07).
        c_obs, m_obs = cenv.observe(), np.asarray(state.obs["state"])
        for a, b in ASSEMBLED_BLOCKS:
            d = np.abs(c_obs[a:b] - m_obs[a:b])
            if d.max() > worst:
                worst, worst_t, worst_i = float(d.max()), t, a + int(d.argmax())

        cenv.last_act = np.asarray(act)
        cenv.phase = np.mod(cenv.phase + cenv._phase_dt + np.pi, 2 * np.pi) - np.pi
        phase_worst = max(phase_worst,
                          float(np.abs(cenv.phase - np.asarray(state.info["phase"])).max()))
        if float(state.done) > 0:
            break

    return {"ok": worst < 1e-4 and phase_worst < 1e-5,
            "max_bookkeeping_diff": worst, "at_step": worst_t, "obs_index": worst_i,
            "max_phase_diff": phase_worst, "steps_checked": t + 1,
            "note": "sensor-derived blocks excluded (MuJoCo sensor staging); "
                    "they are covered by validate_observation"}


def rollout_pair(env, cenv: CPolicyEnv, policy_fn, episode: int,
                 max_steps: int) -> dict:
    """Run one episode from an identical initial state in MJX and in MuJoCo-C.

    HORIZON LIMIT. The environment resamples the velocity command once
    ``info["step"] > 500`` (joystick.py: state.info["command"] = where(step >
    500, sample_command(...), command)). This module holds the command fixed,
    so beyond 500 steps MJX would be tracking a target this function is not
    measuring against while MuJoCo-C still matches the original -- which
    manufactures a large spurious advantage for C (t = -11 at 1000 steps, C
    "better" in 61/64 episodes, correlated with command magnitude). Keep
    max_steps <= 500.

    Performance is scored on episode length, termination and velocity-tracking
    error rather than on reward. Reproducing the ~20 reward terms against MjData
    would add a large surface for exactly the silent reimplementation bugs this
    module already had to fix twice, and would buy nothing: episode length and
    tracking error answer "does the policy still walk" directly, and are the
    metrics the dose-response already uses.
    """
    import jax

    state = jax.jit(env.reset)(jax.random.PRNGKey(episode))
    step_fn = jax.jit(env.step)
    command = np.asarray(state.info["command"])

    cenv.reset(qpos=np.asarray(state.data.qpos), qvel=np.asarray(state.data.qvel),
               phase=np.asarray(state.info["phase"]), command=command,
               phase_dt=np.asarray(state.info["phase_dt"]))

    # --- MJX ---------------------------------------------------------------
    mjx_len, mjx_track = 0, []
    s = state
    for _ in range(max_steps):
        act, _ = policy_fn(s.obs, jax.random.PRNGKey(0))
        s = step_fn(s, act)
        mjx_len += 1
        obs = np.asarray(s.obs["state"])
        mjx_track.append(float(np.linalg.norm(obs[:2] - command[:2])))
        if float(s.done) > 0:
            break

    # --- MuJoCo-C, same initial state, same policy --------------------------
    # The policy is jitted once and reused. brax's inference function is not
    # jitted by default, and calling it eagerly once per step dominated the
    # runtime completely -- 0.24 s/step, versus roughly 1 ms of actual MuJoCo
    # work for 10 substeps of a 29-DoF humanoid.
    key0 = jax.random.PRNGKey(0)
    jit_policy = jax.jit(lambda o: policy_fn({"state": o}, key0)[0])

    c_len, c_track = 0, []
    obs_c = cenv.observe()
    for _ in range(max_steps):
        act = jit_policy(jax.numpy.asarray(obs_c, dtype=jax.numpy.float32))
        obs_c, done_c = cenv.step(np.asarray(act, dtype=np.float64))
        c_len += 1
        c_track.append(float(np.linalg.norm(obs_c[:2] - command[:2])))
        if done_c:
            break

    return {
        "episode": episode,
        "command": command.tolist(),
        "mjx_length": mjx_len, "c_length": c_len,
        "mjx_terminated": mjx_len < max_steps, "c_terminated": c_len < max_steps,
        "mjx_track_err": float(np.mean(mjx_track)) if mjx_track else float("nan"),
        "c_track_err": float(np.mean(c_track)) if c_track else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--ckpt", default=None, help="needed for trajectory validation")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=0,
                    help="run the paired MJX vs MuJoCo-C comparison over N episodes")
    ap.add_argument("--max-steps", type=int, default=500,
                    help="must stay <= 500: the env resamples the command past "
                         "that point, which invalidates the tracking comparison")
    ap.add_argument("--impl", default="jax", choices=["jax", "warp"])
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="continue from episodes already saved on disk")
    ap.add_argument("--naconmax", type=int, default=8192)
    args = ap.parse_args()

    from mujoco_playground import registry

    cfg = registry.get_default_config(args.env)
    cfg.impl, cfg.naconmax = args.impl, args.naconmax
    cfg.noise_config.level = 0.0          # determinism: see module docstring
    cfg.push_config.enable = False
    env = registry.load(args.env, config=cfg)

    cenv = CPolicyEnv(env)
    print(f"[cenv] nu={cenv.nu} obs_dim={cenv.observe().shape[0]} "
          f"(expected {env.observation_size['state'][0]})")

    if args.validate:
        # Persisted, not just printed. The paper quotes these agreement figures
        # as evidence the reimplementation is faithful, and tools/
        # verify_paper_numbers.py can only trace a claim back to a file on disk.
        report = {"env": args.env, "steps": args.steps}

        rep = validate_observation(env, cenv)
        print(f"[validate reset] {rep}")
        print("  PASS" if rep.get("ok") else "  FAIL -- do not evaluate until fixed")
        report["reset"] = rep

        if args.ckpt:
            from brax.io import model as brax_model
            from mujoco_playground.config import locomotion_params

            from sim2sim import _jax_compat
            _jax_compat.install()
            from sim2sim.evaluate import build_inference_fn

            params = brax_model.load_params(args.ckpt)
            pol = build_inference_fn(
                env, locomotion_params.brax_ppo_config(args.env), params)
            rep2 = validate_trajectory(env, cenv, pol, steps=args.steps)
            print(f"[validate trajectory] {rep2}")
            print("  PASS -- observation and phase track MJX along the episode"
                  if rep2.get("ok") else
                  "  FAIL -- bookkeeping diverges over time; fix before evaluating")
            report["trajectory"] = rep2
        else:
            print("  (pass --ckpt to also run the trajectory-wide check)")

        out = RESULTS / f"ceval_validation_{args.env}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=float))
        print(f"[saved] {out}")

    if args.episodes:
        if not args.ckpt:
            raise SystemExit("--episodes requires --ckpt")
        if args.max_steps > 500:
            raise SystemExit(
                f"--max-steps {args.max_steps} exceeds the command-resample "
                "boundary at 500. Past it the two engines track different "
                "commands and the comparison is invalid.")
        from brax.io import model as brax_model
        from mujoco_playground.config import locomotion_params

        from sim2sim import _jax_compat, _wandb
        _jax_compat.install()
        from sim2sim.evaluate import build_inference_fn

        params = brax_model.load_params(args.ckpt)
        pol = build_inference_fn(env, locomotion_params.brax_ppo_config(args.env), params)

        wb = _wandb.init(enabled=not args.no_wandb,
                         name=f"ceval_{pathlib.Path(args.ckpt).name}_{args.impl}",
                         group=args.env, job_type="ceval",
                         config={"env": args.env, "ckpt": args.ckpt,
                                 "episodes": args.episodes, "max_steps": args.max_steps,
                                 "mjx_impl": args.impl})

        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"ceval_{pathlib.Path(args.ckpt).name}_{args.impl}.json"

        # Persist after every episode. At ~88 s per paired episode a 64-episode
        # run takes over an hour, and an interruption that far in should cost one
        # episode rather than the whole run -- which is exactly what happened the
        # first time this was attempted.
        rows = []
        if path.exists() and args.resume:
            try:
                prev = json.loads(path.read_text())
                if prev.get("max_steps") == args.max_steps:
                    rows = prev.get("episodes", [])
                    print(f"  resuming: {len(rows)} episodes already on disk")
            except (json.JSONDecodeError, KeyError):
                pass

        for ep in range(len(rows), args.episodes):
            r = rollout_pair(env, cenv, pol, ep, args.max_steps)
            rows.append(r)
            print(f"  ep {ep:3d}  mjx_len={r['mjx_length']:4d} c_len={r['c_length']:4d}  "
                  f"track mjx={r['mjx_track_err']:.3f} c={r['c_track_err']:.3f}", flush=True)
            path.write_text(json.dumps(
                {"env": args.env, "ckpt": args.ckpt, "mjx_impl": args.impl,
                 "max_steps": args.max_steps, "episodes": rows,
                 "complete": len(rows) == args.episodes}, indent=2))

        def paired(a, b, label):
            """Paired stats for C minus MJX on one metric."""
            d = np.asarray(b) - np.asarray(a)
            n = len(d)
            sd = d.std(ddof=1) if n > 1 else 0.0
            t = d.mean() / (sd / np.sqrt(n)) if sd else float("nan")
            pos = int((d > 0).sum())
            print(f"  {label:22s} MJX {np.mean(a):8.3f}  C {np.mean(b):8.3f}  "
                  f"diff {d.mean():+8.4f}  t({n-1})={t:7.3f}  C worse in {pos}/{n}")
            return {"mjx_mean": float(np.mean(a)), "c_mean": float(np.mean(b)),
                    "paired_diff": float(d.mean()), "t_stat": float(t),
                    "n": n, "c_worse_count": pos}

        print()
        ml = [r["mjx_length"] for r in rows]; cl = [r["c_length"] for r in rows]
        mt = [r["mjx_track_err"] for r in rows]; ct = [r["c_track_err"] for r in rows]
        summary = {
            "episode_length": paired(ml, cl, "episode length"),
            "tracking_error": paired(mt, ct, "velocity tracking err"),
        }
        print(f"  {'termination rate':22s} MJX {np.mean([r['mjx_terminated'] for r in rows]):8.3f}"
              f"  C {np.mean([r['c_terminated'] for r in rows]):8.3f}")
        # Episode length is uninformative when no episode terminates; say so
        # rather than reporting a diff of 0.00 as if it were a measurement.
        if all(not r["mjx_terminated"] and not r["c_terminated"] for r in rows):
            print("  NOTE: no episode terminated in either engine at this horizon, "
                  "so episode length carries no signal here -- tracking error is "
                  "the informative metric. Raise --max-steps to probe survival.")

        out = {"env": args.env, "ckpt": args.ckpt, "mjx_impl": args.impl,
               "max_steps": args.max_steps, "episodes": rows, "summary": summary}
        path.write_text(json.dumps(out, indent=2))
        print(f"[saved] {path}")
        _wandb.summary(wb, {f"{k}/{kk}": vv for k, v in summary.items()
                            for kk, vv in v.items()}); _wandb.finish(wb)


if __name__ == "__main__":
    main()
