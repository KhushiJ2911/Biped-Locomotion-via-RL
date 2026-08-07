"""Unified rollout wrappers so MuJoCo-C and MJX can be driven identically.

Both engines are built from the *same* mj_model, so any state divergence is
attributable to the engine implementation (and whatever physics option we
deliberately perturb), not to a model mismatch.
"""
from __future__ import annotations

import copy
import dataclasses

import jax
import mujoco
import numpy as np
from mujoco import mjx


@dataclasses.dataclass
class PhysicsOpts:
    """Physics options we sweep in the attribution study.

    ``None`` means "leave the model's own value alone".
    """

    iterations: int | None = None
    ls_iterations: int | None = None
    solver: int | None = None  # 0=PGS, 1=CG, 2=Newton
    cone: int | None = None  # 0=pyramidal, 1=elliptic
    integrator: int | None = None  # 0=Euler, 1=RK4, 2=implicit, 3=implicitfast
    timestep: float | None = None
    friction_scale: float | None = None  # multiplies geom_friction
    solref_scale: float | None = None  # multiplies geom_solref (contact softness)
    armature_scale: float | None = None  # multiplies dof_armature

    def apply(self, model: mujoco.MjModel) -> mujoco.MjModel:
        """Return a copy of ``model`` with these options applied."""
        m = _copy_model(model)
        if self.iterations is not None:
            m.opt.iterations = self.iterations
        if self.ls_iterations is not None:
            m.opt.ls_iterations = self.ls_iterations
        if self.solver is not None:
            m.opt.solver = self.solver
        if self.cone is not None:
            m.opt.cone = self.cone
        if self.integrator is not None:
            m.opt.integrator = self.integrator
        if self.timestep is not None:
            m.opt.timestep = self.timestep
        # NOTE: explicit <pair> elements override the per-geom contact params.
        # Playground bipeds define foot<->floor as explicit pairs, so scaling
        # geom_friction/geom_solref alone is a silent no-op on the contacts
        # that actually carry the robot. Both arrays must be scaled.
        if self.friction_scale is not None:
            m.geom_friction[:] = m.geom_friction * self.friction_scale
            if m.npair:
                m.pair_friction[:] = m.pair_friction * self.friction_scale
        if self.solref_scale is not None:
            m.geom_solref[:] = m.geom_solref * self.solref_scale
            if m.npair:
                m.pair_solref[:] = m.pair_solref * self.solref_scale
        if self.armature_scale is not None:
            m.dof_armature[:] = m.dof_armature * self.armature_scale
        return m

    def label(self) -> str:
        parts = [
            f"{f.name}={getattr(self, f.name)}"
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        ]
        return ",".join(parts) if parts else "baseline"


def _copy_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Deep-copy an MjModel so option edits don't leak into the caller's model."""
    return copy.deepcopy(model)


def rollout_mjc(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out MuJoCo-C. Returns (qpos[T+1, nq], qvel[T+1, nv])."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)

    T = ctrl_seq.shape[0]
    qpos = np.zeros((T + 1, model.nq))
    qvel = np.zeros((T + 1, model.nv))
    qpos[0], qvel[0] = data.qpos.copy(), data.qvel.copy()

    for t in range(T):
        data.ctrl[:] = ctrl_seq[t]
        mujoco.mj_step(model, data)
        qpos[t + 1], qvel[t + 1] = data.qpos.copy(), data.qvel.copy()
    return qpos, qvel


class ConstraintOverflow(RuntimeError):
    """Raised when the solver dropped constraints, which corrupts the physics."""


def _capture_fd1(fn):
    """Run ``fn`` capturing file-descriptor 1, returning (result, text).

    MuJoCo's "nefc overflow" warning is emitted by the C/Warp library straight
    to fd 1, so neither ``contextlib.redirect_stdout`` nor a stderr redirect
    sees it. Without capturing at the descriptor level the warning scrolls past
    while the rollout returns a plausible-looking but physically wrong
    trajectory -- which is exactly how it went unnoticed across three Op3 sweeps.
    """
    import os
    import tempfile

    with tempfile.TemporaryFile(mode="w+") as tf:
        saved = os.dup(1)
        os.dup2(tf.fileno(), 1)
        try:
            result = fn()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        tf.seek(0)
        return result, tf.read()


def rollout_mjx(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl_seq: np.ndarray,
    impl: str = "jax",
    njmax: int | None = None,
    naconmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out MJX from the same mj_model. Returns (qpos[T+1, nq], qvel[T+1, nv]).

    ``impl`` selects the MJX backend: "jax" (the classic XLA reimplementation),
    "warp" (MuJoCo Warp -- what MuJoCo Playground now defaults to), or "cpp"
    (a thin wrapper over the same C engine as ``rollout_mjc``, useful as a
    null-gap control: it should agree with MuJoCo-C to round-off).

    Passing ``impl=None`` to ``put_model`` auto-resolves, which currently picks
    JAX; we name it explicitly so the recorded engine can never drift from the
    label we report.
    """
    mx = mjx.put_model(model, impl=impl)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)
    # njmax sizes the constraint buffer. If it is too small the solver silently
    # DROPS constraints and prints "nefc overflow" to stderr -- the rollout still
    # returns, but the physics is wrong. Robots with more contact geoms need
    # more: Op3's four box feet overflow the default that G1, T1 and Berkeley
    # fit inside. Callers running a sweep should set this explicitly rather than
    # trust a per-robot default.
    dx = mjx.put_data(model, data, impl=impl, njmax=njmax, naconmax=naconmax)

    step = jax.jit(mjx.step)
    T = ctrl_seq.shape[0]
    qpos = np.zeros((T + 1, model.nq))
    qvel = np.zeros((T + 1, model.nv))
    qpos[0], qvel[0] = np.asarray(dx.qpos), np.asarray(dx.qvel)

    def _run():
        nonlocal dx
        for t in range(T):
            dx = dx.replace(ctrl=jax.numpy.asarray(ctrl_seq[t]))
            dx = step(mx, dx)
            qpos[t + 1], qvel[t + 1] = np.asarray(dx.qpos), np.asarray(dx.qvel)

    _, captured = _capture_fd1(_run)
    if "nefc overflow" in captured:
        need = None
        for tok in captured.split():
            if tok.rstrip(".").isdigit():
                need = max(need or 0, int(tok.rstrip(".")))
        raise ConstraintOverflow(
            f"solver dropped constraints during a {impl!r} rollout "
            f"(njmax={njmax}); needs at least {need}. The returned trajectory "
            f"would be physically wrong, so this is raised rather than warned. "
            f"Re-run with a larger --njmax."
        )
    return qpos, qvel


# ---------------------------------------------------------------- metrics


def quat_angle(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between two arrays of wxyz quaternions.

    Uses the atan2 form rather than ``2*arccos(|dot|)``. arccos loses almost all
    precision as dot -> 1, which is exactly the near-agreement regime we care
    about here: it silently floors small-but-real angles to exactly 0. With
    q_b' = q_b*sign(dot),

        ||q_a - q_b'|| = 2 sin(t/4),  ||q_a + q_b'|| = 2 cos(t/4)

    so t = 4*atan2(||q_a - q_b'||, ||q_a + q_b'||), which stays well-conditioned
    all the way down to machine epsilon.
    """
    dot = np.sum(q_a * q_b, axis=-1, keepdims=True)
    q_b = q_b * np.sign(np.where(dot == 0.0, 1.0, dot))  # resolve double cover
    d_minus = np.linalg.norm(q_a - q_b, axis=-1)
    d_plus = np.linalg.norm(q_a + q_b, axis=-1)
    return 4.0 * np.arctan2(d_minus, d_plus)


def divergence_metrics(
    qpos_a: np.ndarray,
    qpos_b: np.ndarray,
    qvel_a: np.ndarray,
    qvel_b: np.ndarray,
    free_base: bool = True,
) -> dict[str, np.ndarray]:
    """Per-timestep divergence between two rollouts of the same model."""
    if free_base:
        base_pos = np.linalg.norm(qpos_a[:, 0:3] - qpos_b[:, 0:3], axis=-1)
        base_rot = quat_angle(qpos_a[:, 3:7], qpos_b[:, 3:7])
        joints_a, joints_b = qpos_a[:, 7:], qpos_b[:, 7:]
    else:
        base_pos = np.zeros(len(qpos_a))
        base_rot = np.zeros(len(qpos_a))
        joints_a, joints_b = qpos_a, qpos_b

    joint_rmse = np.sqrt(np.mean((joints_a - joints_b) ** 2, axis=-1))
    vel_rmse = np.sqrt(np.mean((qvel_a - qvel_b) ** 2, axis=-1))
    return {
        "base_pos_err_m": base_pos,
        "base_rot_err_rad": base_rot,
        "joint_rmse_rad": joint_rmse,
        "qvel_rmse": vel_rmse,
    }
