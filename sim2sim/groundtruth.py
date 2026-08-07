"""Experiment 0: absolute engine accuracy against closed-form solutions.

Experiments 1 and 2 measure whether the engines *agree*. Neither can say which
one is *right*: MuJoCo-C is treated as the reference by convention, not because
it has been validated against physics. Without hardware, "closer to C" is a
statement about agreement, not accuracy.

These cases have exact analytic solutions, so every engine can be scored
against truth rather than against each other. They also separate two error
sources that the humanoid conflates:

  * ``free_fall`` and ``pendulum`` have **no contacts**, so any error is
    integrator error.
  * ``incline`` is **contact-dominated**, so its error is solver error.

A caveat that must not be glossed: a discrete integrator is not supposed to
reproduce the continuous solution exactly. Semi-implicit Euler on constant
acceleration accumulates a deterministic O(dt) offset -- for free fall,
exactly ``g*dt^2*n/2`` after n steps. That offset is *expected physics*, not
engine error, so we report the residual against the discrete prediction as
well. An engine that matches the discrete prediction is implementing Euler
correctly; one that does not has a real bug.

    python -m sim2sim.groundtruth --engines c jax warp
"""
from __future__ import annotations

import argparse
import json
import pathlib

import mujoco
import numpy as np

from sim2sim.engines import rollout_mjc, rollout_mjx

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

G = 9.81

FREE_FALL_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="{dt}" integrator="Euler"/>
  <worldbody>
    <body name="ball" pos="0 0 10">
      <freejoint/>
      <geom type="sphere" size="0.1" density="1000" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Point mass on a massless rod: a rigid pendulum with an analytic energy
# invariant. contype/conaffinity are zeroed so nothing can collide.
PENDULUM_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="{dt}" integrator="{integrator}"/>
  <worldbody>
    <body name="arm" pos="0 0 2">
      <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.02" density="1"
            contype="0" conaffinity="0"/>
      <body name="bob" pos="0 0 -1">
        <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def free_fall(dt: float, steps: int, engines) -> dict:
    """Body released from rest under gravity. No contacts, no constraints."""
    model = mujoco.MjModel.from_xml_string(FREE_FALL_XML.format(dt=dt))
    q0 = np.zeros(model.nq)
    q0[:3] = [0, 0, 10.0]
    q0[3] = 1.0  # identity quaternion (wxyz)
    v0 = np.zeros(model.nv)
    ctrl = np.zeros((steps, model.nu)) if model.nu else np.zeros((steps, 0))

    n = np.arange(steps + 1)
    t = n * dt
    # Continuous solution.
    z_exact = 10.0 - 0.5 * G * t**2
    # Semi-implicit Euler applies the *updated* velocity over each step, giving
    # sum_{k=1..n} g*dt^2*k = g*dt^2*n(n+1)/2.
    z_discrete = 10.0 - G * dt**2 * n * (n + 1) / 2.0

    out = {}
    for eng in engines:
        if eng == "c":
            qp, _ = rollout_mjc(model, q0, v0, ctrl)
        else:
            qp, _ = rollout_mjx(model, q0, v0, ctrl, impl=eng)
        z = qp[:, 2]
        out[eng] = {
            "err_vs_continuous": float(np.abs(z - z_exact)[-1]),
            "err_vs_discrete_euler": float(np.abs(z - z_discrete)[-1]),
            "final_z": float(z[-1]),
        }
    out["_reference"] = {
        "z_continuous_final": float(z_exact[-1]),
        "z_discrete_euler_final": float(z_discrete[-1]),
        "note": "err_vs_discrete_euler is the one that indicates a bug; "
        "err_vs_continuous is dominated by expected discretisation error.",
    }
    return out


def pendulum(dt: float, steps: int, engines, integrator: str = "implicitfast") -> dict:
    """Rigid pendulum released from horizontal. Scored on energy drift.

    Total mechanical energy is conserved by the true dynamics, so drift is a
    pure integrator artefact and needs no analytic trajectory to detect.
    """
    model = mujoco.MjModel.from_xml_string(
        PENDULUM_XML.format(dt=dt, integrator=integrator)
    )
    # MJX raises NotImplementedError on mjENBL_ENERGY, so the flag cannot be set
    # on the model we simulate. Energy is instead evaluated afterwards on a
    # separate C model that has it enabled -- which is what we want anyway,
    # since it keeps the energy function identical across engines.
    energy_model = mujoco.MjModel.from_xml_string(
        PENDULUM_XML.format(dt=dt, integrator=integrator)
    )
    energy_model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    q0 = np.zeros(model.nq)
    q0[0] = np.pi / 2  # horizontal
    v0 = np.zeros(model.nv)
    ctrl = np.zeros((steps, model.nu)) if model.nu else np.zeros((steps, 0))

    out = {}
    for eng in engines:
        if eng == "c":
            qp, qv = rollout_mjc(model, q0, v0, ctrl)
        else:
            qp, qv = rollout_mjx(model, q0, v0, ctrl, impl=eng)

        # Recompute energy with the C engine for every trajectory, so the
        # energy function itself is identical across engines and only the
        # trajectory differs.
        data = mujoco.MjData(energy_model)
        energies = []
        for i in range(len(qp)):
            data.qpos[:] = qp[i]
            data.qvel[:] = qv[i]
            mujoco.mj_forward(energy_model, data)
            energies.append(float(data.energy[0] + data.energy[1]))
        e = np.array(energies)
        e0 = e[0]
        out[eng] = {
            "energy_initial": float(e0),
            "energy_final": float(e[-1]),
            "abs_drift": float(abs(e[-1] - e0)),
            "rel_drift": float(abs(e[-1] - e0) / abs(e0)) if e0 else float("nan"),
            "max_abs_drift": float(np.max(np.abs(e - e0))),
        }
    return out


INCLINE_XML = """
<mujoco>
  <option gravity="{gx} 0 {gz}" timestep="{dt}" integrator="Euler"
          cone="{cone}" impratio="{impratio}"/>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" friction="{mu} 0.005 0.0001"/>
    <body name="block" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" density="1000"
            friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def incline(
    dt: float,
    steps: int,
    engines,
    theta_deg: float,
    mu: float,
    cone: str = "pyramidal",
    impratio: float = 1.0,
) -> dict:
    """Block on a slope, scored against the analytic Coulomb-friction result.

    Rather than rotating geometry, gravity is tilted by ``theta`` and the floor
    kept flat -- mechanically identical, and it keeps the contact normal exactly
    along +z so the analytic prediction is unambiguous.

    Two regimes, and the static one is the interesting test for locomotion:

      * ``tan(theta) > mu``  -> slides at a = g(sin(theta) - mu*cos(theta))
      * ``tan(theta) <= mu`` -> must not move at all

    Soft contact models permit a static block to *creep* slowly downhill. That
    artefact is exactly what makes a robot's planted foot drift, so measuring it
    is directly relevant rather than academic.
    """
    th = np.radians(theta_deg)
    model = mujoco.MjModel.from_xml_string(
        INCLINE_XML.format(
            gx=G * np.sin(th), gz=-G * np.cos(th), dt=dt, mu=mu,
            cone=cone, impratio=impratio,
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q0, v0 = data.qpos.copy(), np.zeros(model.nv)
    ctrl = np.zeros((steps, model.nu)) if model.nu else np.zeros((steps, 0))

    sliding = np.tan(th) > mu
    a_analytic = G * (np.sin(th) - mu * np.cos(th)) if sliding else 0.0
    t_end = steps * dt
    x_analytic = 0.5 * a_analytic * t_end**2 if sliding else 0.0

    out = {
        "_case": {
            "theta_deg": theta_deg,
            "mu": mu,
            "regime": "sliding" if sliding else "static",
            "a_analytic": float(a_analytic),
            "x_analytic": float(x_analytic),
            "cone": cone,
        }
    }
    for eng in engines:
        if eng == "c":
            qp, qv = rollout_mjc(model, q0, v0, ctrl)
        else:
            qp, qv = rollout_mjx(model, q0, v0, ctrl, impl=eng)
        x = qp[:, 0] - qp[0, 0]
        out[eng] = {
            "x_final": float(x[-1]),
            "x_analytic": float(x_analytic),
            "abs_err_m": float(abs(x[-1] - x_analytic)),
            "vx_final": float(qv[-1, 0]),
            # For the static case this is the creep distance; for sliding it is
            # the relative error in travelled distance.
            "rel_err": (
                float(abs(x[-1] - x_analytic) / x_analytic) if x_analytic else None
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["c", "jax", "warp"])
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    engines = tuple(args.engines)

    results = {"dt": args.dt, "steps": args.steps, "engines": list(engines)}

    print(f"=== free fall (contact-free: integrator error) dt={args.dt} ===")
    ff = free_fall(args.dt, args.steps, engines)
    ref = ff.pop("_reference")
    print(f"  reference: continuous z={ref['z_continuous_final']:.6f}  "
          f"discrete-Euler z={ref['z_discrete_euler_final']:.6f}")
    for eng in engines:
        r = ff[eng]
        print(f"  {eng:5s} final_z={r['final_z']:.6f}  "
              f"|err vs discrete|={r['err_vs_discrete_euler']:.3e}  "
              f"|err vs continuous|={r['err_vs_continuous']:.3e}")
    results["free_fall"] = ff
    results["free_fall_reference"] = ref

    print(f"\n=== pendulum (contact-free: energy conservation) ===")
    pen = pendulum(args.dt, args.steps, engines)
    for eng in engines:
        r = pen[eng]
        print(f"  {eng:5s} E0={r['energy_initial']:.6f} "
              f"Ef={r['energy_final']:.6f}  rel_drift={r['rel_drift']:.3e}")
    results["pendulum"] = pen

    # Contact-dominated cases. Experiment 0's contact-free cases put a floor on
    # engine disagreement (float32 precision); these attribute anything above
    # that floor to the contact solver.
    for name, theta, mu in [("incline_sliding", 30.0, 0.3), ("incline_static", 15.0, 0.8)]:
        res = incline(args.dt, args.steps, engines, theta_deg=theta, mu=mu)
        case = res.pop("_case")
        print(f"\n=== {name}: theta={theta}deg mu={mu} -> {case['regime']} "
              f"(contact solver error) ===")
        if case["regime"] == "sliding":
            print(f"  analytic: a={case['a_analytic']:.4f} m/s^2  "
                  f"x_after_{args.steps*args.dt:.2f}s={case['x_analytic']:.6f} m")
        else:
            print("  analytic: block must not move at all (x = 0)")
        for eng in engines:
            r = res[eng]
            rel = f"{r['rel_err']:.3e}" if r["rel_err"] is not None else "n/a"
            print(f"  {eng:5s} x_final={r['x_final']:+.6e}  "
                  f"abs_err={r['abs_err_m']:.3e} m  rel_err={rel}")
        results[name] = res
        results[name + "_case"] = case

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "groundtruth.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
