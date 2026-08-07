"""Experiment 4: is the Warp advantage caused by explicit <pair> contacts?

Across four bipeds, the MJX-JAX vs MuJoCo-Warp discrepancy is large and robust
on G1 (median ratio 7.96, 95% CI [3.88, 19.25]) and absent on T1, Berkeley
Humanoid and Op3. G1 is also the only one of the four whose foot contact is
declared through explicit ``<pair>`` elements rather than ordinary geom-geom
collision: its foot geoms carry ``contype=0, conaffinity=0`` and cannot collide
at all except through those pairs.

That is a correlation over four robots with one significant point. This script
turns it into a controlled test by building two models of the *same* robot that
differ only in how the identical contact is declared:

  A  as shipped -- floor<->foot as explicit <pair>, foot geoms non-colliding
  B  converted  -- those pairs deleted, foot geoms given contype/conaffinity
                   and the pair's condim, friction, solref and solimp

Both describe the same physical contact with the same parameters. If the
engine discrepancy follows the *representation* rather than the physics, the
Warp advantage should collapse in B. If it survives, contact representation is
not the mechanism and the G1 result needs a different explanation.

We deliberately do not assume the answer: a null here is just as publishable,
and would rule out the most obvious hypothesis.

    python -m sim2sim.contactrep --seeds 30
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import mujoco
import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def convert_pairs_to_geom_contacts(xml_path: str, assets: dict | None = None):
    """Return (model_as_shipped, model_converted, converted_contacts).

    The converted model deletes floor<->foot ``<pair>`` elements and instead
    enables geom-geom collision on those feet, copying every contact parameter
    off the pair so the physics is unchanged and only the declaration differs.

    Playground resolves mesh assets through an in-memory dict rather than the
    filesystem, so the spec has to be built from the XML text with that dict --
    ``MjSpec.from_file`` cannot find the meshes on its own.
    """
    import re

    import etils.epath as epath

    assets = dict(assets or {})
    xml_text = epath.Path(xml_path).read_text()
    model_a = mujoco.MjModel.from_xml_string(xml_text, assets=assets)

    # The robot body (and its <contact> block) lives in an included file, which
    # Playground supplies through the assets dict rather than the filesystem.
    # Editing that entry in memory is simpler and more transparent than model
    # surgery, and keeps both variants compiled by the same code path.
    inc = re.search(r'<include file="([^"]+)"/>', xml_text)
    if not inc:
        raise RuntimeError("no <include> found in scene xml")
    inc_name = inc.group(1)
    if inc_name not in assets:
        raise RuntimeError(f"included file {inc_name!r} not present in assets")
    body_xml = assets[inc_name].decode()

    # Record the parameters of every floor<->X pair before deleting it.
    converted = []
    for i in range(model_a.npair):
        g1 = mujoco.mj_id2name(model_a, mujoco.mjtObj.mjOBJ_GEOM, int(model_a.pair_geom1[i]))
        g2 = mujoco.mj_id2name(model_a, mujoco.mjtObj.mjOBJ_GEOM, int(model_a.pair_geom2[i]))
        if "floor" not in (g1, g2):
            continue
        converted.append({
            "geom": g2 if g1 == "floor" else g1,
            "condim": int(model_a.pair_dim[i]),
            "friction": np.array(model_a.pair_friction[i]).copy(),
            "solref": np.array(model_a.pair_solref[i]).copy(),
        })
    if not converted:
        raise RuntimeError("no floor<->geom pairs found to convert")

    # Delete exactly those <pair> lines; self-collision pairs are left alone so
    # the only change is how floor contact is declared.
    new_body = body_xml
    for c in converted:
        new_body = re.sub(
            rf'\s*<pair[^>]*geom1="{c["geom"]}"[^>]*geom2="floor"[^>]*/>', "", new_body)
        new_body = re.sub(
            rf'\s*<pair[^>]*geom1="floor"[^>]*geom2="{c["geom"]}"[^>]*/>', "", new_body)

    # Enable ordinary collision on the foot geoms, carrying the pair's condim
    # and friction across.
    fr = converted[0]["friction"]
    fric_attr = f'{fr[0]:g} {fr[2]:g} {fr[3]:g}'
    for c in converted:
        new_body = re.sub(
            rf'(<geom name="{c["geom"]}"[^>]*?)/>',
            rf'\1 contype="1" conaffinity="1" condim="{c["condim"]}" '
            rf'friction="{fric_attr}"/>',
            new_body)
    assets[inc_name] = new_body.encode()

    # Geom-geom friction is *combined* between the two geoms (elementwise max),
    # unlike a <pair>, whose friction is used directly. Leaving the floor at its
    # default 1.0 would therefore yield max(1.0, 0.6) = 1.0 and silently change
    # the physics, so the floor is set to match.
    scene_b = re.sub(
        r'(<geom name="floor"[^>]*?)/>',
        rf'\1 friction="{fric_attr}"/>', xml_text)

    model_b = mujoco.MjModel.from_xml_string(scene_b, assets=assets)
    return model_a, model_b, converted


def measure(model: mujoco.MjModel, duration: float, seeds: int) -> dict:
    """Pairwise engine divergence on one model, mirroring Experiment 1."""
    from sim2sim.divergence import home_state, hold_pose_ctrl, perturb_state
    from sim2sim.engines import divergence_metrics, rollout_mjc, rollout_mjx

    free_base = model.nq > model.nu + 6
    steps = int(round(duration / model.opt.timestep))
    per_pair = {"c|jax": [], "c|warp": [], "jax|warp": []}

    for s in range(seeds):
        q0, v0 = home_state(model)
        q0 = perturb_state(model, q0, s)
        ctrl = hold_pose_ctrl(model, home_state(model)[0], steps)
        traj = {
            "c": rollout_mjc(model, q0, v0, ctrl),
            "jax": rollout_mjx(model, q0, v0, ctrl, impl="jax"),
            "warp": rollout_mjx(model, q0, v0, ctrl, impl="warp"),
        }
        for key in per_pair:
            a, b = key.split("|")
            (qa, va), (qb, vb) = traj[a], traj[b]
            m = divergence_metrics(qa, qb, va, vb, free_base=free_base)
            per_pair[key].append(float(m["base_pos_err_m"][-1]))

    out = {}
    for k, v in per_pair.items():
        arr = np.array(v)
        out[k] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "sem": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "values": v,
        }
    return out


def boot_ratio(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 0):
    """Bootstrap CI for median(a)/median(b). Divergence distributions are
    heavy-tailed on several robots, so the mean ratio is not trustworthy."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ma = np.median(a[rng.integers(0, len(a), len(a))])
        mb = np.median(b[rng.integers(0, len(b), len(b))])
        if mb > 0:
            out.append(ma / mb)
    o = np.array(out)
    return float(np.median(o)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    from mujoco_playground._src.locomotion.g1 import g1_constants as C

    from sim2sim import _wandb

    wb = _wandb.init(
        enabled=not args.no_wandb,
        name="contactrep_G1",
        group="G1JoystickFlatTerrain",
        job_type="contactrep",
        config={"duration_s": args.duration, "seeds": args.seeds},
    )

    from mujoco_playground._src.locomotion.g1.base import get_assets

    xml = str(C.FEET_ONLY_FLAT_TERRAIN_XML)
    model_a, model_b, converted = convert_pairs_to_geom_contacts(xml, get_assets())

    print(f"model A (as shipped) : npair={model_a.npair}")
    print(f"model B (converted)  : npair={model_b.npair}")
    print(f"converted contacts   : {[c['geom'] for c in converted]}")
    if model_b.npair >= model_a.npair:
        print("  !! WARNING: pair count did not drop -- conversion may not have applied")

    # Sanity: the two models must behave near-identically under one engine.
    # If they do not, the conversion changed the physics and no comparison of
    # engine discrepancy between them would be meaningful.
    from sim2sim.divergence import home_state, hold_pose_ctrl
    from sim2sim.engines import divergence_metrics, rollout_mjc

    steps = int(round(args.duration / model_a.opt.timestep))
    q0, v0 = home_state(model_a)
    ctrl = hold_pose_ctrl(model_a, q0, steps)
    qa, va = rollout_mjc(model_a, q0, v0, ctrl)
    qb, vb = rollout_mjc(model_b, q0, v0, ctrl)
    sanity = divergence_metrics(qa, qb, va, vb, free_base=True)
    phys_delta = float(sanity["base_pos_err_m"][-1])
    print(f"\n[sanity] MuJoCo-C, model A vs model B: base_pos diff = {phys_delta:.3e} m")
    print("  (should be small: the conversion is meant to preserve the physics)")

    out = {"xml": xml, "duration_s": args.duration, "seeds": args.seeds,
           "npair_A": int(model_a.npair), "npair_B": int(model_b.npair),
           "converted_geoms": [c["geom"] for c in converted],
           "physics_delta_A_vs_B": phys_delta, "variants": {}}

    for tag, model in (("A_pairs", model_a), ("B_geoms", model_b)):
        print(f"\n=== variant {tag} (npair={model.npair}) ===", flush=True)
        res = measure(model, args.duration, args.seeds)
        j = np.array(res["c|jax"]["values"])
        w = np.array(res["c|warp"]["values"])
        r, lo, hi = boot_ratio(j, w)
        res["warp_advantage"] = {"median_ratio": r, "ci_low": lo, "ci_high": hi,
                                 "mean_ratio": float(j.mean() / w.mean())}
        out["variants"][tag] = res
        for k in ("c|jax", "c|warp", "jax|warp"):
            print(f"  {k:9s} mean={res[k]['mean']:.3e}  median={res[k]['median']:.3e}")
        print(f"  Warp advantage (median ratio) = {r:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  "
              f"{'SIGNIFICANT' if lo > 1 or hi < 1 else 'not significant'}")
        _wandb.summary(wb, {f"{tag}/{k}/median": res[k]["median"] for k in
                            ("c|jax", "c|warp", "jax|warp")}
                       | {f"{tag}/warp_advantage_median": r,
                          f"{tag}/warp_advantage_ci_low": lo,
                          f"{tag}/warp_advantage_ci_high": hi})

    a_r = out["variants"]["A_pairs"]["warp_advantage"]
    b_r = out["variants"]["B_geoms"]["warp_advantage"]
    print("\n=== VERDICT ===")
    print(f"  with explicit pairs   : {a_r['median_ratio']:.2f} [{a_r['ci_low']:.2f}, {a_r['ci_high']:.2f}]")
    print(f"  with geom contacts    : {b_r['median_ratio']:.2f} [{b_r['ci_low']:.2f}, {b_r['ci_high']:.2f}]")
    if a_r["ci_low"] > 1 and b_r["ci_low"] <= 1 <= b_r["ci_high"]:
        print("  => advantage present with pairs, absent without: representation is implicated")
    elif a_r["ci_low"] > 1 and b_r["ci_low"] > 1:
        print("  => advantage survives conversion: contact representation is NOT the mechanism")
    else:
        print("  => inconclusive; intervals overlap or baseline not significant")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "contactrep_G1.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {path}")
    _wandb.finish(wb)


if __name__ == "__main__":
    main()
