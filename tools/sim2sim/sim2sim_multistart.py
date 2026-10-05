"""Multi-start sim2sim survival statistics.

Runs the SAME policy from N different motion frames (with motion
velocities) and reports per-start survival time + aggregate stats. This
removes the high variance of judging S1 from a single fixed frame.

Usage: .venv/bin/python tools/sim2sim/sim2sim_multistart.py \
    --model data/models/smp_policies/x1_policy_6k1.pt [--starts 8] [--duration 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pickle  # noqa: E402
from tools.sim2sim.sim2sim_x1 import (  # noqa: E402
    FPS, SUBSTEPS, parse_urdf_velocity, parse_x1_xml, load_policy,
    normalize, unnormalize, actor_forward)
import math  # noqa: E402


def run_start(policy, spec, frames, fi, fps, max_steps):
    from tools.sim2sim.sim2sim_x1 import X1Sim
    import mujoco
    sim = X1Sim(spec)
    sim.set_init(frames[fi], frames[fi + 1] if fi + 1 < len(frames) else None,
                 fps=float(fps))
    ankle_bodies = set()
    for s in ("left", "right"):
        ankle_bodies.add(sim.m.body(f"{s}_ankle_roll_link").id)
        ankle_bodies.add(sim.m.body(f"{s}_ankle_pitch_link").id)
    foot_bodies = {sim.foot_bids["left"], sim.foot_bids["right"]}
    survived = 0
    for step in range(max_steps):
        obs = sim.build_obs()
        na = actor_forward(policy, normalize(policy, obs, "obs"))
        a = unnormalize(policy, na, "a")
        sim.apply_action(a)
        for _ in range(SUBSTEPS):
            sim.step_sim()
        root_pos = sim.d.qpos[0:3].copy()
        w = sim.d.qpos[3:7]
        x, y, z = w[1], w[2], w[3]
        body_up_z = 1 - 2 * (x * x + y * y)
        tilt = math.degrees(math.acos(min(1.0, abs(body_up_z))))
        nonfoot = False
        for bid in range(sim.m.nbody):
            name = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, bid)
            if (not name or name == "world" or bid in foot_bodies
                    or bid in ankle_bodies or bid == sim.base_bid):
                continue
            if sim.d.xpos[bid][2] < 0.05:
                nonfoot = True
                break
        if nonfoot or root_pos[2] < 0.35 or tilt > 60:
            break
        survived += 1
    return survived, float(sim.d.qpos[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--motion",
                    default="data/motions/x1_v3/x1_run1_subject5_seg0.pkl")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    policy = load_policy(args.model)

    with open(args.motion, "rb") as f:
        mot = pickle.load(f)
    frames = np.array(mot["frames"])
    fps = float(mot["fps"])
    n = len(frames)
    max_steps = int(args.duration * FPS)

    # evenly spaced start frames across the clip
    fis = np.linspace(0, n - 2, args.starts).astype(int)
    survivals = []
    print(f"[multistart] {args.model} over {args.starts} starts")
    for fi in fis:
        s, z = run_start(policy, spec, frames, int(fi), fps, max_steps)
        survivals.append(s)
        print(f"  frame {fi:4d}: survived {s:4d} steps = {s/FPS:5.2f}s "
              f"(end z={z:.3f})")

    S = np.array(survivals)
    report = dict(
        model=args.model,
        starts=int(args.starts),
        mean_steps=float(S.mean()), max_steps=float(S.max()),
        min_steps=float(S.min()),
        mean_s=float(S.mean() / FPS), max_s=float(S.max() / FPS),
        full_survivals=int((S >= max_steps).sum()),
    )
    print(f"[multistart] mean {report['mean_s']:.2f}s | max {report['max_s']:.2f}s"
          f" | full-run {report['full_survivals']}/{args.starts}")
    out = Path("output") / "multistart.json"
    with open(out, "w") as f:
        import json
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
