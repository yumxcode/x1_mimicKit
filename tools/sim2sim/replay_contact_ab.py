"""Contact-parameter A/B on the corrected open-loop replay.

The replay shows joints track well (mean err ~0.04 rad) while the base
falls at step 14-18 — a balance/contact-level divergence. MuJoCo contacts
are soft (solref 0.02 1 by default) while PhysX plane contacts are rigid.
Sweep contact solref/friction to see if a stiffer twin reproduces the
engine trajectory.

Run: .venv/bin/python tools/sim2sim/replay_contact_ab.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from tools.sim2sim.sim2sim_x1 import (  # noqa: E402
    X1Sim, parse_x1_xml, parse_urdf_velocity)
from tools.sim2sim.replay_gym_dump import set_exact  # noqa: E402


def run(d, solref, friction, N=60):
    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    sim = X1Sim(spec)
    m = sim.m
    if solref is not None:
        for g in range(m.ngeom):
            m.geom_solref[g] = solref
            m.geom_solimp[g] = (0.9, 0.95, 0.001, 0.5, 2.0)
    if friction is not None:
        for g in range(m.ngeom):
            m.geom_friction[g] = friction
    set_exact(sim, d["init"])
    actions = np.asarray(d["action"])
    rp = np.asarray(d["root_pos"])
    dof = np.asarray(d["dof"])
    dz, mj = [], []
    for t in range(min(N, len(actions))):
        sim.apply_action(actions[t])
        sim.step_sim()
        dz.append(abs(sim.d.qpos[2] - rp[t][2]))
        mj.append(np.abs(sim.d.qpos[sim.qadr] - dof[t]).max())
    dz, mj = np.asarray(dz), np.asarray(mj)
    fall = next((k for k in range(len(dz)) if dz[k] > 0.10), None)
    return float(mj[:8].mean()), float(dz[:20].max()), fall


def main():
    d = pickle.load(open("/tmp/d3_traj_ep0.pt", "rb"))
    variants = [
        ("default", None, None),
        ("solref 0.01", (0.01, 1.0), None),
        ("solref 0.004", (0.004, 1.0), None),
        ("solref 0.002", (0.002, 1.0), None),
        ("friction 1.5", None, (1.5, 0.05, 0.05)),
        ("solref 0.004 + fric 1.5", (0.004, 1.0), (1.5, 0.05, 0.05)),
    ]
    for name, solref, fric in variants:
        try:
            m8, dz20, fall = run(d, solref, fric)
            print(f"{name:26s}: first8 mean_dof {m8:.3f} | dz20 max "
                  f"{dz20:.3f} | fall@{fall}", flush=True)
        except Exception as e:
            print(f"{name:26s}: FAIL {e}", flush=True)


if __name__ == "__main__":
    main()
