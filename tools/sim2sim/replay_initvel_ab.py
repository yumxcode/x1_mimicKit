"""Init-velocity convention A/B on the corrected replay.

The engine dump's init has root_vel/root_ang_vel/dof_vel. If any of these
uses a different frame/sign convention than what set_exact assumes
(world-frame ang vel -> MuJoCo local qvel), the spurious initial spin
topples the robot within ~15 steps. Test variants.

Run: .venv/bin/python tools/sim2sim/replay_initvel_ab.py
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


def set_exact_variant(sim, init, ang_mode):
    init = {k: np.asarray(v, dtype=float).flatten()
            for k, v in (init.item() if isinstance(init, np.ndarray)
                         else init).items()}
    mujoco.mj_resetData(sim.m, sim.d)
    sim.d.qpos[0:3] = init["root_pos"]
    q = init["root_rot"] / np.linalg.norm(init["root_rot"])
    sim.d.qpos[3:7] = np.roll(q, 1)  # xyzw -> wxyz
    sim.d.qpos[sim.qadr] = init["dof"]
    sim.d.qvel[0:3] = init["root_vel"]
    mujoco.mj_forward(sim.m, sim.d)
    R = sim.d.xmat[sim.base_bid].reshape(3, 3)
    w = init["root_ang_vel"]
    if ang_mode == "world":
        sim.d.qvel[3:6] = R.T @ w
    elif ang_mode == "local":
        sim.d.qvel[3:6] = w
    elif ang_mode == "negworld":
        sim.d.qvel[3:6] = -(R.T @ w)
    elif ang_mode == "zero":
        sim.d.qvel[3:6] = 0.0
    sim.d.qvel[sim.vadr] = init["dof_vel"]
    mujoco.mj_forward(sim.m, sim.d)


def run(d, ang_mode, zero_lin=False, zero_dofv=False, N=60):
    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    sim = X1Sim(spec)
    init = d["init"]
    set_exact_variant(sim, init, ang_mode)
    if zero_lin:
        sim.d.qvel[0:3] = 0.0
    if zero_dofv:
        sim.d.qvel[sim.vadr] = 0.0
    actions = np.asarray(d["action"])
    rp = np.asarray(d["root_pos"])
    dz = []
    for t in range(min(N, len(actions))):
        sim.apply_action(actions[t])
        sim.step_sim()
        dz.append(abs(sim.d.qpos[2] - rp[t][2]))
    dz = np.asarray(dz)
    fall = next((k for k in range(len(dz)) if dz[k] > 0.10), None)
    return float(dz[:20].max()), fall


def main():
    d = pickle.load(open("/tmp/d3_traj_ep0.pt", "rb"))
    for name, kw in [
        ("ang=world (current)", dict(ang_mode="world")),
        ("ang=local", dict(ang_mode="local")),
        ("ang=-world", dict(ang_mode="negworld")),
        ("ang=zero", dict(ang_mode="zero")),
        ("ang+lin=zero", dict(ang_mode="zero", zero_lin=True)),
        ("all vel zero", dict(ang_mode="zero", zero_lin=True,
                              zero_dofv=True)),
    ]:
        dz20, fall = run(d, **kw)
        print(f"{name:22s}: dz20 max {dz20:.3f} | fall@{fall}", flush=True)


if __name__ == "__main__":
    main()
