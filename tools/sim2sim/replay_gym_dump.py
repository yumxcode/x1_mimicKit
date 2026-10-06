"""Open-loop action replay: gym-engine dump vs local MuJoCo.

Sets the local sim EXACTLY to the dump's init state (positions AND
velocities), applies the dumped action sequence frame-by-frame (30 Hz),
and compares the resulting state trajectory with the dump. Any divergence
isolates the dynamics/PD/action path (the policy is out of the loop).

Run: .venv/bin/python tools/sim2sim/replay_gym_dump.py /tmp/traj_ep0.pt
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from tools.sim2sim.sim2sim_x1 import (  # noqa: E402
    X1Sim, parse_x1_xml, parse_urdf_velocity, quat_mul)


def set_exact(sim, init):
    if isinstance(init, np.ndarray):
        init = init.item()  # pickled as 0-d object array
    init = {k: np.asarray(v, dtype=float).flatten()
            for k, v in init.items()}
    mujoco.mj_resetData(sim.m, sim.d)
    sim.d.qpos[0:3] = init["root_pos"]
    # dump root_rot is xyzw (isaac_gym raw); MuJoCo freejoint wants wxyz
    q = init["root_rot"] / np.linalg.norm(init["root_rot"])
    sim.d.qpos[3:7] = np.roll(q, 1)
    sim.d.qpos[sim.qadr] = init["dof"]
    sim.d.qvel[0:3] = init["root_vel"]
    mujoco.mj_forward(sim.m, sim.d)
    R = sim.d.xmat[sim.base_bid].reshape(3, 3)
    # engine ang vel is world-frame; MuJoCo freejoint qvel[3:6] is local
    sim.d.qvel[3:6] = R.T @ init["root_ang_vel"]
    sim.d.qvel[sim.vadr] = init["dof_vel"]
    mujoco.mj_forward(sim.m, sim.d)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/traj_ep0.pt"
    d = pickle.load(open(path, "rb"))
    if os.environ.get("X1_REPLAY_HOLD"):
        actions = np.tile(np.asarray(d["dof"])[0:1], (300, 1))
    else:
        actions = np.asarray(d.get("action", []))
    rp = np.asarray(d["root_pos"])
    dof = np.asarray(d["dof"])
    T = min(len(actions), len(rp) - 1)
    print(f"replay {path}: {T} steps")
    if T == 0:
        print("NO ACTIONS RECORDED - dump needs 'action' field")
        return

    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    sim = X1Sim(spec)
    if os.environ.get("X1_REPLAY_NOFRICTION"):
        for jid in range(sim.m.njnt):
            dofadr = sim.m.jnt_dofadr[jid]
            if dofadr >= 0:
                sim.m.dof_frictionloss[dofadr] = 0.0
    if os.environ.get("X1_REPLAY_NOCLIP"):
        sim.m.actuator_forcelimited[:] = 0
    set_exact(sim, d["init"])

    dz, dj = [], []
    for t in range(T):
        sim.apply_action(actions[t])
        sim.step_sim()  # 30 Hz step
        dz.append(abs(sim.d.qpos[2] - rp[t + 1, 2]))
        dj.append(np.abs(sim.d.qpos[sim.qadr] - dof[t + 1]).max())
    dz, dj = np.asarray(dz), np.asarray(dj)
    print("step | droot_z | max_dof_err")
    for k in list(range(0, T, max(1, T // 12))) + [T - 1]:
        print(f"{k:4d} | {dz[k]:.4f} | {dj[k]:.4f}")
    print(f"first frame droot_z>0.10 at:",
          next((k for k in range(T) if dz[k] > 0.10), None))
    print(f"first frame max_dof>0.20 at:",
          next((k for k in range(T) if dj[k] > 0.20), None))


if __name__ == "__main__":
    main()
