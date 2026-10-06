"""Engine-vs-sim2sim observation parity from a gym-engine dump.

Loads traj_ep*.pt (dumped by scripts_remote/dump_policy_traj.py: per-step
root_pos/root_rot(wxyz)/dof + the ENGINE obs), sets the local MuJoCo sim
to each state (velocities by finite diff, same convention as set_init),
rebuilds the sim2sim obs, and reports per-segment max|diff|.

A LARGE mismatch (e.g. root_rot tan-norm segment ~O(1)) means the two
sides compute observations differently (quaternion convention etc.) and
sim2sim results are meaningless. Small (~1e-2, finite-diff velocity
error) means parity holds.

Run: .venv/bin/python tools/sim2sim/parity_gym_dump.py /tmp/traj_ep0.pt
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))



def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/traj_ep0.pt"
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    obs_eng = np.asarray(d["obs"])           # (T, 228) engine obs
    rp = np.asarray(d["root_pos"])           # (T, 3)
    rr = np.asarray(d["root_rot"])           # (T, 4) wxyz
    dof = np.asarray(d["dof"])               # (T, 29)
    T = min(len(obs_eng), len(rp))
    print(f"dump {path}: frames {T}  obs_dim {obs_eng.shape[-1]}")

    from tools.sim2sim.sim2sim_x1 import (
        X1Sim, parse_x1_xml, parse_urdf_velocity,
        quat_mul, axis_angle_to_quat)
    import mujoco

    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    sim = X1Sim(spec)

    dt = 1.0 / 30.0
    diffs = []
    for t in range(T - 1):
        # dump quats are xyzw (isaac_gym raw); MuJoCo wants wxyz
        q = np.roll(rr[t] / np.linalg.norm(rr[t]), 1)
        q1 = np.roll(rr[t + 1] / np.linalg.norm(rr[t + 1]), 1)
        mujoco.mj_resetData(sim.m, sim.d)
        sim.d.qpos[0:3] = rp[t]
        sim.d.qpos[3:7] = q  # wxyz freejoint
        sim.d.qpos[sim.qadr] = dof[t]
        sim.d.qvel[0:3] = (rp[t + 1] - rp[t]) / dt
        dq = quat_mul(q1, np.r_[q[0], -q[1:]])  # q1 * conj(q0)
        if dq[0] < 0:
            dq = -dq
        n = np.linalg.norm(dq[1:])
        angle = 2.0 * np.arctan2(n, dq[0])
        if angle > np.pi:
            angle -= 2.0 * np.pi
        omega_world = (angle / max(n, 1e-9)) * dq[1:] / dt
        mujoco.mj_forward(sim.m, sim.d)
        R = sim.d.xmat[sim.base_bid].reshape(3, 3)
        sim.d.qvel[3:6] = R.T @ omega_world
        sim.d.qvel[sim.vadr] = (dof[t + 1] - dof[t]) / dt
        mujoco.mj_forward(sim.m, sim.d)

        obs_local = np.asarray(sim.build_obs()).flatten()
        obs_local = obs_local[:obs_eng.shape[-1]]
        diffs.append(np.abs(obs_local - obs_eng[t]))

    diffs = np.asarray(diffs)
    segs = [(0, 1, "root_h"), (1, 7, "root_rot_tn"), (7, 10, "root_vel"),
            (10, 13, "root_ang_vel"), (13, 187, "joint_tn"),
            (187, 216, "dof_vel"), (216, 228, "key_pos")]
    print("max|diff| per obs segment (local sim2sim vs gym engine):")
    for a, b, name in segs:
        if b <= diffs.shape[1]:
            print(f"  {name:12s} [{a:3d}:{b:3d}]  max {diffs[:, a:b].max():.4f}")
    print(f"  TOTAL max {diffs.max():.4f}")
    worst = np.unravel_index(diffs.argmax(), diffs.shape)
    print(f"  worst at frame {worst[0]} dim {worst[1]}")


if __name__ == "__main__":
    main()
