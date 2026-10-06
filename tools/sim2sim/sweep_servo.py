"""Sweep MuJoCo servo settings to match the gym-engine dump trajectory.

The engine (PhysX pos drive) tracks position targets aggressively (~0.2-
0.5 rad per 33ms control step, effectively unclipped torque). Sweep
effort-clip x kp-scale x kd-scale to minimize open-loop replay divergence
against the dumped trajectory (first N steps).

Run: .venv/bin/python tools/sim2sim/sweep_servo.py /tmp/d3_traj_ep0.pt
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
    X1Sim, parse_x1_xml, parse_urdf_velocity)
from tools.sim2sim.replay_gym_dump import set_exact  # noqa: E402


def run(d, clip, kp_scale, kd_scale, N=60):
    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    sim = X1Sim(spec)
    # rescale gains
    for i in range(len(spec["names"])):
        jid = sim.m.joint(spec["names"][i]).id
        kp = float(spec["kp"][i]) * kp_scale
        kd = float(spec["kd"][i]) * kd_scale
        sim.m.dof_damping[sim.m.joint(spec["names"][i]).dofadr[0]] = kd
        sim.m.actuator_gainprm[i, 0] = kp
        sim.m.actuator_biasprm[i, 1] = -kp
        sim.m.actuator_ctrllimited[i] = 0
        if clip:
            sim.m.actuator_forcerange[i, 0] = -spec["effort"][i]
            sim.m.actuator_forcerange[i, 1] = spec["effort"][i]
            sim.m.actuator_forcelimited[i] = 1
        else:
            sim.m.actuator_forcelimited[i] = 0
    set_exact(sim, d["init"])
    actions = np.asarray(d["action"])
    dof = np.asarray(d["dof"])
    rp = np.asarray(d["root_pos"])
    errs_d, errs_z = [], []
    for t in range(min(N, len(actions))):
        sim.d.ctrl[:] = np.clip(actions[t], spec["a_low"], spec["a_high"])
        for _ in range(4):
            mujoco.mj_step(sim.m, sim.d)
        errs_d.append(np.abs(sim.d.qpos[sim.qadr] - dof[t + 1]).mean())
        errs_z.append(abs(sim.d.qpos[2] - rp[t + 1, 2]))
    return float(np.mean(errs_d)), float(np.max(errs_z))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d3_traj_ep0.pt"
    d = pickle.load(open(path, "rb"))
    if isinstance(d["init"], np.ndarray):
        pass
    results = []
    for clip in (True, False):
        for kp_s in (1.0, 2.0, 4.0, 8.0):
            for kd_s in (1.0, 2.0):
                try:
                    md, mz = run(d, clip, kp_s, kd_s)
                    results.append((md, mz, clip, kp_s, kd_s))
                    print(f"clip={int(clip)} kp x{kp_s} kd x{kd_s}: "
                          f"mean_dof_err {md:.4f} max_dz {mz:.3f}",
                          flush=True)
                except Exception as e:
                    print(f"clip={int(clip)} kp x{kp_s} kd x{kd_s}: FAIL {e}",
                          flush=True)
    results.sort()
    print("\nBEST 5:")
    for md, mz, clip, kp_s, kd_s in results[:5]:
        print(f"  clip={int(clip)} kp x{kp_s} kd x{kd_s}: "
              f"mean_dof_err {md:.4f} max_dz {mz:.3f}")


if __name__ == "__main__":
    main()
