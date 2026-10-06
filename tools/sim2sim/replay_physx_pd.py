"""Replay variant: PhysX-style combined PD clip.

force = clip(kp*(ctrl-q) - kd*qdot, +/-effort) entirely inside the
actuator forcerange (dof damping = 0). This replicates Isaac Gym's
DOF_MODE_POS drive semantics where the combined PD output saturates at
the drive force limit, instead of MuJoCo's split (clipped spring +
unclipped implicit damping).

Run: .venv/bin/python tools/sim2sim/replay_physx_pd.py /tmp/d3_traj_ep0.pt
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d3_traj_ep0.pt"
    d = pickle.load(open(path, "rb"))
    init = d["init"]
    if isinstance(init, np.ndarray):
        init = init.item()
    actions = np.asarray(d["action"])
    rp = np.asarray(d["root_pos"])
    dof = np.asarray(d["dof"])

    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    sim = X1Sim(spec)
    m = sim.m
    for i, name in enumerate(spec["names"]):
        jid = m.joint(name).id
        dofadr = m.joint(name).dofadr[0]
        kp = float(spec["kp"][i])
        kd = float(spec["kd"][i])
        # joint-level damping OFF; damping becomes actuator velocity bias
        m.dof_damping[dofadr] = 0.0
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[i, 0] = kp
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[i, 0] = 0.0
        m.actuator_biasprm[i, 1] = -kp
        m.actuator_biasprm[i, 2] = -kd
        m.actuator_ctrllimited[i] = 0
        m.actuator_forcerange[i, 0] = -spec["effort"][i]
        m.actuator_forcerange[i, 1] = spec["effort"][i]
        m.actuator_forcelimited[i] = 1

    set_exact(sim, init)
    dz, mj = [], []
    for t in range(len(actions)):
        sim.d.ctrl[:] = np.clip(actions[t], spec["a_low"], spec["a_high"])
        for _ in range(4):
            mujoco.mj_step(m, sim.d)
        dz.append(abs(sim.d.qpos[2] - rp[t][2]))
        mj.append(np.abs(sim.d.qpos[sim.qadr] - dof[t]).max())
    dz, mj = np.asarray(dz), np.asarray(mj)
    print(' t  | droot_z | max_dof')
    for k in list(range(0, 10)) + [15, 20, 40, 100, 299]:
        if k < len(dz):
            print(f'{k:3d} | {dz[k]:.4f} | {mj[k]:.4f}')
    fall = next((k for k in range(len(dz)) if dz[k] > 0.10), None)
    print('first8 max_dof mean:', mj[:8].mean(), 'max:', mj[:8].max())
    print('root fall (>0.10m) at step:', fall)


if __name__ == "__main__":
    main()
