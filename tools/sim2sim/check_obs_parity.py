"""Parity check: sim2sim build_obs vs MimicKit policy obs (char_env).

Loads a pkl frame, sets the MuJoCo sim to that exact state, builds the
sim2sim observation, and computes the policy observation the way the
training env does (char_env.compute_char_obs on motion data via
motion_lib). Reports per-segment max abs diff.

Run: .venv/bin/python tools/sim2sim/check_obs_parity.py
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mimickit"))

from tools.sim2sim.sim2sim_x1 import X1Sim, parse_x1_xml, parse_urdf_velocity


def main():
    import torch

    pkl = REPO / "data/motions/x1_v3/x1_run1_subject5_seg0.pkl"
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    frames = np.array(d["frames"])

    # use a LATE frame with large joint angles (parity stress: y-axis
    # joints like hip_roll/ankle_roll must match too)
    frame_idx = min(200, len(frames) - 1)

    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])

    sim = X1Sim(spec)

    # ---------- training-side obs: motion_lib + char_env.compute_char_obs
    os.chdir(REPO)
    import anim.motion_lib as motion_lib
    from anim.mjcf_char_model import MJCFCharModel
    import envs.char_env as char_env

    cm = MJCFCharModel(torch.device("cpu"))
    cm.load("data/assets/x1/x1.xml")
    ml = motion_lib.MotionLib(motion_file=str(pkl),
                              kin_char_model=cm, device=torch.device("cpu"))

    key_bodies = ["left_ankle_roll_link", "right_ankle_roll_link",
                  "left_wrist_roll_link", "right_wrist_roll_link"]
    key_ids = [cm.get_body_id(n) for n in key_bodies]

    mid = torch.tensor([0], dtype=torch.long)
    t = torch.tensor([frame_idx / float(d['fps'])])
    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
        ml.calc_motion_frame(mid, t)
    root_pos, root_vel = root_vel[:0].new_tensor([]), None  # placeholder

    # recompute properly (calc_motion_frame returns stacked tensors [1,*])
    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
        ml.calc_motion_frame(mid, t)
    body_pos, _ = cm.forward_kinematics(root_pos=root_pos,
                                        root_rot=root_rot,
                                        joint_rot=joint_rot)
    key_pos = body_pos[:, key_ids, :]

    obs_train_t = char_env.compute_char_obs(
        root_pos=root_pos, root_rot=root_rot, root_vel=root_vel,
        root_ang_vel=root_ang_vel, joint_rot=joint_rot, dof_vel=dof_vel,
        key_pos=key_pos, global_obs=True, root_height_obs=True)
    obs_train = obs_train_t[0].numpy()

    # ---------- sim2sim-side obs: set sim state = frame 0 and build
    frame = frames[frame_idx]
    fi1 = min(frame_idx + 1, len(frames) - 1)
    sim.set_init(frame, frames[fi1], fps=float(d['fps']))
    # note: dof_vel from motion at t=0 may be nonzero; sim set_init zeroes
    # velocities, so compare the pose-dependent segments and report vel
    # segments separately.
    obs_sim = sim.build_obs()

    print("train obs", obs_train.shape, "| sim obs", obs_sim.shape)
    assert obs_train.shape == obs_sim.shape

    segs = {
        "root_h(1)": (0, 1), "root_rot_tan_norm(6)": (1, 7),
        "root_vel(3)": (7, 10), "root_ang_vel(3)": (10, 13),
        "joint_tn(174)": (13, 187), "dof_vel(29)": (187, 216),
        "key_pos(12)": (216, 228),
    }
    pose_worst = vel_worst = 0.0
    for name, (a, b) in segs.items():
        diff = np.abs(obs_train[a:b] - obs_sim[a:b]).max()
        tag = "vel" if "vel" in name else "pose"
        if tag == "pose":
            pose_worst = max(pose_worst, diff)
        else:
            vel_worst = max(vel_worst, diff)
        print(f"{name:22s} max|diff| = {diff:.6f}")
    print(f"POSE WORST: {pose_worst:.6f}  VEL WORST: {vel_worst:.6f}")
    ok = pose_worst < 0.05
    print("PARITY:", "PASS" if ok else "FAIL",
          "(vel segs expected to differ: sim starts at rest)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
