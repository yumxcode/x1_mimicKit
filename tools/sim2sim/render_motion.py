"""Render a retargeted motion pkl with the X1 mesh model (acceptance video).

Usage: .venv/bin/python tools/sim2sim/render_motion.py \
    --motion data/motions/x1_v3/x1_run1_subject5_seg0.pkl \
    --out output/videos/ref_motion.mp4 [--seconds 6]
"""
import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    with open(args.motion, "rb") as f:
        d = pickle.load(f)
    fps = float(d["fps"])
    frames = np.array(d["frames"])[: int(args.seconds * fps)]

    rots = []
    for e in frames[:, 3:6]:
        ang = float(np.linalg.norm(e))
        if ang < 1e-8:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = e / ang
            q = np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]
        rots.append(q)

    traj = dict(root_pos=frames[:, 0:3],
                root_rot=np.array(rots),
                dof=frames[:, 6:35])

    from tools.sim2sim.render_traj import render_trajectory
    render_trajectory(traj, args.out, fps=int(fps))


if __name__ == "__main__":
    main()
