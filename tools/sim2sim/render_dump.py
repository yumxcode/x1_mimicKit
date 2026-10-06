"""Render a gym-engine dump (xyzw quats) with the X1 mesh model."""
import pickle
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'tools/sim2sim')

from tools.sim2sim.render_traj import render_trajectory  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d3_traj_ep0.pt"
out = sys.argv[2] if len(sys.argv) > 2 else "output/videos/isaac_dump_ep0_mesh.mp4"

d = pickle.load(open(path, "rb"))
traj = {
    "root_pos": np.asarray(d["root_pos"]),
    "root_rot": np.roll(np.asarray(d["root_rot"]), 1, axis=1),  # xyzw->wxyz
    "dof": np.asarray(d["dof"]),
}
render_trajectory(traj, out, fps=30)
print("rendered", out, len(traj["dof"]), "frames")
