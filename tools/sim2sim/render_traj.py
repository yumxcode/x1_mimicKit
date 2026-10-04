"""Render X1 trajectories with the ORIGINAL X1 mesh model (X1_29DOF MJCF).

Serves both:
  - MuJoCo sim2sim rollouts (tools/sim2sim/sim2sim_x1.py --video)
  - IsaacLab training-side rollouts dumped as trajectory pkls
    (scripts_remote/dump_policy_traj.py), rendered locally

Input trajectory: dict with
  root_pos   (N,3)   world position of base_link
  root_rot   (N,4)   wxyz quaternion
  dof        (N,29)  joint angles in X1_DOF_ORDER (tools/x1_pipeline
                     X1_DOF_ORDER: lumbar, L-arm, R-arm, L-leg, R-leg)

Output: mp4 (camera tracking the robot).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MESH_XML = REPO / "X1_29DOF/mjcf/xyber_x1_flat.xml"


class MeshRenderer:
    def __init__(self, width=960, height=540):
        import mujoco
        self.m = mujoco.MjModel.from_xml_path(str(MESH_XML))
        # enlarge offscreen framebuffer for high-res rendering
        self.m.vis.global_.offwidth = max(width, 640)
        self.m.vis.global_.offheight = max(height, 480)
        self.d = mujoco.MjData(self.m)
        self.cam = mujoco.MjvCamera()
        self.renderer = mujoco.Renderer(self.m, height=height, width=width)

        from tools.sim2sim.sim2sim_x1 import parse_x1_xml
        spec = parse_x1_xml()
        # map X1_DOF_ORDER joint name -> mesh model qpos address
        self.qadr = np.array([self.m.joint(n).qposadr[0]
                              for n in spec["names"]])
        self.base_bid = self.m.body("base_link").id

    def set_state(self, root_pos, root_rot_wxyz, dof):
        import mujoco
        self.d.qpos[:] = 0
        self.d.qpos[0:3] = root_pos
        self.d.qpos[3:7] = root_rot_wxyz / np.linalg.norm(root_rot_wxyz)
        self.d.qpos[self.qadr] = dof
        mujoco.mj_forward(self.m, self.d)

    def render_frame(self, root_pos, root_rot_wxyz, dof, cam_back=3.2,
                     cam_height=1.1):
        self.set_state(root_pos, root_rot_wxyz, dof)
        self.cam.lookat[:] = [root_pos[0] + 0.8, root_pos[1],
                              cam_height * 0.7]
        self.cam.distance = cam_back
        self.cam.azimuth = 90.0
        self.cam.elevation = -12.0
        self.renderer.update_scene(self.d, camera=self.cam)
        return self.renderer.render()


def render_trajectory(traj, out_mp4, fps=30, width=960, height=540):
    import imageio.v2 as imageio
    mr = MeshRenderer(width=width, height=height)
    frames = []
    root_pos = np.asarray(traj["root_pos"])
    root_rot = np.asarray(traj["root_rot"])
    dof = np.asarray(traj["dof"])
    n = len(root_pos)
    for i in range(n):
        frames.append(mr.render_frame(root_pos[i], root_rot[i], dof[i]))
        if (i + 1) % 60 == 0:
            print(f"  rendered {i+1}/{n}", flush=True)
    imageio.mimwrite(out_mp4, frames, fps=fps, quality=8)
    print(f"[render] wrote {out_mp4} ({n} frames)")
    return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True,
                    help="trajectory pkl/npz with root_pos/root_rot/dof")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import pickle
    if args.traj.endswith(".npz"):
        z = np.load(args.traj)
        traj = {k: z[k] for k in ("root_pos", "root_rot", "dof")}
    else:
        with open(args.traj, "rb") as f:
            traj = pickle.load(f)
    render_trajectory(traj, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
