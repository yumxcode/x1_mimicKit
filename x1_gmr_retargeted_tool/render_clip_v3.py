"""Render a G1 source clip and its X1 retargeted pkl side by side to MP4.

v3: the X1 side uses the ORIGINAL X1_29DOF mjcf (MESH visual, matching the
URDF-mesh renders the user validates by eye), posed by joint name from the
pkl. v2 rendered primitive geoms, which hid mesh-level sole penetration.

Usage:
  python tools/x1_pipeline/render_clip_v3.py --csv <g1.csv> --pkl <x1.pkl> \
      --out output/renders/<name>.mp4 [--every 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco as _mj_cast  # noqa: F401 (ensure mujoco imported before Renderer)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from lib_g1 import load_csv, g1_fk
from validate_retarget import X1Player
from retarget_v3 import X1_DOF_ORDER

G1_MENAGERIE = REPO_ROOT / ".research/g1_menagerie.xml"
X1_SRC_MJCF = REPO_ROOT / "X1_29DOF/mjcf/xyber_x1_flat.xml"


def free_cam(look, travel_dir, dist, height):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(look, dtype=float)
    cam.lookat[2] += 0.1
    cam.distance = dist
    cam.azimuth = np.degrees(np.arctan2(travel_dir[0], travel_dir[1])) + 180.0
    cam.elevation = -np.degrees(np.arctan2(height, dist))
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    import mujoco
    import imageio
    from scipy.spatial.transform import Rotation as Rot

    OPT = mujoco.MjvOption()
    for g in range(6):
        OPT.geomgroup[g] = 1

    G1_ASSETS = REPO_ROOT / ".research/g1_assets_repo/unitree_g1/assets"
    xml = G1_MENAGERIE.read_text().replace(
        'meshdir="assets"', f'meshdir="{G1_ASSETS}"')
    mg = mujoco.MjModel.from_xml_string(xml)
    frames_csv = load_csv(args.csv)

    pl = X1Player(args.pkl)
    # mesh model for the X1 side (same 29 joints by name)
    mx = mujoco.MjModel.from_xml_path(str(X1_SRC_MJCF))
    # FIX: extend the visual floor — retarget pkls live at arbitrary world
    # coords beyond the mjcf floor's |y|<=3 m coverage (robots would be
    # rendered hovering over a void)
    for g in range(mx.ngeom):
        if (mx.geom(g).name or '') == 'floor':
            mx.geom_size[g][1] = 200.0
    dx = mujoco.MjData(mx)
    xadr = np.array([mx.joint(j).qposadr[0] for j in X1_DOF_ORDER])

    rg = mujoco.Renderer(mg, height=480, width=480)
    rx = mujoco.Renderer(mx, height=480, width=480)

    n = len(pl.frames)
    src_idx = np.clip(np.round(np.arange(n) / pl.time_scale).astype(int),
                      0, len(frames_csv["pos"]) - 1)
    path = np.array(pl.frames)[:, :2]
    v = np.gradient(path, axis=0)
    v = np.convolve(v[:, 0], np.ones(15) / 15, mode="same"), \
        np.convolve(v[:, 1], np.ones(15) / 15, mode="same")
    v = np.stack(v, axis=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=8)

    for i in range(0, n, args.every):
        d = v[min(i, n - 1)]
        if np.linalg.norm(d) < 1e-3:
            d = v[max(0, i - 1)] if i > 0 else np.array([1.0, 0.0])
        if np.linalg.norm(d) < 1e-3:
            d = np.array([1.0, 0.0])
        d = d / np.linalg.norm(d)
        cam_dist, cam_h = 2.6, 0.9

        fr = pl.frames[i]
        dx.qpos[:] = 0
        dx.qpos[:3] = fr[:3]
        dx.qpos[3:7] = Rot.from_rotvec(fr[3:6]).as_quat()[[3, 0, 1, 2]]
        dx.qpos[xadr] = fr[6:]
        mujoco.mj_forward(mx, dx)
        look = dx.qpos[:3].copy()
        rx.update_scene(dx, camera=free_cam(look, d, cam_dist, cam_h),
                        scene_option=OPT)
        pix_x = rx.render()
        imx = np.asarray(pix_x)[..., :3].copy()

        k = src_idx[i]
        gd = g1_fk(mg, frames_csv["pos"][k],
                   frames_csv["quat_xyzw"][k], frames_csv["dof"][k])
        lookg = frames_csv["pos"][k].copy()
        rg.update_scene(gd, camera=free_cam(lookg, d, cam_dist, cam_h),
                        scene_option=OPT)
        pix_g = rg.render()
        img = np.asarray(pix_g)[..., :3].copy()

        writer.append_data(np.concatenate([img, imx], axis=1))

    writer.close()
    print(f"saved {args.out} ({n} frames, {n / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
