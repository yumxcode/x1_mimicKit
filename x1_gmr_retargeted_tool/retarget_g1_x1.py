"""G1 -> X1 motion retargeting with physical-limit-aware speed adaptation.

Pipeline per clip:
  1. Load G1 CSV, run FK on the meshless G1 model.
  2. Compute kinematic scale (leg ratio) and G1 speed profile.
  3. Time-warp so that the scaled peak speed respects the X1 speed cap
     (X1 physical limit policy, see --speed_cap and analysis output).
  4. Per-frame IK on the X1 model: track feet/hands positions + torso
     orientation, warm-started from the previous frame, joint limits as
     bounds, posture/smoothness regularization.
  5. Ground-snap the root (remove residual foot penetration).
  6. Save MimicKit Motion pkl: frames [root_pos(3), root_expmap(3), dof(29)]
     in X1 depth-first joint order, at --out_fps.

Run:
  python tools/x1_pipeline/retarget_g1_x1.py --csv data/LAFAN1_g1/g1/run1_subject2.csv \
      --out data/motions/x1/x1_run1_subject2.pkl [--speed_cap 2.0]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from lib_g1 import (G1_DOF_ORDER, build_fk_model, load_csv, g1_fk,
                    site_pos, site_quat, quat_xyzw_to_wxyz)

X1_XML = REPO_ROOT / "data/assets/x1/x1_sim.xml"

# X1 depth-first joint order (must match MJCFCharModel parse order)
X1_DOF_ORDER = (
    [f"lumbar_{n}_joint" for n in ("yaw", "roll", "pitch")] +
    [f"left_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll",
                                 "shoulder_yaw", "elbow_pitch", "elbow_yaw",
                                 "wrist_pitch", "wrist_roll")] +
    [f"right_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll",
                                  "shoulder_yaw", "elbow_pitch", "elbow_yaw",
                                  "wrist_pitch", "wrist_roll")] +
    [f"left_{n}_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw",
                                 "knee_pitch", "ankle_pitch", "ankle_roll")] +
    [f"right_{n}_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw",
                                  "knee_pitch", "ankle_pitch", "ankle_roll")]
)

# G1 dof index -> X1 dof index correspondence (by semantic role)
G1_TO_X1_DOF = [
    ("left_hip_pitch_joint", "left_hip_pitch_joint"),
    ("left_hip_roll_joint", "left_hip_roll_joint"),
    ("left_hip_yaw_joint", "left_hip_yaw_joint"),
    ("left_knee_joint", "left_knee_pitch_joint"),
    ("left_ankle_pitch_joint", "left_ankle_pitch_joint"),
    ("left_ankle_roll_joint", "left_ankle_roll_joint"),
    ("right_hip_pitch_joint", "right_hip_pitch_joint"),
    ("right_hip_roll_joint", "right_hip_roll_joint"),
    ("right_hip_yaw_joint", "right_hip_yaw_joint"),
    ("right_knee_joint", "right_knee_pitch_joint"),
    ("right_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ("right_ankle_roll_joint", "right_ankle_roll_joint"),
    ("waist_yaw_joint", "lumbar_yaw_joint"),
    ("waist_roll_joint", "lumbar_roll_joint"),
    ("waist_pitch_joint", "lumbar_pitch_joint"),
    ("left_shoulder_pitch_joint", "left_shoulder_pitch_joint"),
    ("left_shoulder_roll_joint", "left_shoulder_roll_joint"),
    ("left_shoulder_yaw_joint", "left_shoulder_yaw_joint"),
    ("left_elbow_joint", "left_elbow_pitch_joint"),
    ("left_wrist_roll_joint", "left_wrist_roll_joint"),
    ("left_wrist_pitch_joint", "left_wrist_pitch_joint"),
    ("left_wrist_yaw_joint", "left_elbow_yaw_joint"),   # closest analog
    ("right_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
    ("right_shoulder_roll_joint", "right_shoulder_roll_joint"),
    ("right_shoulder_yaw_joint", "right_shoulder_yaw_joint"),
    ("right_elbow_joint", "right_elbow_pitch_joint"),
    ("right_wrist_roll_joint", "right_wrist_roll_joint"),
    ("right_wrist_pitch_joint", "right_wrist_pitch_joint"),
    ("right_wrist_yaw_joint", "right_elbow_yaw_joint"),  # closest analog
]



def site_world_quat(d, sid):
    """Site world quaternion (wxyz) from site_xmat (mujoco 3.1.x compat)."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(d.site_xmat[sid].reshape(3, 3)).as_quat()
    return np.r_[q[3], q[0:3]]



def parse_vel(jname):
    from build_x1_assets import parse_urdf_limits
    return parse_urdf_limits()[jname]["velocity"]
CV_W = 0.8


def quat_mul(q1, q2):
    """Hamilton product, wxyz."""
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    return np.r_[w1 * w2 - v1 @ v2, w1 * v2 + w2 * v1 + np.cross(v1, v2)]


def quat_conj(q):
    return np.r_[q[0], -q[1:]]


def quat_log_diff(q_a, q_b):
    """Rotation vector taking quat a to quat b (wxyz inputs)."""
    import mujoco
    # q_diff = q_a^-1 * q_b
    qd = quat_mul(quat_conj(q_a), q_b)
    if qd[0] < 0:
        qd = -qd
    ang = 2 * np.arccos(np.clip(qd[0], -1, 1))
    if ang < 1e-8:
        return np.zeros(3)
    return qd[1:] / np.linalg.norm(qd[1:]) * ang


def expmap_from_quat_wxyz(q):
    v = quat_log_diff(np.array([1.0, 0, 0, 0]), q)
    return v


# ------------------------------------------------------------------ models
class Retargeter:
    def __init__(self):
        import mujoco
        self.g1 = build_fk_model()
        self.x1 = mujoco.MjModel.from_xml_path(str(X1_XML))
        self.dx1 = mujoco.MjData(self.x1)
        self.x1_qadr = np.array(
            [self.x1.joint(j).qposadr[0] for j in X1_DOF_ORDER])
        self.x1_jids = [self.x1.joint(j).id for j in X1_DOF_ORDER]
        self.x1_lower = np.array([self.x1.jnt_range[self.x1.joint(j).id][0]
                                  for j in X1_DOF_ORDER])
        self.x1_upper = np.array([self.x1.jnt_range[self.x1.joint(j).id][1]
                                  for j in X1_DOF_ORDER])
        # body offsets
        self.g1_hip_local = np.array([0.0, 0.0, -0.0])  # pelvis origin ~ hip level
        self.x1_hip_local = np.array([0.00245, 0.0, -0.0121])
        self._compute_scales()
        self._compute_neutral_ori()

    def _neutral(self, model, dof_order):
        import mujoco
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        return d

    def _compute_scales(self):
        g1d = self._neutral(self.g1, G1_DOF_ORDER)
        x1d = self._neutral(self.x1, X1_DOF_ORDER)
        g1_leg = (site_pos(g1d, "pelvis")[2]
                  - min(site_pos(g1d, "left_ankle_roll_link")[2],
                        site_pos(g1d, "right_ankle_roll_link")[2]))
        x1_leg = (site_pos(x1d, "x_base")[2]
                  - min(site_pos(x1d, "x_lfoot")[2],
                        site_pos(x1d, "x_rfoot")[2]))
        g1_arm = np.linalg.norm(site_pos(g1d, "left_wrist_yaw_link")
                                - site_pos(g1d, "torso_link"))
        x1_arm = np.linalg.norm(site_pos(x1d, "x_lhand")
                                - site_pos(x1d, "x_torso"))
        self.s_leg = x1_leg / g1_leg
        self.s_arm = x1_arm / g1_arm
        print(f"[retarget] scales: leg {self.s_leg:.4f} (X1 {x1_leg:.3f} m / "
              f"G1 {g1_leg:.3f} m), arm {self.s_arm:.4f} "
              f"(X1 {x1_arm:.3f} / G1 {g1_arm:.3f})")

    def _compute_neutral_ori(self):
        """Neutral (dof=0, root identity) world quats of key links, used to
        map orientations across skeletons via delta rotations."""
        g1d = self._neutral(self.g1, G1_DOF_ORDER)
        x1d = self._neutral(self.x1, X1_DOF_ORDER)
        self.g1_neutral = {
            "torso": site_quat(g1d, "torso_link").copy(),
            "lfoot": site_quat(g1d, "left_ankle_roll_link").copy(),
            "rfoot": site_quat(g1d, "right_ankle_roll_link").copy(),
        }
        self.x1_neutral = {
            "torso": site_quat(x1d, "x_torso").copy(),
            "lfoot": site_quat(x1d, "x_lfoot").copy(),
            "rfoot": site_quat(x1d, "x_rfoot").copy(),
        }

    # ------------------------------------------------------------- targets
    def frame_targets(self, gpos, gquat_xyzw, gdof):
        """World-space targets for one G1 frame."""
        gd = g1_fk(self.g1, gpos, gquat_xyzw, gdof)
        t = dict(
            root_pos=gpos.copy(),
            root_quat_wxyz=quat_xyzw_to_wxyz(gquat_xyzw),
            torso_quat_wxyz=site_quat(gd, "torso_link").copy(),
            lfoot=site_pos(gd, "left_ankle_roll_link"),
            rfoot=site_pos(gd, "right_ankle_roll_link"),
            lfoot_quat=site_quat(gd, "left_ankle_roll_link").copy(),
            rfoot_quat=site_quat(gd, "right_ankle_roll_link").copy(),
            lhand=site_pos(gd, "left_wrist_yaw_link"),
            rhand=site_pos(gd, "right_wrist_yaw_link"),
            lhand_quat=site_quat(gd, "left_wrist_yaw_link").copy(),
            rhand_quat=site_quat(gd, "right_wrist_yaw_link").copy(),
            hip_pos=gpos.copy(),  # approx: pelvis ~= hip center for G1
        )
        return t

    def map_root(self, t):
        """Map G1 root pose to X1 root pose (position scaled, quat kept)."""
        s = self.s_leg
        hip_g1 = t["root_pos"]  # G1 pelvis ~ hip center
        hip_x1_target = hip_g1 * np.array([s, s, s])
        hip_x1_target[2] = hip_g1[2] * s  # keep vertical scale by leg ratio
        q = t["root_quat_wxyz"]
        # hip offset in X1 base frame, rotated to world
        R = quat_to_R(q)
        hip_world_offset = R @ self.x1_hip_local
        base_pos = hip_x1_target - hip_world_offset
        return base_pos, q

    # ------------------------------------------------------------------- IK
    def solve_frame(self, t, q_prev, q_ref=None, warmup=False, q_prev2=None):
        import mujoco
        from scipy.optimize import least_squares
        m, d = self.x1, self.dx1
        base_pos, root_quat = self.map_root(t)

        s_leg, s_arm = self.s_leg, self.s_arm
        # All targets share ONE anchor policy: offsets from the G1 root are
        # scaled and re-attached to the mapped X1 base. (Inconsistent
        # anchoring - scaling the root about world origin but limbs about
        # the root - produced ~1 m offsets.)
        base_anchor = base_pos

        foot_targets = np.stack([
            base_anchor + (t["lfoot"] - t["root_pos"]) * s_leg,
            base_anchor + (t["rfoot"] - t["root_pos"]) * s_leg,
        ])
        hand_targets = np.stack([
            base_anchor + (t["lhand"] - t["root_pos"]) * s_arm,
            base_anchor + (t["rhand"] - t["root_pos"]) * s_arm,
        ])

        # never demand foot targets below the floor: ankle site sits 0.055 m
        # above the sole when flat, so clamp target z at 0.058
        foot_targets[:, 2] = np.maximum(foot_targets[:, 2], 0.058)

        # predicted leg-cross: when the swing foot target passes within
        # 0.12 m lateral of the other foot target, soften its position
        # weight so clearance can win without dragging the whole pose
        lat_gap = abs(foot_targets[0, 1] - foot_targets[1, 1])
        foot_w = 30.0 if lat_gap > 0.12 else 12.0

        # Orientation targets map via delta-from-neutral rotations:
        # delta expressed in each skeleton's LOCAL frame axes:
        #   dR = R_g(neutral)^T R_g(f) ;  R_x_target = R_x(neutral) dR
        def delta_target(g_quat, key):
            dR_local = quat_mul(quat_conj(self.g1_neutral[key]),
                                np.array(g_quat))
            return quat_mul(self.x1_neutral[key], dR_local)

        foot_quat_targets = [
            delta_target(t["lfoot_quat"], "lfoot"),
            delta_target(t["rfoot_quat"], "rfoot"),
        ]
        torso_quat_target = delta_target(t["torso_quat_wxyz"], "torso")

        foot_ids = [m.site("x_lfoot").id, m.site("x_rfoot").id]
        hand_ids = [m.site("x_lhand").id, m.site("x_rhand").id]
        torso_id = m.site("x_torso").id

        # clearance pairs (self-penetration guards): torso vs forearms,
        # pelvis vs torso
        clear_pairs = []
        _names = ("lumbar_pitch_link_col", "base_link_col",
                  "left_elbow_yaw_link_col", "right_elbow_yaw_link_col",
                  "left_shoulder_yaw_link_col", "right_shoulder_yaw_link_col",
                  "left_ankle_roll_link_sole", "right_ankle_roll_link_sole",
                  "left_hip_yaw_link_col", "right_hip_yaw_link_col")
        _gid = {}
        for nm in _names:
            try:
                _gid[nm] = m.geom(nm).id
            except KeyError:
                pass
        if "lumbar_pitch_link_col" in _gid:
            for arm in ("left_elbow_yaw_link_col", "right_elbow_yaw_link_col",
                        "left_shoulder_yaw_link_col", "right_shoulder_yaw_link_col"):
                if arm in _gid:
                    clear_pairs.append((_gid["lumbar_pitch_link_col"], _gid[arm]))
        if "base_link_col" in _gid and "lumbar_pitch_link_col" in _gid:
            clear_pairs.append((_gid["base_link_col"], _gid["lumbar_pitch_link_col"]))
        sole_pair = None
        if "left_ankle_roll_link_sole" in _gid and "right_ankle_roll_link_sole" in _gid:
            sole_pair = (_gid["left_ankle_roll_link_sole"],
                         _gid["right_ankle_roll_link_sole"])
            clear_pairs.append(sole_pair)
        for el, hip in (("left_elbow_yaw_link_col", "left_hip_yaw_link_col"),
                        ("right_elbow_yaw_link_col", "right_hip_yaw_link_col"),
                        ("left_elbow_yaw_link_col", "right_hip_yaw_link_col"),
                        ("right_elbow_yaw_link_col", "left_hip_yaw_link_col")):
            if el in _gid and hip in _gid:
                clear_pairs.append((_gid[el], _gid[hip]))
        cross_leg_pairs = []
        for so, sh in (("left_ankle_roll_link_sole", "right_knee_pitch_link_col"),
                       ("right_ankle_roll_link_sole", "left_knee_pitch_link_col"),
                       ("left_ankle_roll_link_sole", "right_hip_yaw_link_col"),
                       ("right_ankle_roll_link_sole", "left_hip_yaw_link_col")):
            if so in _gid and sh in _gid:
                p = (_gid[so], _gid[sh])
                cross_leg_pairs.append(p)
                clear_pairs.append(p)
        if "base_link_col" in _gid and "lumbar_pitch_link_col" in _gid:
            cross_leg_pairs.append((_gid["base_link_col"], _gid["lumbar_pitch_link_col"]))

        nv = m.nv
        dof_sel = slice(6, nv)  # skip free joint dofs

        def set_state(q):
            d.qpos[:] = 0
            d.qpos[:3] = base_pos
            d.qpos[3:7] = root_quat
            d.qpos[self.x1_qadr] = q
            mujoco.mj_forward(m, d)

        def residuals(q):
            set_state(q)
            res = []
            # feet positions (2 x 3)
            for i, sid in enumerate(foot_ids):
                res.append(foot_w * (d.site_xpos[sid] - foot_targets[i]))
                # foot pitch/roll alignment to G1 foot (projected on X1 ankle
                # 2-dof achievable subspace is hard; track full ori lightly)
                rvec = quat_log_diff(site_world_quat(d, sid),
                                     foot_quat_targets[i])
                res.append(0.5 * rvec)
            # hands positions
            for i, sid in enumerate(hand_ids):
                res.append(8.0 * (d.site_xpos[sid] - hand_targets[i]))
            # torso orientation (X1 lumbar ROM is far smaller than G1 waist;
            # best-effort tracking)
            rvec = quat_log_diff(site_world_quat(d, torso_id),
                                 torso_quat_target)
            res.append(3.0 * rvec)
            # posture + smoothness regularization
            if q_ref is not None:
                res.append(0.02 * (q - q_ref))
            if q_prev is not None:
                # split regularization: redundant joints (hip yaw/roll,
                # lumbar) hop between IK branches -> strong POSITION prior;
                # tracking-critical joints get a constant-velocity prior
                # (penalizes acceleration, no lag on steady swings)
                if q_prev2 is not None:
                    q_pred = 2.0 * q_prev - q_prev2
                else:
                    q_pred = q_prev
                REDUNDANT = ("hip_yaw", "hip_roll", "lumbar")
                wvec = np.array([
                    3.0 if any(r in j for r in REDUNDANT) else CV_W
                    for j in X1_DOF_ORDER])
                res.append(wvec * (q - q_pred))
            # joint limit margin (soft, handled by bounds too)
            margin = 0.05
            over = np.maximum(0, q - (self.x1_upper - margin)) \
                 + np.minimum(0, q - (self.x1_lower + margin))
            res.append(10.0 * over)
            # self-penetration clearance (no-penetration strict criterion)
            hard_pairs = {sole_pair} | set(cross_leg_pairs)
            for ga, gb in clear_pairs:
                dist = mujoco.mj_geomDistance(m, d, ga, gb, 0.5, None)
                w = 200.0 if (ga, gb) in hard_pairs else 60.0
                res.append(np.array([w * max(0.0, 0.008 - dist)]))
            return np.concatenate(res)

        if q_prev is None:
            # cold start from a standing basin (home pose), NOT mid-range:
            # high-effort first solves can dive into crossed-feet local minima
            q0 = np.zeros(len(self.x1_lower))
            home = {"left_hip_pitch_joint": 0.48, "left_hip_roll_joint": 0.06,
                    "left_hip_yaw_joint": -0.33, "left_knee_pitch_joint": 0.6,
                    "left_ankle_pitch_joint": -0.27,
                    "right_hip_pitch_joint": -0.48, "right_hip_roll_joint": -0.06,
                    "right_hip_yaw_joint": 0.33, "right_knee_pitch_joint": 0.6,
                    "right_ankle_pitch_joint": -0.27}
            for jn, v in home.items():
                k = X1_DOF_ORDER.index(jn)
                q0[k] = v
            q0 = np.clip(q0, self.x1_lower + 0.05, self.x1_upper - 0.05)
        else:
            q0 = q_prev.copy()
        nfev = 400 if q_prev is None else 100
        try:
            sol = least_squares(residuals, q0, jac="2-point",
                                bounds=(self.x1_lower + 1e-4,
                                        self.x1_upper - 1e-4),
                                max_nfev=nfev, verbose=0)
            q = np.clip(sol.x, self.x1_lower, self.x1_upper)
            cost = sol.cost
        except (np.linalg.LinAlgError, ValueError):
            q = np.clip(q0, self.x1_lower, self.x1_upper)  # keep warm start
            cost = float("nan")
        set_state(q)
        return q, base_pos, root_quat, cost


def quat_to_R(q):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()


def exp_map_from_quat_wxyz(q):
    """wxyz quat -> 3D exp map (MimicKit root_rot format)."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(np.r_[q[1:], q[0]]).as_rotvec()


# --------------------------------------------------------------- speed tool
def speed_profile(pos, dt, win=9):
    v = np.gradient(pos, dt, axis=0)
    v_horiz = np.linalg.norm(v[:, :2], axis=1)
    if win > 1:
        k = np.ones(win) / win
        v_horiz = np.convolve(v_horiz, k, mode="same")
    return v_horiz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed_cap", type=float, default=2.0,
                    help="X1 max sustained run speed (m/s)")
    ap.add_argument("--peak_ratio", type=float, default=1.15,
                    help="allowed peak/cap ratio")
    ap.add_argument("--src_fps", type=int, default=30)
    ap.add_argument("--out_fps", type=int, default=30)
    ap.add_argument("--loop", default="clamp", choices=["wrap", "clamp"])
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--no_slowdown", action="store_true")
    ap.add_argument("--cv_weight", type=float, default=0.8)
    args = ap.parse_args()

    global CV_W
    CV_W = args.cv_weight
    frames = load_csv(args.csv)
    e = len(frames["pos"]) if args.end_frame < 0 else args.end_frame
    pos = frames["pos"][args.start_frame:e]
    quat = frames["quat_xyzw"][args.start_frame:e]
    dof = frames["dof"][args.start_frame:e]
    n = len(pos)
    dt = 1.0 / args.src_fps
    print(f"[retarget] {args.csv}: {n} frames @ {args.src_fps} fps "
          f"({n / args.src_fps:.1f}s)")

    rt = Retargeter()

    # ---- speed policy
    v_g1 = speed_profile(pos, dt)
    print(f"[retarget] G1 speed: mean {v_g1.mean():.2f} p50 {np.quantile(v_g1, 0.5):.2f}"
          f" p90 {np.quantile(v_g1, 0.9):.2f} p99 {np.quantile(v_g1, 0.99):.2f} max {v_g1.max():.2f} m/s")
    v_x1_raw = v_g1 * rt.s_leg
    v_cap = args.speed_cap
    if args.no_slowdown:
        warp = 1.0
    else:
        # target: p99 speed <= cap (allow peak_ratio on instantaneous max)
        warp = max(1.0, np.quantile(v_x1_raw, 0.99) / v_cap)
        peak = v_x1_raw.max() / warp
        if peak > v_cap * args.peak_ratio:
            warp = v_x1_raw.max() / (v_cap * args.peak_ratio)
    vel_lim = np.array([parse_vel(j) for j in X1_DOF_ORDER])

    def run_pass(warp):
        """resample + IK + filter + ground lift for one warp factor."""
        # Slow-motion semantics: output plays the source clip over T*warp
        # seconds. Output frame j shows source time j/(out_fps*warp).
        n_out = max(int(round(n * warp * args.out_fps / args.src_fps)), 2)
        t_out = np.arange(n_out) / (args.out_fps * warp)
        t_out = np.minimum(t_out, (n - 1) * dt)  # stay inside the clip

        idx_f = t_out * args.src_fps
        i0 = np.clip(np.floor(idx_f).astype(int), 0, n - 2)
        alpha = (idx_f - i0)[..., None]
        pos_s = pos[i0] * (1 - alpha) + pos[i0 + 1] * alpha
        quat_s = quat[i0] * (1 - alpha) + quat[i0 + 1] * alpha
        quat_s /= np.linalg.norm(quat_s, axis=1, keepdims=True)
        dof_s = dof[i0] * (1 - alpha) + dof[i0 + 1] * alpha

        out_frames = []
        q_prev = None
        cost_acc = []
        q_prev2 = None
        for i in range(n_out):
            t = rt.frame_targets(pos_s[i], quat_s[i], dof_s[i])
            q, base_pos, root_quat, cost = rt.solve_frame(
                t, q_prev, q_ref=None, warmup=(i < 30), q_prev2=q_prev2)
            cost_acc.append(cost)
            q_prev2 = q_prev
            q_prev = q
            out_frames.append((base_pos, root_quat, q))

        base_p = np.stack([f[0] for f in out_frames])
        root_q = np.stack([f[1] for f in out_frames])
        dof_out = np.stack([f[2] for f in out_frames])
        return n_out, base_p, root_q, dof_out, float(np.nanmean(cost_acc))

    # gait-critical joints (warp-driven); lumbar is stylistic counter-
    # rotation and its IK tends to bang-bang between +-1.0 limits -> treat
    # with strong smoothing + limiter instead of global slow-down
    CRIT = {"hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"}

    def is_critical(j):
        return any(c in j for c in CRIT)

    def filter_and_lift(n_out, base_p, root_q, dof_out):
        """zero-phase 8 Hz lowpass (gait-preserving) + hard joint-velocity
        limiter for spike-driven violations, THEN the ground lift."""
        from scipy.signal import butter, filtfilt
        if n_out >= 9:
            # per-joint zero-phase cutoffs (gait swing is 2-3 Hz; these
            # preserve it while removing broadband IK jitter)
            cuts = np.full(len(X1_DOF_ORDER), 8.0)
            for k, j in enumerate(X1_DOF_ORDER):
                if "wrist" in j:
                    cuts[k] = 2.5
            for k in range(len(X1_DOF_ORDER)):
                fc = min(cuts[k], 0.9 * (args.out_fps / 2.0))
                b, a = butter(4, fc / (args.out_fps / 2.0))
                dof_out[:, k] = filtfilt(b, a, dof_out[:, k])

        # hard limiter ONLY for non-critical joints (spike suppression);
        # critical gait joints are handled by global time-warp instead
        for _ in range(6):
            qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
            over = [qd[:, k].max() > 0.95 * vel_lim[k]
                    and "wrist" in X1_DOF_ORDER[k]
                    for k in range(len(X1_DOF_ORDER))]
            if not any(over):
                break
            for k in np.where(over)[0]:
                fc = 4.0
                while fc > 0.1:
                    b, a = butter(4, fc / (args.out_fps / 2.0))
                    cand = filtfilt(b, a, dof_out[:, k])
                    if np.abs(np.gradient(cand, 1.0 / args.out_fps)).max() <= 0.95 * vel_lim[k]:
                        dof_out[:, k] = cand
                        break
                    fc *= 0.5

        dof_out = np.minimum(np.maximum(dof_out, rt.x1_lower), rt.x1_upper)
        # sustained (p99) velocity ratio on critical joints -> warp driver
        qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
        crit_ratio = max(
            (np.quantile(qd[:, k], 0.99) / vel_lim[k])
            for k in range(len(X1_DOF_ORDER)) if is_critical(X1_DOF_ORDER[k]))

        # ground lift AFTER filtering (final trajectories decide contact)
        import mujoco
        d = mujoco.MjData(rt.x1)
        sole_gids = [g for g in range(rt.x1.ngeom)
                     if (rt.x1.geom(g).name or "").endswith("_sole")]
        minz_i = np.zeros(n_out)
        for i in range(n_out):
            d.qpos[:] = 0
            d.qpos[:3] = base_p[i]
            d.qpos[3:7] = root_q[i]
            d.qpos[rt.x1_qadr] = dof_out[i]
            mujoco.mj_forward(rt.x1, d)
            minz = np.inf
            for g in sole_gids:
                R = d.geom_xmat[g].reshape(3, 3)
                half = rt.x1.geom_size[g]
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            c = (d.geom_xpos[g] + R @ np.array(
                                [sx * half[0], sy * half[1], sz * half[2]]))[2]
                            minz = min(minz, c)
            minz_i[i] = minz
        lift = np.maximum(0.0, -minz_i + 0.002)
        lift = np.minimum(lift, 0.15)
        lift = np.array([lift[max(0, i - 1):i + 2].max() for i in range(n_out)])
        k5 = np.ones(5) / 5
        lift_s = np.convolve(np.r_[lift[:2][::-1], lift, lift[-2:][::-1]],
                             k5, mode="valid")
        lift = np.minimum(np.maximum(lift, lift_s), 0.15)
        base_p[:, 2] += lift
        return base_p, dof_out, crit_ratio

    # ---- warp grid with self-validation: the IK is noise-sensitive, so
    # try a small grid of extra slow-down factors and keep the FIRST
    # variant that passes the full six-criteria validation
    import copy

    def save_result(warp_f, n_out_f, base_f, rootq_f, dof_f):
        k5 = np.ones(5) / 5
        base_f[:, 2] = np.convolve(np.r_[base_f[:2, 2][::-1], base_f[:, 2],
                                         base_f[-2:, 2][::-1]], k5, mode="valid")
        expmaps = np.stack([exp_map_from_quat_wxyz(q) for q in rootq_f])
        frames_out = np.concatenate([base_f, expmaps, dof_f],
                                    axis=1).astype(np.float32)
        out = dict(loop_mode=(1 if args.loop == "wrap" else 0),
                   fps=args.out_fps, frames=frames_out.tolist(),
                   time_scale=float(warp_f), s_leg=float(rt.s_leg),
                   s_arm=float(rt.s_arm))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(out, f)

    best = None
    for extra in (1.0, 1.06, 1.15, 1.3, 1.5, 1.8, 2.2):
        warp_try = warp * extra
        n_out, base_p, root_q, dof_raw, cost_mean = run_pass(warp_try)
        base_p, dof_out, crit_ratio = filter_and_lift(
            n_out, base_p.copy(), root_q, dof_raw.copy())
        qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
        ratio = (qd / vel_lim).max()
        print(f"[retarget] warp x{warp_try:.3f} ({n_out} frames, cost "
              f"{cost_mean:.2f}) max |qdot| {qd.max():.2f} (ratio {ratio:.3f},"
              f" crit p99 {crit_ratio:.3f})")
        save_result(warp_try, n_out, base_p, root_q, dof_out)
        if ratio > 1.05 or crit_ratio > 1.0:
            continue  # physically invalid; try slower
        try:
            from validate_retarget import validate as _val
            res = _val(args.csv, args.out, sample_step=4)
            ok = res.get("PASS", False)
            fails = [k for k, v in res.items()
                     if isinstance(v, dict) and v.get("pass_") is False]
            print(f"[retarget] self-validate: {'PASS' if ok else 'FAIL ' + str(fails)}")
        except Exception as e:
            ok, fails = False, [f"validator error: {e}"]
            print(f"[retarget] self-validate error: {e}")
        if ok:
            best = (warp_try, n_out)
            break
        if best is None:
            best = (warp_try, n_out)
    warp = best[0]
    print(f"[retarget] final time warp x{warp:.3f}: X1 p99 speed "
          f"{np.quantile(v_x1_raw, 0.99) / warp:.2f} m/s, clip "
          f"{best[1] / args.out_fps:.1f}s @ {args.out_fps} fps")
    print(f"[retarget] final time warp x{warp:.3f}: X1 p99 speed "
          f"{np.quantile(v_x1_raw, 0.99) / warp:.2f} m/s, clip "
          f"{n_out / args.out_fps:.1f}s @ {args.out_fps} fps")

    print(f"[retarget] saved {args.out}")


if __name__ == "__main__":
    main()
