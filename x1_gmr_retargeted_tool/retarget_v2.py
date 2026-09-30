"""G1 -> X1 retargeting v2 — analytic joint-space reference + IK refinement.

v1 failed visually: X1's hips are 45-degree diagonal axes with L/R mirroring
(left hip_pitch axis (0,-0.707,0.707), left hip_roll axis (-1,0,0)); pure
position-IK with a G1-convention warm seed fell into crossed-leg local
minima (hip_roll -1.4 rad, knee never flexing) while still tracking foot
positions to 4 mm — metrics passed, video looked broken.

v2 fixes:
  1. Per-frame ANALYTIC joint reference q_ref: for each corresponding
     sub-chain (hip 3-dof, knee 1, ankle 2, torso 3, shoulder 3, forearm 3)
     read G1's link-relative rotation from FK and solve the X1 joints whose
     chain rotation reproduces it (small least-squares on the rotation
     vector, joint limits as bounds). Signs/axes emerge from the fit — no
     hand-coded conventions.
  2. IK keeps position targets (feet w=30/12, hands w=8) but is seeded at
     q_ref and regularized toward it (w=1.0), so the null space follows the
     source pattern instead of drifting into twisted minima.
  3. Root mapping is anchor-based: X1_base = anchor + (G1_pelvis - G1_first)
     * s_leg (horizontal) and standing-offset-based vertical scaling.

Output: MimicKit Motion pkl, frames [root_pos(3), root_expmap(3), dof(29)]
in X1 depth-first order.
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

# Morphological lateral widening of foot targets (per side, m): the X1's
# feet are proportionally wider than the 0.77-scaled G1's, so narrow-run
# foot passages of the source cause sole interpenetration on the X1.
LAT_WIDEN = 0.05

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

# ------------------------------------------------------------- quat helpers
def quat_mul(q1, q2):
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    return np.r_[w1 * w2 - v1 @ v2, w1 * v2 + w2 * v1 + np.cross(v1, v2)]


def quat_conj(q):
    return np.r_[q[0], -q[1:]]


def quat_log_diff(q_a, q_b):
    qd = quat_mul(quat_conj(q_a), q_b)
    if qd[0] < 0:
        qd = -qd
    ang = 2 * np.arccos(np.clip(qd[0], -1, 1))
    if ang < 1e-8:
        return np.zeros(3)
    return qd[1:] / np.linalg.norm(qd[1:]) * ang


def rotvec_to_quat_wxyz(v):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_rotvec(v).as_quat()  # xyzw
    return np.r_[q[3], q[0:3]]


def quat_to_R(q):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()


def expmap_from_quat_wxyz(q):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(np.r_[q[1:], q[0]]).as_rotvec()


def body_world_quat(d, name):
    """Body world quaternion (wxyz)."""
    from scipy.spatial.transform import Rotation as Rot
    bid = d.body(name).id
    q = Rot.from_matrix(d.xmat[bid].reshape(3, 3)).as_quat()
    return np.r_[q[3], q[0:3]]


# ------------------------------------------------------------------- chains
class RetargeterV2:
    def __init__(self):
        import mujoco
        self.g1 = build_fk_model()
        self.x1 = mujoco.MjModel.from_xml_path(str(X1_XML))
        self.dx1 = mujoco.MjData(self.x1)
        self.x1_qadr = np.array(
            [self.x1.joint(j).qposadr[0] for j in X1_DOF_ORDER])
        self.x1_jids = [self.x1.joint(j).id for j in X1_DOF_ORDER]
        self.x1_lower = np.array([self.x1.jnt_range[j][0] for j in self.x1_jids])
        self.x1_upper = np.array([self.x1.jnt_range[j][1] for j in self.x1_jids])
        from semmap import SemanticMapper
        self.mapper = SemanticMapper(self.x1, self.g1)
        self._compute_scales()
        self._compute_neutral_ori()

    def _neutral(self, model):
        import mujoco
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        return d

    def _compute_scales(self):
        g1d = self._neutral(self.g1)
        x1d = self._neutral(self.x1)
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
        # X1 standing pelvis (base) height above sole line
        self.x1_stand_z = (site_pos(x1d, "x_base")[2]
                           - max(site_pos(x1d, "x_lfoot")[2],
                                 site_pos(x1d, "x_rfoot")[2]))
        # G1 standing pelvis height above sole
        self.g1_stand_z = (site_pos(g1d, "pelvis")[2]
                           - max(site_pos(g1d, "left_ankle_roll_link")[2],
                                 site_pos(g1d, "right_ankle_roll_link")[2]))
        print(f"[retarget-v2] s_leg {self.s_leg:.4f} s_arm {self.s_arm:.4f} "
              f"X1 stand {self.x1_stand_z:.3f} G1 stand {self.g1_stand_z:.3f}")

    def _compute_neutral_ori(self):
        g1d = self._neutral(self.g1)
        x1d = self._neutral(self.x1)
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

    # ------------------------------------------------ analytic joint mapping
    def q_ref(self, gd):
        """Semantic-frame analytic mapping of G1 frame gd -> X1 joints."""
        qd = self.mapper.q_ref(gd)
        return np.array([qd.get(j, 0.0) for j in X1_DOF_ORDER])

    # ------------------------------------------------------------- IK frame
    def frame_targets(self, gpos, gquat_xyzw, gdof):
        gd = g1_fk(self.g1, gpos, gquat_xyzw, gdof)
        return dict(
            root_pos=gpos.copy(),
            root_quat_wxyz=quat_xyzw_to_wxyz(gquat_xyzw),
            torso_quat_wxyz=site_quat(gd, "torso_link").copy(),
            lfoot=site_pos(gd, "left_ankle_roll_link"),
            rfoot=site_pos(gd, "right_ankle_roll_link"),
            lfoot_quat=site_quat(gd, "left_ankle_roll_link").copy(),
            rfoot_quat=site_quat(gd, "right_ankle_roll_link").copy(),
            lhand=site_pos(gd, "left_wrist_yaw_link"),
            rhand=site_pos(gd, "right_wrist_yaw_link"),
            fk=gd,
        )

    def map_root(self, t, g1_first, x1_anchor_xy):
        """Anchor-based root mapping (see module docstring point 3)."""
        rel = t["root_pos"] - g1_first
        xy = x1_anchor_xy + rel[:2] * self.s_leg
        z = self.x1_stand_z + (t["root_pos"][2] - self.g1_stand_z) * self.s_leg
        base = np.array([xy[0], xy[1], z])
        return base, t["root_quat_wxyz"]

    def solve_frame(self, t, q_ref, q_prev, base_pos, root_quat):
        """IK refinement: LEGS ONLY (12 dof) tracking foot position+quat.

        Everything above the pelvis (lumbar 3 + arms 14) is PINNED to the
        analytic q_ref. The previous whole-body IK compromised between the
        analytic prior and a neutral-delta torso-quat target that is
        structurally inconsistent (the x_torso site carries a 90-deg URDF
        export rotation; measured disagreement 25-159 deg) — the solver
        swung lumbar_yaw +-1 rad, rotating the whole validated upper body
        (J1 arm gates failed at uarm 27-37 deg). Legs-only IK is
        well-posed: 12 dof vs 12 foot constraints.
        """
        import mujoco
        from scipy.optimize import least_squares
        m, d = self.x1, self.dx1
        s_leg = self.s_leg

        ik_mask = np.array([("hip" in j) or ("knee" in j)
                            or ("ankle" in j) for j in X1_DOF_ORDER])
        ik_idx = np.where(ik_mask)[0]
        q_fixed = q_ref.copy()

        lat_dir = quat_to_R(root_quat) @ np.array([0.0, 1.0, 0.0])
        foot_targets = np.stack([
            base_pos + (t["lfoot"] - t["root_pos"]) * s_leg + LAT_WIDEN * lat_dir,
            base_pos + (t["rfoot"] - t["root_pos"]) * s_leg - LAT_WIDEN * lat_dir,
        ])
        foot_targets[:, 2] = np.maximum(foot_targets[:, 2], 0.058)

        lat_gap = abs(foot_targets[0, 1] - foot_targets[1, 1])
        foot_w = 30.0 if lat_gap > 0.12 else 12.0

        def delta_target(g_quat, key):
            dR = quat_mul(quat_conj(self.g1_neutral[key]), np.array(g_quat))
            return quat_mul(self.x1_neutral[key], dR)

        foot_quat_targets = [
            delta_target(t["lfoot_quat"], "lfoot"),
            delta_target(t["rfoot_quat"], "rfoot"),
        ]

        foot_ids = [m.site("x_lfoot").id, m.site("x_rfoot").id]

        # clearance pairs (legs only)
        _names = ("left_ankle_roll_link_sole", "right_ankle_roll_link_sole",
                  "left_hip_yaw_link_col", "right_hip_yaw_link_col",
                  "left_knee_pitch_link_col", "right_knee_pitch_link_col")
        _gid = {}
        for nm in _names:
            try:
                _gid[nm] = m.geom(nm).id
            except KeyError:
                pass
        clear_pairs = []
        sole_pair = None
        if "left_ankle_roll_link_sole" in _gid and "right_ankle_roll_link_sole" in _gid:
            sole_pair = (_gid["left_ankle_roll_link_sole"],
                         _gid["right_ankle_roll_link_sole"])
            clear_pairs.append(sole_pair)
        cross_leg_pairs = []
        for so, sh in (("left_ankle_roll_link_sole", "right_knee_pitch_link_col"),
                       ("right_ankle_roll_link_sole", "left_knee_pitch_link_col"),
                       ("left_ankle_roll_link_sole", "right_hip_yaw_link_col"),
                       ("right_ankle_roll_link_sole", "left_hip_yaw_link_col")):
            if so in _gid and sh in _gid:
                cross_leg_pairs.append((_gid[so], _gid[sh]))
                clear_pairs.append((_gid[so], _gid[sh]))

        qadr = self.x1_qadr

        def set_state(qv):
            d.qpos[:] = 0
            d.qpos[:3] = base_pos
            d.qpos[3:7] = root_quat
            d.qpos[qadr] = q_fixed
            d.qpos[qadr[ik_idx]] = qv
            mujoco.mj_forward(m, d)

        W_CV = 0.3

        def residuals(qv):
            set_state(qv)
            res = []
            for i, sid in enumerate(foot_ids):
                res.append(foot_w * (d.site_xpos[sid] - foot_targets[i]))
                rvec = quat_log_diff(
                    body_world_quat_from_site(d, sid), foot_quat_targets[i])
                res.append(0.5 * rvec)
            # joint-space prior toward the analytic reference: keeps the
            # redundant DOFs on the source pattern (THE v2 fix)
            res.append(1.0 * (qv - q_ref[ik_idx]))
            # light constant-velocity smoothness
            if q_prev is not None:
                res.append(W_CV * (qv - q_prev[ik_idx]))
            margin = 0.05
            lo_i = self.x1_lower[ik_idx]
            hi_i = self.x1_upper[ik_idx]
            over = np.maximum(0, qv - (hi_i - margin)) \
                 + np.minimum(0, qv - (lo_i + margin))
            res.append(10.0 * over)
            hard_pairs = set([sole_pair]) if sole_pair else set()
            hard_pairs |= set(cross_leg_pairs)
            for ga, gb in clear_pairs:
                dist = mujoco.mj_geomDistance(m, d, ga, gb, 0.5, None)
                w = 200.0 if (ga, gb) in hard_pairs else 60.0
                res.append(np.array([w * max(0.0, 0.008 - dist)]))
            return np.concatenate(res)

        q0 = np.clip(q_ref[ik_idx] if q_prev is None else q_prev[ik_idx],
                     self.x1_lower[ik_idx] + 1e-4,
                     self.x1_upper[ik_idx] - 1e-4)
        nfev = 200 if q_prev is None else 60
        try:
            sol = least_squares(residuals, q0, jac="2-point",
                                bounds=(self.x1_lower[ik_idx] + 1e-4,
                                        self.x1_upper[ik_idx] - 1e-4),
                                max_nfev=nfev, verbose=0)
            qv = np.clip(sol.x, self.x1_lower[ik_idx],
                         self.x1_upper[ik_idx])
            cost = sol.cost
        except (np.linalg.LinAlgError, ValueError):
            qv = q0
            cost = float("nan")
        q = q_fixed.copy()
        q[ik_idx] = qv
        set_state(qv)
        return q, cost


def body_world_quat_from_site(d, sid):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(d.site_xmat[sid].reshape(3, 3)).as_quat()
    return np.r_[q[3], q[0:3]]


# --------------------------------------------------------------- speed tool
def speed_profile(pos, dt, win=9):
    v = np.gradient(pos, dt, axis=0)
    vh = np.linalg.norm(v[:, :2], axis=1)
    if win > 1:
        k = np.ones(win) / win
        vh = np.convolve(vh, k, mode="same")
    return vh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed_cap", type=float, default=2.0)
    ap.add_argument("--peak_ratio", type=float, default=1.15)
    ap.add_argument("--src_fps", type=int, default=30)
    ap.add_argument("--out_fps", type=int, default=30)
    ap.add_argument("--loop", default="clamp", choices=["wrap", "clamp"])
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--no_slowdown", action="store_true")
    ap.add_argument("--skip_ik", action="store_true",
                    help="pure analytic q_ref output (debug/validator calib)")
    ap.add_argument("--single_warp", type=float, default=None,
                    help="skip the grid; run one warp factor (iteration)")
    args = ap.parse_args()

    frames = load_csv(args.csv)
    e = len(frames["pos"]) if args.end_frame < 0 else args.end_frame
    pos = frames["pos"][args.start_frame:e]
    quat = frames["quat_xyzw"][args.start_frame:e]
    dof = frames["dof"][args.start_frame:e]
    n = len(pos)
    dt = 1.0 / args.src_fps
    print(f"[retarget-v2] {args.csv}: {n} frames @ {args.src_fps} fps")

    rt = RetargeterV2()

    from build_x1_assets import parse_urdf_limits
    vel_lim = np.array([parse_urdf_limits()[j]["velocity"] for j in X1_DOF_ORDER])

    v_g1 = speed_profile(pos, dt)
    print(f"[retarget-v2] G1 speed: mean {v_g1.mean():.2f} "
          f"p99 {np.quantile(v_g1, 0.99):.2f} max {v_g1.max():.2f} m/s")
    v_x1_raw = v_g1 * rt.s_leg
    if args.no_slowdown:
        warp = 1.0
    else:
        warp = max(1.0, np.quantile(v_x1_raw, 0.99) / args.speed_cap)
        peak = v_x1_raw.max() / warp
        v_cap_peak = args.speed_cap * args.peak_ratio
        if peak > v_cap_peak:
            warp = v_x1_raw.max() / v_cap_peak

    g1_first = pos[0].copy()

    def run_pass(warp_f, skip_ik=False):
        n_out = max(int(round(n * warp_f * args.out_fps / args.src_fps)), 2)
        t_out = np.arange(n_out) / (args.out_fps * warp_f)
        t_out = np.minimum(t_out, (n - 1) * dt)
        idx_f = t_out * args.src_fps
        i0 = np.clip(np.floor(idx_f).astype(int), 0, n - 2)
        alpha = (idx_f - i0)[..., None]
        pos_s = pos[i0] * (1 - alpha) + pos[i0 + 1] * alpha
        quat_s = quat[i0] * (1 - alpha) + quat[i0 + 1] * alpha
        quat_s /= np.linalg.norm(quat_s, axis=1, keepdims=True)
        dof_s = dof[i0] * (1 - alpha) + dof[i0 + 1] * alpha

        base_p = np.zeros((n_out, 3))
        root_q = np.zeros((n_out, 4))
        dof_out = np.zeros((n_out, len(X1_DOF_ORDER)))
        q_prev = None
        cost_acc = []
        for i in range(n_out):
            t = rt.frame_targets(pos_s[i], quat_s[i], dof_s[i])
            qref = rt.q_ref(t["fk"])
            base_pos, root_quat = rt.map_root(t, g1_first,
                                              np.zeros(2))
            if skip_ik:
                q = qref
            else:
                q, cost = rt.solve_frame(t, qref, q_prev, base_pos, root_quat)
                cost_acc.append(cost)
            q_prev = q
            base_p[i] = base_pos
            root_q[i] = root_quat
            dof_out[i] = q
        mc = float(np.nanmean(cost_acc)) if cost_acc else 0.0
        return n_out, base_p, root_q, dof_out, mc

    def filter_and_lift(n_out, base_p, root_q, dof_out):
        from scipy.signal import butter, filtfilt
        if n_out >= 9:
            cuts = np.full(len(X1_DOF_ORDER), 8.0)
            for k, j in enumerate(X1_DOF_ORDER):
                if "wrist" in j or "lumbar" in j:
                    cuts[k] = 2.8
            for k in range(len(X1_DOF_ORDER)):
                fc = min(cuts[k], 0.9 * (args.out_fps / 2.0))
                b, a = butter(4, fc / (args.out_fps / 2.0))
                dof_out[:, k] = filtfilt(b, a, dof_out[:, k])
        smoothable = ["wrist", "lumbar", "elbow_yaw", "shoulder_yaw"]
        for _ in range(6):
            qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
            over = [qd[:, k].max() > 0.95 * vel_lim[k]
                    and any(s in X1_DOF_ORDER[k] for s in smoothable)
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
        # whole-body one-frame spike suppressor (IK branch hops): if any
        # joint still exceeds 1.02x, progressively lowpass just that joint
        for _ in range(4):
            qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
            bad = [k for k in range(len(X1_DOF_ORDER))
                   if qd[:, k].max() > 1.02 * vel_lim[k]]
            if not bad:
                break
            for k in bad:
                fc = 6.0
                while fc > 0.2:
                    b, a = butter(4, fc / (args.out_fps / 2.0))
                    cand = filtfilt(b, a, dof_out[:, k])
                    if np.abs(np.gradient(cand, 1.0 / args.out_fps)).max() <= 1.02 * vel_lim[k]:
                        dof_out[:, k] = cand
                        break
                    fc *= 0.6
        dof_out = np.minimum(np.maximum(dof_out, rt.x1_lower), rt.x1_upper)
        qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
        CRIT = {"hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"}
        crit_ratio = max(
            (np.quantile(qd[:, k], 0.99) / vel_lim[k])
            for k in range(len(X1_DOF_ORDER))
            if any(c in X1_DOF_ORDER[k] for c in CRIT))
        # ground lift (reuse v1 logic)
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
        # stance snap-down: the lift above only pushes UP. Scaled foot
        # targets often float 3-6 cm through stance (p10 sole z 2.2 cm on
        # measured failures — feet never plant), which breaks the step
        # rhythm (R1: X1 detects 3 spurious strikes, stances hover). For
        # frames whose lowest sole is already near ground (<5 cm) lower
        # the base so the sole reaches 2 mm; median-filtered, capped, and
        # never applied while clearly airborne.
        snap = np.clip(0.002 - minz_i, -0.05, 0.0)
        snap[minz_i > 0.05] = 0.0
        if n_out >= 7:
            k7 = np.ones(7) / 7
            snap_s = np.convolve(
                np.r_[snap[:3][::-1], snap, snap[-3:][::-1]], k7, mode="valid")
            snap = np.minimum(snap, snap_s)
        base_p[:, 2] += snap
        return base_p, dof_out, crit_ratio

    def save_result(warp_f, n_out_f, base_f, rootq_f, dof_f):
        k5 = np.ones(5) / 5
        base_f[:, 2] = np.convolve(
            np.r_[base_f[:2, 2][::-1], base_f[:, 2], base_f[-2:, 2][::-1]],
            k5, mode="valid")
        expmaps = np.stack([expmap_from_quat_wxyz(q) for q in rootq_f])
        frames_out = np.concatenate([base_f, expmaps, dof_f],
                                    axis=1).astype(np.float32)
        out = dict(loop_mode=(1 if args.loop == "wrap" else 0),
                   fps=args.out_fps, frames=frames_out.tolist(),
                   time_scale=float(warp_f), s_leg=float(rt.s_leg),
                   s_arm=float(rt.s_arm), version="v2")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(out, f)

    best = None       # (score, extra, n_out) — smaller is better
    best_extra = None
    extras = ((args.single_warp,) if args.single_warp
              else (1.0, 1.06, 1.15, 1.3, 1.5, 1.8, 2.2))
    for extra in extras:
        warp_try = warp * extra
        n_out, base_p, root_q, dof_raw, cost_mean = run_pass(
            warp_try, skip_ik=args.skip_ik)
        base_p, dof_out, crit_ratio = filter_and_lift(
            n_out, base_p.copy(), root_q, dof_raw.copy())
        qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
        ratio = (qd / vel_lim).max()
        print(f"[retarget-v2] warp x{warp_try:.3f} ({n_out} fr, IK cost "
              f"{cost_mean:.2f}) max|qdot| {qd.max():.2f} (ratio {ratio:.3f},"
              f" crit {crit_ratio:.3f})")
        save_result(warp_try, n_out, base_p, root_q, dof_out)
        if args.skip_ik:
            best_extra = extra
            break
        # rank: fewest failed gates first (validator verdict), then
        # critical-joint feasibility, then worst velocity ratio
        score = (0, 0 if crit_ratio <= 1.0 else 1, max(0.0, ratio))
        if best is None or score < best[0]:
            best = (score, extra, n_out)
        if ratio > 1.05 or crit_ratio > 1.0:
            continue
        try:
            from validate_retarget_v2 import validate as _val
            res = _val(args.csv, args.out, sample_step=4)
            ok = res.get("PASS", False)
            fails = [k for k, v in res.items()
                     if isinstance(v, dict) and v.get("pass_") is False]
            print(f"[retarget-v2] self-validate: "
                  f"{'PASS' if ok else 'FAIL ' + str(fails)}")
            n_fails = len(fails)
        except Exception as ex:
            ok, fails, n_fails = False, [f"validator error: {ex}"], 9
            print(f"[retarget-v2] self-validate error: {ex}")
        score = (n_fails, 0 if crit_ratio <= 1.0 else 1, max(0.0, ratio))
        if best is None or score < best[0]:
            best = (score, extra, n_out)
        if ok:
            best = (0.0, warp_try, n_out)
            best_extra = extra
            break
    if args.skip_ik:
        best_extra = extras[0]
    elif best is not None and best[0][0] > 0:
        # no fully-passing variant: regenerate the best-ranked one so the
        # saved file matches the announced warp (the loop's last save was
        # the last TRIED warp, not the best)
        extra = best[1]
        n_out, base_p, root_q, dof_raw, cost_mean = run_pass(warp * extra)
        base_p, dof_out, crit_ratio = filter_and_lift(
            n_out, base_p.copy(), root_q, dof_raw.copy())
        save_result(warp * extra, n_out, base_p, root_q, dof_out)
    if best is None:
        best = (0.0, warp, n_out)
    print(f"[retarget-v2] saved {args.out} (warp x{best[1]:.3f})")


if __name__ == "__main__":
    main()
