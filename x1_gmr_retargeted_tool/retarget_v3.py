"""G1 -> X1 retargeting v3 — fixes the three visible artifacts of v2.

Driven by quantified failures of the v2 data re-measured on the FIXED
sole-box assets (see output/diag_v3/diag_v2data_fixed.py):
  * soles buried 8.0-8.6 cm below ground for 63-88% of frames
    (the v2 assets' broken sole boxes hid this: left box hung 7.4 cm below
    the true sole, right box sat on the dorsum);
  * stance-foot sole tilt 20-60 deg ("脚底不平": foot-quat IK weight 0.5
    was far too weak vs position weight 12-30);
  * hip_roll/yaw |2nd-diff| spikes to 26-32 deg/frame^2 (analytic q_ref
    roughness p99 ~11.5 deg + isolated IK branch hops).

v3 changes:
  1. STANCE FOOT LEVELING — per foot, a stance weight w(clearance) ramps
     0->1 as the G1 source foot's ground clearance drops 6 cm -> 2 cm
     (5-frame median-smoothed). The IK foot-orientation target is slerp'd
     toward a yaw-only version of the X1 standing foot quat (sole parallel
     to ground), and the foot-quat IK weight ramps 0.5 -> 4.0 with w.
  2. GROUND CLOSURE — sole min-z is measured from the FIXED boxes (8
     corners) on the FINAL joint sequence, and the base z is lifted in a
     closed loop (<=5 iters) until every frame's min sole z >= +1 mm,
     AFTER all dof smoothing (v2 lifted before a final base smoothing that
     could re-bury the feet). Snap-down for near-ground frames follows and
     is re-closed. The old magic foot floor 0.058 is derived from the
     asset (0.043 sole offset + 3 mm) = 0.046.
  3. HIP JITTER — hip_roll/hip_yaw Butterworth cutoff lowered 8 -> 5 Hz,
     analytic q_ref lowpassed (6 Hz) on hip_roll/yaw BEFORE IK, and a
     Hampel (3-tap rolling median, 6-sigma MAD) spike suppressor on all
     leg joints AFTER IK to kill isolated branch hops.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from lib_g1 import (build_fk_model, load_csv, g1_fk,
                    site_pos, site_quat, quat_xyzw_to_wxyz)

X1_XML = REPO_ROOT / "data/assets/x1/x1_sim.xml"

LAT_WIDEN = 0.05          # unchanged from v2
FOOT_Z_FLOOR = 0.046      # sole plane 0.043 below ankle + 3 mm clearance
STANCE_C0, STANCE_C1 = 0.02, 0.06   # clearance ramp for leveling weight

from retarget_v2 import (X1_DOF_ORDER, quat_mul, quat_conj, quat_log_diff,
                         rotvec_to_quat_wxyz, quat_to_R, expmap_from_quat_wxyz,
                         speed_profile)
from build_x1_assets import parse_urdf_limits


def quat_slerp_wxyz(q1, q2, w):
    """Slerp between two wxyz quats; w=1 -> q2."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    d = float(q1 @ q2)
    if d < 0.0:
        q2 = -q2
        d = -d
    if d > 0.9995:
        q = q1 + w * (q2 - q1)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th0)
    q = (np.sin((1.0 - w) * th0) / s) * q1 + (np.sin(w * th0) / s) * q2
    return q / np.linalg.norm(q)


class RetargeterV3:
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
        self.x1_stand_z = (site_pos(x1d, "x_base")[2]
                           - max(site_pos(x1d, "x_lfoot")[2],
                                 site_pos(x1d, "x_rfoot")[2]))
        self.g1_stand_z = (site_pos(g1d, "pelvis")[2]
                           - max(site_pos(g1d, "left_ankle_roll_link")[2],
                                 site_pos(g1d, "right_ankle_roll_link")[2]))
        # G1 ankle-site height above its sole at standing = clearance offset
        self.g1_ankle_z0 = min(site_pos(g1d, "left_ankle_roll_link")[2],
                               site_pos(g1d, "right_ankle_roll_link")[2])
        print(f"[retarget-v3] s_leg {self.s_leg:.4f} s_arm {self.s_arm:.4f} "
              f"X1 stand {self.x1_stand_z:.3f} G1 stand {self.g1_stand_z:.3f} "
              f"g1 ankle z0 {self.g1_ankle_z0:.3f}")

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
            lclear=site_pos(gd, "left_ankle_roll_link")[2] - self.g1_ankle_z0,
            rclear=site_pos(gd, "right_ankle_roll_link")[2] - self.g1_ankle_z0,
            fk=gd,
        )

    def map_root(self, t, g1_first, x1_anchor_xy):
        rel = t["root_pos"] - g1_first
        xy = x1_anchor_xy + rel[:2] * self.s_leg
        z = self.x1_stand_z + (t["root_pos"][2] - self.g1_stand_z) * self.s_leg
        return np.array([xy[0], xy[1], z]), t["root_quat_wxyz"]

    def level_quat(self, key, q_target):
        """Yaw-only version of the X1 neutral foot quat at the target yaw."""
        R_t = quat_to_R(np.asarray(q_target, dtype=float))
        R_n = quat_to_R(self.x1_neutral[key])
        yaw_t = np.arctan2(R_t[1, 2], R_t[0, 2])   # forward = local z axis
        yaw_n = np.arctan2(R_n[1, 2], R_n[0, 2])
        d = yaw_t - yaw_n
        c, s = np.cos(d), np.sin(d)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        from scipy.spatial.transform import Rotation as Rot
        q = Rot.from_matrix(Rz @ R_n).as_quat()      # xyzw
        return np.r_[q[3], q[0:3]]

    def solve_frame(self, t, q_ref, q_prev, base_pos, root_quat,
                    stance_w=(0.0, 0.0)):
        """Legs-only IK (v2 design) + stance leveling (v3)."""
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
        foot_targets[:, 2] = np.maximum(foot_targets[:, 2], FOOT_Z_FLOOR)

        lat_gap = abs(foot_targets[0, 1] - foot_targets[1, 1])
        foot_w = 30.0 if lat_gap > 0.12 else 12.0

        def delta_target(g_quat, key):
            dR = quat_mul(quat_conj(self.g1_neutral[key]), np.array(g_quat))
            return quat_mul(self.x1_neutral[key], dR)

        # v3.4: stance orientation is enforced as a 2-dof FLAT-SOLE
        # constraint (sole normal || gravity, yaw-free) instead of a slerp'd
        # quat target fighting the position terms. Per leg the stance
        # constraint set is then: xy pos (2) + flat sole (2) + yaw prior ->
        # well-posed for 6 dof, no weight war. Swing keeps the v2 weak
        # transferred-quat tracking.
        foot_quat_targets = [delta_target(t["lfoot_quat"], "lfoot"),
                             delta_target(t["rfoot_quat"], "rfoot")]

        foot_ids = [m.site("x_lfoot").id, m.site("x_rfoot").id]

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

        # v3.5: swing-leg IK branch flips on hip_roll/yaw showed up as the
        # visible "外展抖动" (alternating +-0.5 rad solutions frame to frame
        # while the other foot is planted). Pin those joints harder to the
        # (smoothed) analytic reference and to the previous solution.
        pin = np.ones(len(ik_idx))
        cvw = np.full(len(ik_idx), W_CV)
        for kk, jj in enumerate([X1_DOF_ORDER[i] for i in ik_idx]):
            if "hip_roll" in jj or "hip_yaw" in jj:
                pin[kk] = 3.0
                cvw[kk] = 0.8

        def residuals(qv):
            set_state(qv)
            res = []
            for i, sid in enumerate(foot_ids):
                dp = d.site_xpos[sid] - foot_targets[i]
                sw_i = stance_w[i]
                # release vertical position weight during stance (the
                # ground closure fixes z; orientation matters more there)
                wz = foot_w * (1.0 - sw_i) + 2.0 * sw_i
                res.append(np.array([foot_w, foot_w, wz]) * dp)
                # swing: weak transferred-quat tracking (as v2)
                rvec = quat_log_diff(
                    body_world_quat_from_site(d, sid), foot_quat_targets[i])
                res.append(0.5 * rvec)
                # stance: flat-sole constraint (yaw-free, 2 dof)
                if sw_i > 1e-3:
                    yv = d.site_xmat[sid].reshape(3, 3) @ np.array([0, 1.0, 0.0])
                    sgn = 1.0 if yv[2] >= 0 else -1.0
                    res.append(25.0 * sw_i * np.array([sgn * yv[0],
                                                       sgn * yv[1]]))
            res.append(pin * (qv - q_ref[ik_idx]))
            if q_prev is not None:
                res.append(cvw * (qv - q_prev[ik_idx]))
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


# ------------------------------------------------------------- post filters
def hampel_legs(dof_out, dof_order, n_iter=2, window=3, n_sigma=6.0):
    """Kill isolated 1-2 frame IK branch hops on the leg joints."""
    from scipy.signal import medfilt
    leg_idx = [k for k, j in enumerate(dof_order)
               if ("hip" in j) or ("knee" in j) or ("ankle" in j)]
    for _ in range(n_iter):
        changed = 0
        for k in leg_idx:
            x = dof_out[:, k]
            med = medfilt(x, window)
            mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
            bad = np.abs(x - med) > n_sigma * mad
            if bad.any():
                x[bad] = med[bad]
                changed += int(bad.sum())
        if not changed:
            break
    return dof_out


def smoothref_q(dof_out, dof_order, fps):
    """4 Hz lowpass on hip_roll/hip_yaw of the analytic reference."""
    from scipy.signal import butter, filtfilt
    for k, j in enumerate(dof_order):
        if "hip_roll" in j or "hip_yaw" in j:
            fc = min(4.0, 0.9 * fps / 2.0)
            b, a = butter(4, fc / (fps / 2.0))
            dof_out[:, k] = filtfilt(b, a, dof_out[:, k])
    return dof_out


def level_planted_feet(rt, base_p, root_q, dof_out, dof_order,
                       plant_z=0.012, tilt_tol_deg=4.0, max_iter=12):
    """Post-IK analytic sole leveling on planted frames.

    For each frame where a foot's sole min z < plant_z, solve that foot's
    ankle_pitch/ankle_roll (2 dof <-> sole pitch/roll) to level the sole,
    with a smoothstep blend so the correction ramps to zero at the plant
    boundary (no one-frame ankle spikes). Clip to joint limits.

    Returns (dof_out, n_leveled_frames).
    """
    import mujoco
    from scipy.optimize import least_squares
    m, d = rt.x1, mujoco.MjData(rt.x1)
    qadr = rt.x1_qadr
    dof_out = dof_out.copy()
    n_lv = 0
    for i in range(len(base_p)):
        d.qpos[:] = 0
        d.qpos[:3] = base_p[i]
        d.qpos[3:7] = root_q[i]
        d.qpos[qadr] = dof_out[i]
        mujoco.mj_forward(m, d)
        for side in ("left", "right"):
            gid = m.geom(f"{side}_ankle_roll_link_sole").id
            R = d.geom_xmat[gid].reshape(3, 3)
            h = m.geom_size[gid]
            c = d.geom_xpos[gid]
            zmin = min((c + R @ np.array([sx*h[0], sy*h[1], sz*h[2]]))[2]
                       for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
            if zmin >= plant_z:
                continue
            # blend weight: 1 at ground, 0 at plant_z
            w = float(np.clip((plant_z - zmin) / (plant_z - 0.001), 0.0, 1.0))
            if w <= 0.05:
                continue
            jn_p = f"{side}_ankle_pitch_joint"
            jn_r = f"{side}_ankle_roll_joint"
            kp_, kr_ = dof_order.index(jn_p), dof_order.index(jn_r)
            qp0, qr0 = dof_out[i, kp_], dof_out[i, kr_]
            lo = np.array([rt.x1_lower[kp_], rt.x1_lower[kr_]])
            hi = np.array([rt.x1_upper[kp_], rt.x1_upper[kr_]])

            def tilt_of(qv):
                d.qpos[qadr[kp_]] = qv[0]
                d.qpos[qadr[kr_]] = qv[1]
                mujoco.mj_forward(m, d)
                Rg = d.geom_xmat[gid].reshape(3, 3)
                y = Rg @ np.array([0.0, 1.0, 0.0])
                return np.degrees(np.arccos(np.clip(abs(y[2]), 0.0, 1.0)))

            t0 = tilt_of(np.array([qp0, qr0]))
            if t0 <= tilt_tol_deg:
                continue
            try:
                sol = least_squares(
                    lambda qv: np.array([tilt_of(qv)]),
                    np.array([qp0, qr0]),
                    bounds=(lo + 1e-4, hi - 1e-4), max_nfev=max_iter)
                qs, t1 = sol.x, tilt_of(sol.x)
            except Exception:
                continue
            if t1 >= t0:
                continue
            dof_out[i, kp_] = (1 - w) * qp0 + w * qs[0]
            dof_out[i, kr_] = (1 - w) * qr0 + w * qs[1]
            n_lv += 1
    return dof_out, n_lv


def sole_minz_all(rt, base_p, root_q, dof_out):
    """Min world z over BOTH feet's sole-box 8 corners, per frame."""
    import mujoco
    d = mujoco.MjData(rt.x1)
    sole_gids = [g for g in range(rt.x1.ngeom)
                 if (rt.x1.geom(g).name or "").endswith("_sole")]
    n = len(base_p)
    minz_i = np.zeros(n)
    for i in range(n):
        d.qpos[:] = 0
        d.qpos[:3] = base_p[i]
        d.qpos[3:7] = root_q[i]          # wxyz quat
        d.qpos[rt.x1_qadr] = dof_out[i]
        mujoco.mj_forward(rt.x1, d)
        minz = np.inf
        for g in sole_gids:
            R = d.geom_xmat[g].reshape(3, 3)
            h = rt.x1.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        c = (d.geom_xpos[g] + R @ np.array(
                            [sx * h[0], sy * h[1], sz * h[2]]))[2]
                        minz = min(minz, c)
        minz_i[i] = minz
    return minz_i


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
    ap.add_argument("--skip_ik", action="store_true")
    ap.add_argument("--single_warp", type=float, default=None)
    ap.add_argument("--fast_validate", action="store_true",
                    help="warp loop early-exits on the cheap R7/R8/R9 FK "
                         "gates only (full validator runs separately)")
    args = ap.parse_args()

    frames = load_csv(args.csv)
    e = len(frames["pos"]) if args.end_frame < 0 else args.end_frame
    pos = frames["pos"][args.start_frame:e]
    quat = frames["quat_xyzw"][args.start_frame:e]
    dof = frames["dof"][args.start_frame:e]
    n = len(pos)
    dt = 1.0 / args.src_fps
    print(f"[retarget-v3] {args.csv}: {n} frames @ {args.src_fps} fps")

    rt = RetargeterV3()

    vel_lim = np.array([parse_urdf_limits()[j]["velocity"] for j in X1_DOF_ORDER])

    v_g1 = speed_profile(pos, dt)
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

        # pass 1: FK targets + analytic q_ref for every frame
        t_list, qref_all = [], []
        base_p = np.zeros((n_out, 3))
        root_q = np.zeros((n_out, 4))
        for i in range(n_out):
            t = rt.frame_targets(pos_s[i], quat_s[i], dof_s[i])
            t_list.append(t)
            qref_all.append(rt.q_ref(t["fk"]))
        qref_all = np.array(qref_all)
        # v3.3: smooth the analytic hip_roll/yaw reference (jitter source 1)
        qref_all = smoothref_q(qref_all, X1_DOF_ORDER, args.out_fps)

        # stance weights from G1-side clearance, median-smoothed
        from scipy.signal import medfilt
        sw = np.zeros((n_out, 2))
        for k, key in enumerate(("lclear", "rclear")):
            clear = np.array([t[key] for t in t_list])
            w = np.clip((STANCE_C1 - clear) / (STANCE_C1 - STANCE_C0), 0.0, 1.0)
            w = medfilt(w, 5)
            sw[:, k] = w

        # pass 2: IK
        dof_out = np.zeros((n_out, len(X1_DOF_ORDER)))
        q_prev = None
        cost_acc = []
        for i in range(n_out):
            t = t_list[i]
            base_pos, root_quat = rt.map_root(t, g1_first, np.zeros(2))
            if skip_ik:
                q = qref_all[i]
            else:
                q, cost = rt.solve_frame(t, qref_all[i], q_prev, base_pos,
                                         root_quat, tuple(sw[i]))
                cost_acc.append(cost)
            q_prev = q
            base_p[i] = base_pos
            root_q[i] = root_quat
            dof_out[i] = q
        mc = float(np.nanmean(cost_acc)) if cost_acc else 0.0
        return n_out, base_p, root_q, dof_out, mc

    def filter_and_lift(n_out, base_p, root_q, dof_out):
        from scipy.signal import butter, filtfilt
        # final base-z smoothing FIRST (v2 did it after lifting, which could
        # re-bury feet; v3 lifts after all smoothing)
        k5 = np.ones(5) / 5
        base_p[:, 2] = np.convolve(
            np.r_[base_p[:2, 2][::-1], base_p[:, 2], base_p[-2:, 2][::-1]],
            k5, mode="valid")

        if n_out >= 9:
            cuts = np.full(len(X1_DOF_ORDER), 8.0)
            for k, j in enumerate(X1_DOF_ORDER):
                if "wrist" in j or "lumbar" in j:
                    cuts[k] = 2.8
                elif "hip_roll" in j or "hip_yaw" in j:
                    cuts[k] = 3.5        # v3: tighter on abduction/yaw
            for k in range(len(X1_DOF_ORDER)):
                fc = min(cuts[k], 0.9 * (args.out_fps / 2.0))
                b, a = butter(4, fc / (args.out_fps / 2.0))
                dof_out[:, k] = filtfilt(b, a, dof_out[:, k])

        # v3: Hampel spike suppression on legs (isolated IK branch hops);
        # wider window on hip_roll/yaw (2-3 frame oscillations)
        dof_out = hampel_legs(dof_out, X1_DOF_ORDER)
        dof_out = hampel_legs(dof_out, X1_DOF_ORDER, window=5, n_sigma=5.0)

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
        dof_out = hampel_legs(dof_out, X1_DOF_ORDER)

        # ---- v3.2: analytic sole leveling on planted frames
        dof_out, n_lv = level_planted_feet(rt, base_p, root_q, dof_out,
                                           X1_DOF_ORDER)
        dof_out = np.minimum(np.maximum(dof_out, rt.x1_lower), rt.x1_upper)
        print(f"[retarget-v3] planted-foot leveling: {n_lv} frames adjusted")

        # ---- v3.6: surgical velocity fix AFTER the leveling (the leveling's
        # per-frame ankle corrections reintroduce isolated spikes that the
        # earlier suppressors can no longer see; blend only offending
        # frames). Metric = per-frame FORWARD differences — the quantity
        # the validator's R6 measures (np.gradient halves single-frame
        # spikes and let ratio-1.5 spikes slip through on sprint seg1).
        for _ in range(12):
            qd_now = np.zeros_like(dof_out)
            qd_now[:-1] = np.abs(np.diff(dof_out, axis=0)) * args.out_fps
            bad = np.where((qd_now > 1.02 * vel_lim[None, :]).any(axis=1))[0]
            if not len(bad):
                break
            for i in bad:
                lo, hi = max(0, i - 1), min(n_out - 1, i + 1)
                for k in range(len(X1_DOF_ORDER)):
                    if qd_now[i, k] > 1.02 * vel_lim[k]:
                        dof_out[i, k] = (0.25 * dof_out[lo, k]
                                         + 0.5 * dof_out[i, k]
                                         + 0.25 * dof_out[hi, k])
            dof_out = np.minimum(np.maximum(dof_out, rt.x1_lower),
                                 rt.x1_upper)
        dof_out = hampel_legs(dof_out, X1_DOF_ORDER)
        qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
        CRIT = {"hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"}
        crit_ratio = max(
            (np.quantile(qd[:, k], 0.99) / vel_lim[k])
            for k in range(len(X1_DOF_ORDER))
            if any(c in X1_DOF_ORDER[k] for c in CRIT))

        # ---- v3 ground closure: iterate lift until every frame >= +1 mm
        for _ in range(5):
            minz_i = sole_minz_all(rt, base_p, root_q, dof_out)
            need = np.maximum(0.0, 0.001 - minz_i)
            if need.max() < 1e-4:
                break
            need = np.minimum(need, 0.10)
            need = np.array([need[max(0, i - 1):i + 2].max()
                             for i in range(n_out)])
            base_p[:, 2] += need

        # stance snap-down (as v2), then re-close
        minz_i = sole_minz_all(rt, base_p, root_q, dof_out)
        snap = np.clip(0.002 - minz_i, -0.05, 0.0)
        snap[minz_i > 0.05] = 0.0
        if n_out >= 7:
            k7 = np.ones(7) / 7
            snap_s = np.convolve(
                np.r_[snap[:3][::-1], snap, snap[-3:][::-1]], k7, mode="valid")
            snap = np.minimum(snap, snap_s)
        base_p[:, 2] += snap
        for _ in range(3):
            minz_i = sole_minz_all(rt, base_p, root_q, dof_out)
            need = np.maximum(0.0, 0.0005 - minz_i)
            if need.max() < 1e-4:
                break
            base_p[:, 2] += need
        minz_i = sole_minz_all(rt, base_p, root_q, dof_out)
        print(f"[retarget-v3] ground closure: min sole z = {minz_i.min()*1000:.1f} mm"
              f" (target >= +0.5 mm)")
        return base_p, dof_out, crit_ratio, float(minz_i.min())

    def save_result(warp_f, n_out_f, base_f, rootq_f, dof_f):
        expmaps = np.stack([expmap_from_quat_wxyz(q) for q in rootq_f])
        frames_out = np.concatenate([base_f, expmaps, dof_f],
                                    axis=1).astype(np.float32)
        out = dict(loop_mode=(1 if args.loop == "wrap" else 0),
                   fps=args.out_fps, frames=frames_out.tolist(),
                   time_scale=float(warp_f), s_leg=float(rt.s_leg),
                   s_arm=float(rt.s_arm), version="v3")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(out, f)

    best = None
    best_extra = None
    extras = ((args.single_warp,) if args.single_warp
              else (1.0, 1.06, 1.15, 1.3, 1.5, 1.8, 2.2))
    for extra in extras:
        warp_try = warp * extra
        n_out, base_p, root_q, dof_raw, cost_mean = run_pass(
            warp_try, skip_ik=args.skip_ik)
        base_p, dof_out, crit_ratio, minz = filter_and_lift(
            n_out, base_p.copy(), root_q, dof_raw.copy())
        qd = np.abs(np.gradient(dof_out, 1.0 / args.out_fps, axis=0))
        ratio = (qd / vel_lim).max()
        print(f"[retarget-v3] warp x{warp_try:.3f} ({n_out} fr, IK cost "
              f"{cost_mean:.2f}) max|qdot| {qd.max():.2f} (ratio {ratio:.3f},"
              f" crit {crit_ratio:.3f})")
        save_result(warp_try, n_out, base_p, root_q, dof_out)
        if args.skip_ik:
            best_extra = extra
            break
        score = (0, 0 if crit_ratio <= 1.0 else 1, max(0.0, ratio))
        if best is None or score < best[0]:
            best = (score, extra, n_out)
        if ratio > 1.05 or crit_ratio > 1.0:
            continue
        if args.fast_validate:
            try:
                from validate_retarget_v3 import new_gates
                ng = new_gates(args.out, args.csv)
                ok = all(ng[k]["pass_"] for k in
                         ("R7_stance_flat", "R8_ground_final", "R9_hip_jerk"))
                fails = [k for k, v in ng.items() if not v["pass_"]]
                print(f"[retarget-v3] fast-gates: "
                      f"{'PASS' if ok else 'FAIL ' + str(fails)}")
            except Exception as ex:
                ok, n_fails = False, 9
                print(f"[retarget-v3] fast-gates error: {ex}")
            score = ((0 if ok else 1),
                     0 if crit_ratio <= 1.0 else 1, max(0.0, ratio))
            if best is None or score < best[0]:
                best = (score, extra, n_out)
            if ok:
                best = ((0, 0, 0.0), warp_try, n_out)
                best_extra = extra
                break
            continue
        try:
            from validate_retarget_v3 import validate as _val
            res = _val(args.csv, args.out, sample_step=4)
            ok = res.get("PASS", False)
            fails = [k for k, v in res.items()
                     if isinstance(v, dict) and v.get("pass_") is False]
            print(f"[retarget-v3] self-validate: "
                  f"{'PASS' if ok else 'FAIL ' + str(fails)}")
            n_fails = len(fails)
        except Exception as ex:
            ok, fails, n_fails = False, [f"validator error: {ex}"], 9
            print(f"[retarget-v3] self-validate error: {ex}")
        score = (n_fails, 0 if crit_ratio <= 1.0 else 1, max(0.0, ratio))
        if best is None or score < best[0]:
            best = (score, extra, n_out)
        if ok:
            best = ((0, 0, 0.0), warp_try, n_out)
            best_extra = extra
            break
    if args.skip_ik:
        best_extra = extras[0]
    elif best is not None and best[0][0] > 0:
        extra = best[1]
        n_out, base_p, root_q, dof_raw, cost_mean = run_pass(warp * extra)
        base_p, dof_out, crit_ratio, minz = filter_and_lift(
            n_out, base_p.copy(), root_q, dof_raw.copy())
        save_result(warp * extra, n_out, base_p, root_q, dof_out)
    if best is None:
        best = ((0, 0, 0.0), warp, n_out)
    print(f"[retarget-v3] saved {args.out} (warp x{best[1]:.3f})")


if __name__ == "__main__":
    main()
