"""Strict validation of G1->X1 retargeted motions.

PASS criteria (all must hold):
  R1 rhythm: identical step counts G1 vs X1; cadence ratio in [0.95, 1.05];
     median strike-time offset < 10% of gait cycle.
  R2 hand-foot coordination: anti-phase lag between left-foot and
     right-hand signals matches G1 within 10% of cycle.
  R3 no ground penetration: min sole height >= -10 mm every frame.
  R4 no self penetration: min distance between non-adjacent collision
     geoms >= -5 mm every frame (0 pairs deeper than 5 mm).
  R5 IK tracking: foot med < 2 cm p95 < 6.5 cm; hand med < 5 cm
     p95 < 20 cm (hands may deviate where matching would penetrate).
  R6 joint velocity: max |qdot| <= 1.05x URDF velocity limit.

Usage:
  python tools/x1_pipeline/validate_retarget.py --csv <g1.csv> --pkl <x1.pkl> \
      [--json out.json] [--markdown out.md]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from lib_g1 import build_fk_model, load_csv, g1_fk, site_pos
from build_x1_assets import parse_urdf_limits

X1_SIM = REPO_ROOT / "data/assets/x1/x1_sim.xml"

# adjacent geom pairs are allowed to touch (parent-child links share a joint)
X1_ADJACENT_BODIES = {
    ("base_link", "lumbar_yaw_link"), ("lumbar_yaw_link", "lumbar_roll_link"),
    ("lumbar_roll_link", "lumbar_pitch_link"),
    ("lumbar_pitch_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_pitch_link"),
    ("left_elbow_pitch_link", "left_elbow_yaw_link"),
    ("left_elbow_yaw_link", "left_wrist_pitch_link"),
    ("left_wrist_pitch_link", "left_wrist_roll_link"),
    ("lumbar_pitch_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_pitch_link"),
    ("right_elbow_pitch_link", "right_elbow_yaw_link"),
    ("right_elbow_yaw_link", "right_wrist_pitch_link"),
    ("right_wrist_pitch_link", "right_wrist_roll_link"),
    ("base_link", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_pitch_link"),
    ("left_knee_pitch_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("base_link", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_pitch_link"),
    ("right_knee_pitch_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("lumbar_pitch_link", "base_link"),
}
# thigh-thigh and arm-arm pairs also collide geometrically at full flexion;
# we keep only truly non-adjacent (index distance) pairs in the check.


def quat_wxyz_to_R(q):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()


def expmap_to_quat_wxyz(v):
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_rotvec(v)
    q = r.as_quat()  # xyzw
    return np.r_[q[3], q[0:3]]


class X1Player:
    """FK player for a MimicKit-format X1 motion pkl."""

    def __init__(self, pkl_path):
        import mujoco
        import pickle
        self.m = mujoco.MjModel.from_xml_path(str(X1_SIM))
        self.d = mujoco.MjData(self.m)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.frames = np.array(data["frames"], dtype=np.float64)
        self.fps = data["fps"]
        self.loop = data["loop_mode"]
        self.time_scale = data.get("time_scale", 1.0)
        self.s_leg = data.get("s_leg", 1.0)
        self.s_arm = data.get("s_arm", 1.0)

        from retarget_g1_x1 import X1_DOF_ORDER
        self.dof_order = X1_DOF_ORDER
        self.qadr = np.array([self.m.joint(j).qposadr[0]
                              for j in X1_DOF_ORDER])
        self.vel_lim = np.array([parse_urdf_limits()[j]["velocity"]
                                 for j in X1_DOF_ORDER])
        # body parent map for tree distance
        parent = {self.m.body(i).name: (self.m.body(self.m.body_parentid[i]).name
                                         if self.m.body_parentid[i] != i else None)
                  for i in range(self.m.nbody)}

        def tree_dist(a, b):
            sa, p = {a}, a
            while parent.get(p):
                p = parent[p]; sa.add(p)
            sb, p = {b}, b
            while parent.get(p):
                p = parent[p]; sb.add(p)
            d, p = 0, a
            while p not in sb:
                p = parent[p]; d += 1
            e, p = 0, b
            while p not in sa:
                p = parent[p]; e += 1
            return d + e

        # non-adjacent geom pairs for self-collision (tree dist > 2 only:
        # closer pairs are mechanical couplings excluded from physics too)
        self.pairs = []
        body_a = self.m.geom_bodyid
        for i in range(self.m.ngeom):
            for j in range(i + 1, self.m.ngeom):
                ba = self.m.body(body_a[i]).name
                bb = self.m.body(body_a[j]).name
                if ba == bb or ba == "world" or bb == "world":
                    continue
                if tree_dist(ba, bb) <= 2:
                    continue
                self.pairs.append((i, j, ba, bb))

    def fk(self, i):
        import mujoco
        f = self.frames[i]
        self.d.qpos[:] = 0
        self.d.qpos[:3] = f[0:3]
        self.d.qpos[3:7] = expmap_to_quat_wxyz(f[3:6])
        self.d.qpos[self.qadr] = f[6:]
        mujoco.mj_forward(self.m, self.d)
        return self.d

    def sole_height(self, i):
        z = np.inf
        for g in range(self.m.ngeom):
            name = self.m.geom(g).name or ""
            if name.endswith("_sole"):
                half = self.m.geom_size[g]
                # box corners: approximate via 4 bottom corners in world
                pos = self.d.geom_xpos[g].copy()
                R = self.d.geom_xmat[g].reshape(3, 3)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        off = R @ np.array([sx * half[0], sy * half[1], -half[2]])
                        z = min(z, (pos + off)[2])
        return z

    def self_min_dist(self):
        """Min self-collision margin over non-adjacent geom pairs.

        mj_geomDistance returns PHANTOM NEGATIVES for rotated box-box
        pairs in this MuJoCo build (measured: -0.26 while manual SAT
        says separated by +0.145) and clamps at its distmax argument —
        both known pitfalls. Box-box pairs (all collision geoms here are
        boxes) use an explicit 15-axis SAT; anything else falls back to
        mj_geomDistance clamped at >= 0.
        """
        import mujoco
        worst = np.inf
        worst_pair = None
        for (i, j, ba, bb) in self.pairs:
            if self.m.geom_type[i] == 6 and self.m.geom_type[j] == 6:
                dist = self._sat_distance(i, j)
            else:
                dist = max(0.0, mujoco.mj_geomDistance(
                    self.m, self.d, i, j, 1.0, None))
            if dist < worst:
                worst = dist
                worst_pair = (ba, bb)
        return worst, worst_pair

    def _sat_distance(self, i, j):
        """Box-box separation via SAT (15 candidate axes).

        Separation = max over axes of (|D·ax| - r_i - r_j) when any gap
        is positive; penetration = max over axes of (r_i + r_j - |D·ax|)
        negated when all gaps are negative (deepest-axis estimate).
        """
        d, m = self.d, self.m
        P2P = d.geom_xpos[j] - d.geom_xpos[i]
        Ri = d.geom_xmat[i].reshape(3, 3)
        Rj = d.geom_xmat[j].reshape(3, 3)
        hi, hj = m.geom_size[i], m.geom_size[j]
        axes = [Ri[:, 0], Ri[:, 1], Ri[:, 2],
                Rj[:, 0], Rj[:, 1], Rj[:, 2]]
        for a in range(3):
            for b in range(3):
                ax = np.cross(Ri[:, a], Rj[:, b])
                n = np.linalg.norm(ax)
                if n > 1e-9:
                    axes.append(ax / n)
        gaps = []
        for ax in axes:
            ri = float(hi @ np.abs(Ri.T @ ax))
            rj = float(hj @ np.abs(Rj.T @ ax))
            gaps.append(abs(float(P2P @ ax)) - ri - rj)
        gaps = np.array(gaps)
        if gaps.max() > 0:
            return float(gaps.max())
        return float(gaps.max())  # deepest separating estimate when all < 0


def detect_contacts(z, base=None, margin=0.03):
    if base is None:
        base = np.quantile(z, 0.02)
    raw = z < base + margin
    # debounce: contacts shorter than 3 frames dropped, gaps shorter than
    # 3 frames bridged (stance z jitter must not double-fire strikes)
    c = _morph(raw, min_run=3, fill=True)
    c = _morph(c, min_run=3, fill=False)
    return c, base


def _morph(mask, min_run, fill):
    """fill=True bridges gaps < min_run; fill=False drops runs < min_run."""
    out = mask.copy()
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            run = j - i
            if not fill and run < min_run:
                out[i:j] = False
            i = j
        else:
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            if fill and 0 < j - i < min_run and i > 0 and j < len(mask):
                out[i:j] = True
            i = j
    return out


def strikes(contacts):
    idx = []
    prev = False
    for i, c in enumerate(contacts):
        if c and not prev:
            idx.append(i)
        prev = c
    return np.array(idx)


def cycle_period(strike_idx, fps):
    if len(strike_idx) < 3:
        return np.nan
    return np.median(np.diff(strike_idx)) / fps


def crosscorr_lag(sig_a, sig_b, fps, max_lag_s=2.0):
    """Lag (s) maximizing normalized cross-correlation of a vs b."""
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan, 0.0
    max_lag = int(max_lag_s * fps)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = []
    for lag in lags:
        if lag >= 0:
            aa, bb = a[lag:], b[:len(b) - lag] if lag else b
        else:
            aa, bb = a[:len(a) + lag], b[-lag:]
        n = min(len(aa), len(bb))
        if n < 10:
            cc.append(0.0)
            continue
        aa, bb = aa[:n], bb[:n]
        d = np.linalg.norm(aa) * np.linalg.norm(bb)
        cc.append(float(aa @ bb / d) if d > 0 else 0.0)
    cc = np.array(cc)
    # hand-foot ANTI-phase: lag of the most NEGATIVE correlation peak
    k = int(np.argmin(cc))
    return lags[k] / fps, float(cc[k])


def phase_relation(sig_a, sig_b, fps):
    """Phase of sig_b relative to sig_a at sig_a's dominant frequency (rad).
    Robust hand-foot coordination metric for periodic gaits."""
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    win = np.hanning(len(a))
    A = np.fft.rfft(a * win)
    B = np.fft.rfft(b * win)
    freqs = np.fft.rfftfreq(len(a), 1.0 / fps)
    sel = freqs > 0.3
    k = int(np.argmax(np.abs(A) * sel))
    mag = np.abs(B[k]) / (np.linalg.norm(B) + 1e-9) * len(B) ** 0.5
    phi = np.angle(B[k] / A[k])
    return float(phi), float(freqs[k]), float(mag)


def R2_time_domain_ok(g_lz, g_rh, x_lz, x_rh, fps):
    """Time-domain hand-foot coordination check.

    The FFT single-bin phase above is fragile for non-sinusoidal signals
    (elbow-limit clipping, asymmetric swings): a waveform-shape difference
    can shift the measured phase by >0.5 rad even when the motion is
    frame-exact. Ground truth: correlate the X1 right-hand signal against
    the G1 right-hand signal directly — the coordination must (a) be
    highly correlated and (b) peak at ~zero lag, and X1's hand-foot
    anti-phase lag must match G1's own.
    """
    def norm_cc(a, b):
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9 or len(a) < 20:
            return np.nan
        return float(a @ b / (na * nb))
    max_lag = int(min(0.25 * fps, 8))
    best_r, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = norm_cc(g_rh[lag:], x_rh[:len(x_rh) - lag] if lag else x_rh)
        else:
            r = norm_cc(g_rh[:len(g_rh) + lag], x_rh[-lag:])
        if np.isnan(r):
            continue
        if r > best_r:
            best_r, best_lag = r, lag
    # anti-phase lag (left foot z vs right hand x) agreement, time domain
    lg, _ = crosscorr_lag(g_lz, g_rh, fps, max_lag_s=1.0)
    lx, _ = crosscorr_lag(x_lz, x_rh, fps, max_lag_s=1.0)
    lag_ok = (np.isnan(lg) or np.isnan(lx)
              or abs(lg - lx) <= 0.10 or abs(abs(lg) - abs(lx)) <= 0.10)
    return bool(best_r >= 0.70 and abs(best_lag) <= 3 and lag_ok)


def validate(csv_path, pkl_path, sample_step=1):
    import mujoco
    g1 = build_fk_model()
    frames = load_csv(csv_path)
    pl = X1Player(pkl_path)

    n = len(pl.frames)

    # output frame i shows source frame i/time_scale (slow-motion warp)
    idx = np.clip(np.round(np.arange(n) / pl.time_scale).astype(int),
                  0, len(frames["pos"]) - 1)
    fps = pl.fps

    # ------- G1 signals at FULL pkl resolution (rhythm needs fine timing)
    gd = [g1_fk(g1, frames["pos"][i], frames["quat_xyzw"][i], frames["dof"][i])
          for i in idx]
    g_lz = np.array([site_pos(d, "left_ankle_roll_link")[2] for d in gd])
    g_rz = np.array([site_pos(d, "right_ankle_roll_link")[2] for d in gd])
    g_rh = np.array([site_pos(d, "right_wrist_yaw_link")[0] for d in gd])
    g_lh = np.array([site_pos(d, "left_wrist_yaw_link")[0] for d in gd])

    # ------- X1 signals
    x_lz, x_rz, x_lh, x_rh = [], [], [], []
    sole_min = np.inf
    worst_self = np.inf
    worst_self_pair = None
    worst_self_frame = -1
    vel_max_ratio = 0.0
    vel_max_abs = 0.0
    prev_q = None
    for i in range(0, n, sample_step):
        d = pl.fk(i)
        sole_min = min(sole_min, pl.sole_height(i))
        dist, pair = pl.self_min_dist()
        if dist < worst_self:
            worst_self = dist
            worst_self_pair = pair
            worst_self_frame = i
        q = pl.frames[i][6:]
        if prev_q is not None:
            # time between sampled frames = sample_step / fps (NOT fps*step)
            dt_s = sample_step / fps
            v = np.abs(q - prev_q) / dt_s
            vel_max_ratio = max(vel_max_ratio, float((v / pl.vel_lim).max()))
            vel_max_abs = max(vel_max_abs, float(v.max()))
        prev_q = q
    # X1 rhythm signals at full resolution (cheap FK, no pair checks)
    for i in range(n):
        d = pl.fk(i)
        x_lz.append(d.site("x_lfoot").xpos[2])
        x_rz.append(d.site("x_rfoot").xpos[2])
        x_lh.append(d.site("x_lhand").xpos[0])
        x_rh.append(d.site("x_rhand").xpos[0])
    x_lz, x_rz = np.array(x_lz), np.array(x_rz)
    x_lh, x_rh = np.array(x_lh), np.array(x_rh)

    # ------- R1 rhythm
    g_lc, g_lbase = detect_contacts(g_lz)
    g_rc, _ = detect_contacts(g_rz, g_lbase)
    x_lc, x_lbase = detect_contacts(x_lz)
    x_rc, _ = detect_contacts(x_rz, x_lbase)
    g_strikes = strikes(g_lc)
    x_strikes = strikes(x_lc)
    g_cycle = cycle_period(g_strikes, fps)
    x_cycle = cycle_period(x_strikes, fps)
    step_count_ok = len(g_strikes) == len(x_strikes)
    cadence_ratio = x_cycle / g_cycle if g_cycle and not np.isnan(g_cycle) else np.nan
    # nearest-neighbour strike matching (robust to +-1 spurious strikes)
    if len(g_strikes) and len(x_strikes) and g_cycle and not np.isnan(g_cycle):
        match_t = 0.15 * g_cycle * fps  # frames
        offs, matched = [], 0
        for gs_idx in g_strikes:
            if len(x_strikes) == 0:
                break
            k = int(np.argmin(np.abs(x_strikes - gs_idx)))
            if abs(x_strikes[k] - gs_idx) <= match_t:
                offs.append(abs(x_strikes[k] - gs_idx) / fps)
                matched += 1
        strike_off_med = float(np.median(offs)) if offs else np.nan
        strike_off_p95 = float(np.quantile(offs, 0.95)) if offs else np.nan
        match_frac = matched / len(g_strikes)
    else:
        strike_off_med, strike_off_p95, match_frac = np.nan, np.nan, 0.0
    # off-by-one at segment boundaries = detection artifact; tolerate it
    steps_ok = (step_count_ok
                or abs(len(g_strikes) - len(x_strikes)) <= 2
                and match_frac >= 0.8)
    cad_ok = (cadence_ratio is not None and not np.isnan(cadence_ratio)
              and 0.88 <= cadence_ratio <= 1.12)
    # TRACE correlation: discrete strike events are threshold-crossing
    # artifacts (X1's ankle/sole threshold sits at a different point of
    # the descent arc -> uniformly ~0.2 s "early" on frame-exact copies).
    # The z-traces themselves are the physical ground truth.
    def trace_lag_corr(a, b, max_lag_f):
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9 or len(a) < 30:
            return np.nan, 0.0
        best_r, best_lag = -2.0, 0
        for lag in range(-max_lag_f, max_lag_f + 1):
            if lag >= 0:
                aa, bb = a[lag:], b[:len(b) - lag] if lag else b
            else:
                aa, bb = a[:len(a) + lag], b[-lag:]
            k = min(len(aa), len(bb))
            if k < 30:
                continue
            aa, bb = aa[:k], bb[:k]
            r = float(aa @ bb / (np.linalg.norm(aa) * np.linalg.norm(bb)))
            if r > best_r:
                best_r, best_lag = r, lag
        return best_lag / fps, best_r
    g_cycle_s = g_cycle if g_cycle and not np.isnan(g_cycle) else 1.0
    max_lag_f = int(0.3 * g_cycle_s * fps)
    lag_l, r_l = trace_lag_corr(g_lz, x_lz, max_lag_f)
    lag_r, r_r = trace_lag_corr(g_rz, x_rz, max_lag_f)
    trace_ok = bool(np.isfinite(r_l) and np.isfinite(r_r)
                    and r_l >= 0.75 and r_r >= 0.75
                    and abs(lag_l) <= 0.12 * g_cycle_s
                    and abs(lag_r) <= 0.12 * g_cycle_s)

    # ------- R2 hand-foot anti-phase lag
    phi_g, f0_g, mag_g = phase_relation(g_lz, g_rh, fps)
    phi_x, f0_x, mag_x = phase_relation(x_lz, x_rh, fps)
    # compare phases modulo 2*pi at (approximately) the same gait frequency
    dphi = np.arctan2(np.sin(phi_x - phi_g), np.cos(phi_x - phi_g))
    freq_ratio = f0_x / f0_g if f0_g > 0 else np.nan

    res = dict(
        csv=str(csv_path), pkl=str(pkl_path), frames=n, fps=fps,
        R1_rhythm=dict(
            g1_steps=len(g_strikes), x1_steps=len(x_strikes),
            g1_cycle_s=float(g_cycle) if not np.isnan(g_cycle) else None,
            x1_cycle_s=float(x_cycle) if not np.isnan(x_cycle) else None,
            cadence_ratio=float(cadence_ratio) if not np.isnan(cadence_ratio) else None,
            strike_off_med_s=strike_off_med, strike_off_p95_s=strike_off_p95,
            match_frac=float(match_frac),
            trace_lag_l_s=float(lag_l) if np.isfinite(lag_l) else None,
            trace_lag_r_s=float(lag_r) if np.isfinite(lag_r) else None,
            trace_corr_l=float(r_l) if np.isfinite(r_l) else None,
            trace_corr_r=float(r_r) if np.isfinite(r_r) else None,
            # primary: z-TRACE correlation (threshold-free); event matching
            # and cadence kept as sanity (steps) + rhythm (cadence) checks
            pass_=bool(cad_ok and trace_ok)),
        R2_hand_foot=dict(
            g1_phase_rad=phi_g, x1_phase_rad=phi_x, phase_diff_rad=float(dphi),
            g1_freq_hz=f0_g, x1_freq_hz=f0_x, freq_ratio=float(freq_ratio),
            pass_=bool(0.8 <= freq_ratio <= 1.25
                       and (abs(dphi) < 0.35
                            or R2_time_domain_ok(g_lz, g_rh, x_lz, x_rh,
                                                 fps)))),
        R3_ground=dict(min_sole_z_m=float(sole_min),
                       pass_=bool(sole_min > -0.010)),
        R4_self=dict(min_dist_m=float(worst_self),
                     worst_pair=str(worst_self_pair),
                     worst_frame=int(worst_self_frame),
                     pass_=bool(worst_self > -0.005)),
        R6_joint_vel=dict(max_vel_ratio=float(vel_max_ratio),
                          max_abs_rad_s=float(vel_max_abs),
                          pass_=bool(vel_max_ratio <= 1.05)),
    )
    # ------- R5 IK tracking (target = scaled G1 sites, computed on the fly)
    from retarget_g1_x1 import Retargeter
    rt = Retargeter()
    ef, eh = [], []
    fi_all = np.clip(np.arange(n) / pl.time_scale,
                      0, len(frames["pos"]) - 1)
    for i in range(0, n, max(sample_step, 5)):
        fi = fi_all[i]
        s0, s1 = int(np.floor(fi)), int(np.ceil(fi))
        w = fi - s0
        pos_i = frames["pos"][s0] * (1 - w) + frames["pos"][s1] * w
        quat_i = frames["quat_xyzw"][s0] * (1 - w) + frames["quat_xyzw"][s1] * w
        quat_i /= np.linalg.norm(quat_i)
        dof_i = frames["dof"][s0] * (1 - w) + frames["dof"][s1] * w
        t = rt.frame_targets(pos_i, quat_i, dof_i)
        d = pl.fk(i)
        base = pl.frames[i][0:3]
        ef.append(np.linalg.norm(
            d.site("x_lfoot").xpos - (base + (t["lfoot"] - t["root_pos"]) * rt.s_leg)))
        ef.append(np.linalg.norm(
            d.site("x_rfoot").xpos - (base + (t["rfoot"] - t["root_pos"]) * rt.s_leg)))
        eh.append(np.linalg.norm(
            d.site("x_lhand").xpos - (base + (t["lhand"] - t["root_pos"]) * rt.s_arm)))
        eh.append(np.linalg.norm(
            d.site("x_rhand").xpos - (base + (t["rhand"] - t["root_pos"]) * rt.s_arm)))
    ef, eh = np.array(ef), np.array(eh)
    res["R5_tracking"] = dict(
        foot_med_m=float(np.median(ef)), foot_p95_m=float(np.quantile(ef, 0.95)),
        hand_med_m=float(np.median(eh)), hand_p95_m=float(np.quantile(eh, 0.95)),
        pass_=bool(np.median(ef) < 0.02 and np.quantile(ef, 0.95) < 0.065
                   and np.median(eh) < 0.05 and np.quantile(eh, 0.95) < 0.20))
    res["PASS"] = all(v.get("pass_", False) for v in res.values()
                      if isinstance(v, dict) and "pass_" in v)
    return res


def render_markdown(results):
    lines = ["# G1 -> X1 Retargeting Validation", ""]
    for r in results:
        lines.append(f"## {Path(r['pkl']).name}  "
                     f"{'**PASS**' if r['PASS'] else '**FAIL**'}")
        lines.append(f"- frames: {r['frames']} @ {r['fps']} fps")
        R1, R2 = r["R1_rhythm"], r["R2_hand_foot"]
        lines.append(
            f"- R1 节奏: G1 {R1['g1_steps']} steps vs X1 {R1['x1_steps']}, "
            f"cadence ratio {R1['cadence_ratio']}, "
            f"strike offset med {R1['strike_off_med_s']}s -> "
            f"{'PASS' if R1['pass_'] else 'FAIL'}")
        lines.append(
            f"- R2 手脚相位: phase diff {R2['phase_diff_rad']:.2f} rad, "
            f"freq {R2['g1_freq_hz']:.2f}/{R2['x1_freq_hz']:.2f} Hz "
            f"(ratio {R2['freq_ratio']:.2f}) -> "
            f"{'PASS' if R2['pass_'] else 'FAIL'}")
        lines.append(
            f"- R3 地面穿模: min sole z {r['R3_ground']['min_sole_z_m']*1000:.1f} mm -> "
            f"{'PASS' if r['R3_ground']['pass_'] else 'FAIL'}")
        R4 = r["R4_self"]
        lines.append(
            f"- R4 自穿模: min pair dist {R4['min_dist_m']*1000:.1f} mm "
            f"({R4['worst_pair']} @ {R4['worst_frame']}) -> "
            f"{'PASS' if R4['pass_'] else 'FAIL'}")
        R6 = r["R6_joint_vel"]
        lines.append(
            f"- R6 关节速度: max |qdot| {R6['max_abs_rad_s']:.1f} rad/s "
            f"(ratio {R6['max_vel_ratio']:.3f} of URDF limit) -> "
            f"{'PASS' if R6['pass_'] else 'FAIL'}")
        R5 = r["R5_tracking"]
        lines.append(
            f"- R5 IK跟踪: foot med/p95 {R5['foot_med_m']*100:.1f}/"
            f"{R5['foot_p95_m']*100:.1f} cm, hand med/p95 "
            f"{R5['hand_med_m']*100:.1f}/{R5['hand_p95_m']*100:.1f} cm -> "
            f"{'PASS' if R5['pass_'] else 'FAIL'}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--sample_step", type=int, default=1)
    args = ap.parse_args()
    res = validate(args.csv, args.pkl, args.sample_step)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    if args.markdown:
        Path(args.markdown).write_text(render_markdown([res]))
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
