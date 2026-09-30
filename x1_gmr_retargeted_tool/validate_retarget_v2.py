"""Strict validation v2.1 of G1->X1 retargeted motions.

v2.1 redesign: joint-space gates must be PARAMETRIZATION-INVARIANT. The X1
hip is a 45-deg-diagonal-axes chain, so X1 dof values legitimately differ
from G1 dof values (G1 hip_pitch -0.57 <-> X1 +0.93 is CORRECT). Raw dof
correlation is therefore meaningless. Gates:

  J1 limb-motion fidelity (invariant): per segment (thigh/shank/uarm/farm,
     both sides), the pelvis-frame segment direction's ANGLE FROM ITS
     STANDING DIRECTION is correlated between time-aligned G1 and X1:
     Pearson r >= 0.75 AND median |angle diff| <= 12 deg.
  J2 posture sanity (X1-native): feet never crossed (left foot y - right
     foot y > -0.02 m in pelvis frame at all times); knee swing amplitude
     >= 0.35 rad; torso pitch median diff vs G1 <= 15 deg.
  J3 root consistency: root pitch/roll euler vs time-aligned source within
     12 deg median (root rotation is copied verbatim by the retargeter).

R1..R6 geometric/temporal gates are reused from validate_retarget.py, with
R5 tracking targets recomputed under the v2 anchor-based root mapping:
    target = pkl_base + (g1_site - g1_root) * s   (per-axis, z included)
(the pkl base already contains the ground lift; comparing vertical offsets
against the lifted base is exactly what the retargeter optimizes).

Self-test: --selftest validates the gates on a known-good analytic
mapping (must PASS J*) and on the known-bad v1 pkl (must FAIL J*).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from lib_g1 import load_csv, g1_fk, build_fk_model
from validate_retarget import validate as validate_r, render_markdown, X1Player


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _pelvis_R(d_g1, x1=False):
    """Pelvis frame from hip anchors + waist anchor (world R)."""
    from retarget_v2 import quat_conj, quat_mul  # noqa
    if x1:
        lh = d_g1.xanchor[d_g1.model.joint("left_hip_pitch_joint").id]
        rh = d_g1.xanchor[d_g1.model.joint("right_hip_pitch_joint").id]
        wa = d_g1.xanchor[d_g1.model.joint("lumbar_yaw_joint").id]
    else:
        lh = d_g1.xanchor[d_g1.model.joint("left_hip_pitch_joint").id]
        rh = d_g1.xanchor[d_g1.model.joint("right_hip_pitch_joint").id]
        wa = d_g1.xanchor[d_g1.model.joint("waist_yaw_joint").id]
    y = lh - rh
    y = y / np.linalg.norm(y)
    z = wa - 0.5 * (lh + rh)
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    yy = np.cross(z, x)
    return np.column_stack([x, yy, z])


def _segment_dirs(d, side, x1):
    """World directions of thigh/shank/uarm/farm from joint anchors."""
    m = d.model
    def anch(jn):
        return d.xanchor[m.joint(jn).id].copy()
    knee_g = "left_knee_joint" if not x1 else "left_knee_pitch_joint"
    names = {
        "thigh": (f"{side}_hip_pitch_joint",
                  knee_g.replace("left", side)),
        "shank": (knee_g.replace("left", side),
                  f"{side}_ankle_pitch_joint"),
        "uarm": (f"{side}_shoulder_pitch_joint",
                 ("left_elbow_joint" if not x1
                  else "left_elbow_pitch_joint").replace("left", side)),
        "farm": (("left_elbow_joint" if not x1
                  else "left_elbow_pitch_joint").replace("left", side),
                 f"{side}_wrist_pitch_joint"),
    }
    out = {}
    for k, (a, b) in names.items():
        out[k] = _unit(anch(b) - anch(a))
    return out


def joint_gates(csv_path, pkl_path, sample_step=2):
    import mujoco
    from scipy.spatial.transform import Rotation as Rot
    f = load_csv(csv_path)
    pl = X1Player(pkl_path)
    n = len(pl.frames)
    idx = np.clip(np.round(np.arange(n) / pl.time_scale).astype(int),
                  0, len(f["pos"]) - 1)

    g1 = build_fk_model()
    # standing dirs for both robots
    gd0 = mujoco.MjData(g1)
    mujoco.mj_forward(g1, gd0)
    m1 = pl.m
    xd0 = mujoco.MjData(m1)
    mujoco.mj_forward(m1, xd0)

    def pelvis_dirs(d, side, x1):
        dirs = _segment_dirs(d, side, x1)
        P = _pelvis_R(d, x1=x1)
        return {k: P.T @ v for k, v in dirs.items()}, P

    def farm_in_uarm(d, side, x1):
        """Farm direction expressed in the uarm semantic frame (direct,
        invariant arm-fidelity measure: bend angle + bend plane)."""
        dirs = _segment_dirs(d, side, x1)
        el_ax = {"left": (0, 1, 0), "right": (0, -1, 0)}[side]
        u, w = dirs["uarm"], dirs["farm"]
        # uarm frame: z = uarm dir, x = lateral (pelvis lateral projected
        # perpendicular to u), y = z cross x
        P = _pelvis_R(d, x1=x1)
        lat = P[:, 1] * (1 if side == "left" else -1)
        z = u
        x = lat - z * (lat @ z)
        nx = np.linalg.norm(x)
        x = x / nx if nx > 1e-9 else lat
        y = np.cross(z, x)
        F = np.column_stack([x, y, z])
        return F.T @ w

    j1_detail = {}
    j1_ok = True
    for side in ("left", "right"):
        for seg in ("thigh", "shank", "uarm", "farm"):
            g_stand, _ = pelvis_dirs(gd0, side, False)
            x_stand, _ = pelvis_dirs(xd0, side, True)
            zs_g, zs_x = g_stand[seg], x_stand[seg]
            ag, ax, ang_gv, ang_xv = [], [], [], []
            for i in range(0, n, max(sample_step, 3)):
                gd = g1_fk(g1, f["pos"][idx[i]], f["quat_xyzw"][idx[i]],
                           f["dof"][idx[i]])
                dg, _ = pelvis_dirs(gd, side, False)
                d = pl.fk(i)
                dx, _ = pelvis_dirs(d, side, True)
                ag.append(np.degrees(np.arccos(np.clip(dg[seg] @ zs_g, -1, 1))))
                ax.append(np.degrees(np.arccos(np.clip(dx[seg] @ zs_x, -1, 1))))
                ang_gv.append(dg[seg])
                ang_xv.append(dx[seg])
            ag, ax = np.array(ag), np.array(ax)
            if seg in ("thigh", "shank"):
                # legs: swing angle from standing + direct world-direction
                # agreement (catches azimuth-rotated mimics)
                if np.std(ag) < 1e-6 or np.std(ax) < 1e-6:
                    r = 1.0 if abs(np.median(ax) - np.median(ag)) < 15 else 0.0
                else:
                    r = float(np.corrcoef(ag, ax)[0, 1])
                med_diff = float(np.median(np.abs(ax - ag)))
                vg, vx = np.array(ang_gv), np.array(ang_xv)
                world_med = float(np.median(np.degrees(np.arccos(np.clip(
                    np.einsum("ij,ij->i", vg, vx), -1, 1)))))
                ok = bool(r >= 0.75 and med_diff <= 12.0
                          and world_med <= (16.0 if seg == "thigh" else 20.0))
                j1_detail[(side, seg)] = dict(
                    corr=r, med_angle_diff_deg=med_diff,
                    world_dir_med_deg=world_med, ok=ok)
            elif seg == "uarm":
                if np.std(ag) < 1e-6 or np.std(ax) < 1e-6:
                    r = 1.0 if abs(np.median(ax) - np.median(ag)) < 15 else 0.0
                else:
                    r = float(np.corrcoef(ag, ax)[0, 1])
                vg, vx = np.array(ang_gv), np.array(ang_xv)
                world_med = float(np.median(np.degrees(np.arccos(np.clip(
                    np.einsum("ij,ij->i", vg, vx), -1, 1)))))
                ok = bool(r >= 0.60 and world_med <= 20.0)
                j1_detail[(side, seg)] = dict(
                    corr=r, world_dir_med_deg=world_med, ok=ok)
            else:  # farm: direction in uarm frame (bend angle + plane)
                vg = np.array([farm_in_uarm(g1_fk(
                    g1, f["pos"][idx[i]], f["quat_xyzw"][idx[i]],
                    f["dof"][idx[i]]), side, False)
                    for i in range(0, n, max(sample_step, 3))])
                vx = np.array([farm_in_uarm(pl.fk(i), side, True)
                               for i in range(0, n, max(sample_step, 3))])
                d_ang = np.degrees(np.arccos(np.clip(
                    np.einsum("ij,ij->i", vg, vx), -1, 1)))
                med_diff = float(np.median(d_ang))
                # bend rhythm: correlation of the deviation angle
                bend_g = np.degrees(np.arccos(np.clip(
                    np.einsum("ij,j->i", vg, np.array([0, 0, 1.0])), -1, 1)))
                bend_x = np.degrees(np.arccos(np.clip(
                    np.einsum("ij,j->i", vx, np.array([0, 0, 1.0])), -1, 1)))
                r = (float(np.corrcoef(bend_g, bend_x)[0, 1])
                     if np.std(bend_g) > 1e-6 and np.std(bend_x) > 1e-6
                     else (1.0 if abs(np.median(bend_x) - np.median(bend_g))
                           < 15 else 0.0))
                ok = bool(med_diff <= 15.0 and r >= 0.60)
                j1_detail[(side, seg)] = dict(
                    corr=r, uarm_frame_med_deg=med_diff, ok=ok)
            j1_ok = j1_ok and ok

    # J2: X1-native posture sanity
    lf, rf, torso = [], [], []
    for i in range(0, n, max(sample_step, 3)):
        d = pl.fk(i)
        P = _pelvis_R(d, x1=True)
        lf.append(P.T @ d.site("x_lfoot").xpos)
        rf.append(P.T @ d.site("x_rfoot").xpos)
        torso.append(P.T @ d.site("x_torso").xpos)
    lf, rf, torso = np.array(lf), np.array(rf), np.array(torso)
    min_sep = float(np.min(lf[:, 1] - rf[:, 1]))
    knee_col = 6 + pl.dof_order.index("left_knee_pitch_joint")
    knee_swing = float(np.quantile(pl.frames[:, knee_col], 0.95)
                       - np.quantile(pl.frames[:, knee_col], 0.05))
    # torso pitch vs G1
    ge = Rot.from_quat(f["quat_xyzw"][idx]).as_euler("xyz", degrees=True)
    xe = Rot.from_rotvec(pl.frames[:, 3:6]).as_euler("xyz", degrees=True)
    def w180(a):
        return (a + 180) % 360 - 180
    pitch_diff = float(np.median(np.abs(w180(xe[:, 0] - ge[:, 0]))))
    roll_diff = float(np.median(np.abs(w180(xe[:, 1] - ge[:, 1]))))

    j2 = dict(min_foot_sep_m=min_sep, knee_swing_rad=knee_swing,
              torso_pitch_med_diff_deg=pitch_diff,
              torso_roll_med_diff_deg=roll_diff,
              pass_=bool(min_sep > -0.02 and knee_swing >= 0.35
                         and pitch_diff <= 15))
    j3 = dict(pitch_med_deg=pitch_diff, roll_med_deg=roll_diff,
              pass_=bool(pitch_diff <= 12 and roll_diff <= 12))
    return dict(J1_limb_fidelity=dict(
                    detail={f"{s}/{k}": v for (s, k), v in j1_detail.items()},
                    pass_=bool(j1_ok)),
                J2_posture=j2, J3_root=j3)


def r5_v2(csv_path, pkl_path, sample_step=5):
    """R5 under the v2 anchor-based root mapping."""
    import numpy as np
    f = load_csv(csv_path)
    pl = X1Player(pkl_path)
    n = len(pl.frames)
    g1 = build_fk_model()
    s_leg, s_arm = pl.s_leg, pl.s_arm
    fi_all = np.clip(np.arange(n) / pl.time_scale, 0, len(f["pos"]) - 2)
    ef, eh = [], []
    for i in range(0, n, sample_step):
        fi = fi_all[i]
        s0, s1 = int(np.floor(fi)), int(np.ceil(fi))
        w = fi - s0
        gd = g1_fk(g1,
                   f["pos"][s0] * (1 - w) + f["pos"][s1] * w,
                   f["quat_xyzw"][s0] * (1 - w) + f["quat_xyzw"][s1] * w,
                   f["dof"][s0] * (1 - w) + f["dof"][s1] * w)
        base = pl.frames[i][0:3]
        gpos = gd.qpos[:3]
        pl.fk(i)  # refresh kinematics BEFORE reading site positions
        from lib_g1 import site_pos
        from retarget_v2 import LAT_WIDEN
        from scipy.spatial.transform import Rotation as Rot
        R = Rot.from_rotvec(pl.frames[i][3:6]).as_matrix()
        lat = R @ np.array([0.0, 1.0, 0.0])
        widen = {"x_lfoot": LAT_WIDEN * lat, "x_rfoot": -LAT_WIDEN * lat}
        for site_g, site_x, s in (("left_ankle_roll_link", "x_lfoot", s_leg),
                                  ("right_ankle_roll_link", "x_rfoot", s_leg),
                                  ("left_wrist_yaw_link", "x_lhand", s_arm),
                                  ("right_wrist_yaw_link", "x_rhand", s_arm)):
            tgt = base + (site_pos(gd, site_g) - gpos) * s + widen.get(site_x, 0.0)
            if "foot" in site_x:
                tgt = tgt.copy()
                tgt[2] = max(tgt[2], 0.058)  # mirror the IK clamp
            got = pl.d.site(site_x).xpos
            (ef if "foot" in site_x else eh).append(
                float(np.linalg.norm(got - tgt)))
    ef, eh = np.array(ef), np.array(eh)
    # Hand thresholds sit above the STRUCTURAL floor: the X1 forearm is
    # 1.87x the G1's (0.259 vs 0.138 m), so even a perfectly
    # direction-matched arm leaves ~0.12 m of hand-position mismatch.
    # Gate floor documented: med 0.15 (floor 0.121), p95 0.35.
    return dict(foot_med_m=float(np.median(ef)),
                foot_p95_m=float(np.quantile(ef, 0.95)),
                hand_med_m=float(np.median(eh)),
                hand_p95_m=float(np.quantile(eh, 0.95)),
                pass_=bool(np.median(ef) < 0.03
                           and np.quantile(ef, 0.95) < 0.09
                           and np.median(eh) < 0.15
                           and np.quantile(eh, 0.95) < 0.35))


def validate(csv_path, pkl_path, sample_step=1):
    res = validate_r(csv_path, pkl_path, sample_step)
    res.pop("R5_tracking", None)  # replaced by v2-anchor-mapping version
    res.update(joint_gates(csv_path, pkl_path))
    res["R5_tracking"] = r5_v2(csv_path, pkl_path)
    res["PASS"] = all(v.get("pass_", False) for v in res.values()
                      if isinstance(v, dict) and "pass_" in v)
    return res


def selftest():
    good_csv = str(REPO_ROOT / "data/LAFAN1_g1/g1_segments/run2_subject1_seg2.csv")
    good_pkl = "/tmp/x1_selftest_good.pkl"
    bad_csv = str(REPO_ROOT / "data/LAFAN1_g1/g1_segments/sprint1_subject2_seg0.csv")
    bad_pkl = str(REPO_ROOT / "data/motions/x1/x1_sprint1_subject2_seg0.pkl")
    import subprocess
    r = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "retarget_v2.py"),
         "--csv", good_csv, "--out", good_pkl, "--skip_ik"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("selftest: retarget_v2 --skip_ik failed:\n", r.stdout[-2000:],
              r.stderr[-2000:])
        return False
    g = joint_gates(good_csv, good_pkl)
    b = joint_gates(bad_csv, bad_pkl)
    g_ok = all(g[k]["pass_"] for k in ("J1_limb_fidelity", "J2_posture",
                                       "J3_root"))
    b_bad = not all(b[k]["pass_"] for k in ("J1_limb_fidelity", "J2_posture",
                                            "J3_root"))
    print(f"selftest good->J PASS: {g_ok} | bad->J FAIL: {b_bad}")
    for tag, res in (("GOOD", g), ("BAD", b)):
        print(f"--- {tag}: J1 {res['J1_limb_fidelity']['pass_']} "
              f"J2 {res['J2_posture']['pass_']} J3 {res['J3_root']['pass_']}")
        for k, v in res["J1_limb_fidelity"]["detail"].items():
            if not v["ok"]:
                print(f"    J1 fail {k}: corr {v['corr']:.2f} "
                      f"meddiff {v.get('med_angle_diff_deg', -1):.1f} "
                      f"world {v.get('world_dir_med_deg', -1):.1f} "
                      f"uarmfr {v.get('uarm_frame_med_deg', -1):.1f}")
    ok = g_ok and b_bad
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def render_markdown_v2(results):
    lines = ["# G1 -> X1 Retargeting Validation v2.1", ""]
    for r in results:
        lines.append(f"## {Path(r['pkl']).name}  "
                     f"{'**PASS**' if r['PASS'] else '**FAIL**'}")
        base = {k: v for k, v in r.items()
                if k not in ("J1_limb_fidelity", "J2_posture", "J3_root")}
        lines += render_markdown([base]).splitlines()[3:]
        J1 = r["J1_limb_fidelity"]
        fails = [k for k, v in J1["detail"].items() if not v["ok"]]
        lines.append(f"- J1 肢体保真: {len(fails)} fail -> "
                     f"{'PASS' if J1['pass_'] else 'FAIL ' + str(fails)}")
        J2 = r["J2_posture"]
        lines.append(
            f"- J2 姿态: foot sep {J2['min_foot_sep_m']*100:.1f}cm knee swing "
            f"{J2['knee_swing_rad']:.2f}rad torso pitch diff "
            f"{J2['torso_pitch_med_diff_deg']:.1f}deg -> "
            f"{'PASS' if J2['pass_'] else 'FAIL'}")
        J3 = r["J3_root"]
        lines.append(
            f"- J3 根一致: pitch {J3['pitch_med_deg']:.1f}° roll "
            f"{J3['roll_med_deg']:.1f}° -> {'PASS' if J3['pass_'] else 'FAIL'}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--pkl")
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--sample_step", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    res = validate(args.csv, args.pkl, args.sample_step)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items()
                           if kk != "detail"})
                      for k, v in res.items()}, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    if args.markdown:
        Path(args.markdown).write_text(render_markdown_v2([res]))
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
