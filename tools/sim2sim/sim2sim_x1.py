"""MuJoCo sim2sim for the X1 29DOF SMP jogging policy.

Replicates MimicKit's observation/action pipeline around a MuJoCo torque
actuated model:

  obs  = compute_char_obs(global_obs=True, root_height_obs=True, key_bodies)
         [root_h(1), root_rot tan-norm(6), root_vel(3), root_ang_vel(3),
          joint_rot tan-norm(29*6), dof_vel(29), key_pos - root_pos(4*3)]
  act  = clip(a_norm.unnormalize(actor(obs_norm(obs))), bounds)  -> joint
         position targets; PD torque = kp*(qdes-q) - kd*qd, clipped to URDF
         effort limits (motor ctrlrange in x1_sim.xml).

Sim: timestep 1/120, 4 substeps per 30 Hz control step (matches the Isaac
engine config sim_freq=120 / control_freq=30).

Pass gates (S1-S6, aligned with the humanoid_pose_standard skill and the
retarget v3 gates):
  S1 no-fall     : no non-foot ground contact, base z >= 0.35 m, and
                   |roll|,|pitch| <= 60 deg for the whole run
  S2 rhythm      : cadence ratio vs source pkl in [0.8, 1.25]; step
                   symmetry index SI >= 0.75
  S3 no-penetration: min sole-corner z (8 corners/foot) >= -8 mm
  S4 flat stance : sole tilt med <= 10 deg, p90 <= 20 deg over planted
                   frames (planted = min corner < 8 mm and center < 55 mm)
  S5 smoothness  : max |qdot| <= 1.1x URDF velocity limit; hip_roll/yaw
                   2nd-diff p99 <= 8 deg/frame^2
  S6 locomotion  : mean horizontal speed in [1.0, 2.6] m/s; backward
                   frames <= 5%

Usage:
  python tools/sim2sim/sim2sim_x1.py --model <policy.pt> \
      [--motion data/motions/x1_v3/x1_run1_subject5_seg0.pkl] \
      [--duration 30] [--video out.mp4] [--json out.json]
"""
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

X1_ASSET = REPO_ROOT / "data/assets/x1/x1.xml"        # PD gains + limits
X1_SIM = REPO_ROOT / "data/assets/x1/x1_sim.xml"      # sim2sim model

FPS = 30.0
SIM_DT = 1.0 / 120.0
SUBSTEPS = 4

KEY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link",
              "left_wrist_roll_link", "right_wrist_roll_link"]

# ------------------------------------------------------------------ helpers
def quat_mul(a, b):
    """Hamilton product, (N,4) wxyz."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def quat_to_tan_norm(q):
    """MimicKit torch_util.quat_to_tan_norm: 6D rep [tangent, normal] where
    tangent = q * x-axis and normal = q * z-axis."""
    x, y, z, w = q[..., 1], q[..., 2], q[..., 3], q[..., 0]
    tx = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                   2 * (x * z - w * y)], axis=-1)
    tz = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                   1 - 2 * (x * x + y * y)], axis=-1)
    return np.concatenate([tx, tz], axis=-1)


def axis_angle_to_quat(axis, angle):
    half = 0.5 * angle
    return np.array([math.cos(half), *(math.sin(half) * np.asarray(axis))])


def load_policy(model_path, device="cpu"):
    """Rebuild actor MLP + normalizers from a MimicKit agent state_dict."""
    import torch

    sd = torch.load(model_path, map_location=device)
    keys = list(sd.keys())
    actor_keys = sorted(k for k in keys if k.startswith("_model._actor_layers"))
    a_norm_prefix = "_a_norm."
    obs_norm_prefix = "_obs_norm."

    out = {"raw": sd}

    def get(pfx, name):
        return sd[pfx + name].numpy().astype(np.float64)

    out["obs_mean"] = get(obs_norm_prefix, "_mean")
    out["obs_std"] = get(obs_norm_prefix, "_std")
    out["a_mean"] = get(a_norm_prefix, "_mean")
    out["a_std"] = get(a_norm_prefix, "_std")

    # rebuild MLP: Linear layers named _model._actor_layers.{i}.weight/bias
    layers = {}
    for k in actor_keys:
        idx = int(k.split(".")[2])
        kind = k.split(".")[-1]
        layers.setdefault(idx, {})[kind] = sd[k].cpu().numpy()
    weight_keys = sorted(layers.keys())
    ops = []
    for i in weight_keys:
        W, b = layers[i]["weight"], layers[i]["bias"]
        ops.append((W, b))
    out["actor_ops"] = ops
    return out


def actor_forward(policy, obs):
    h = obs
    n = len(policy["actor_ops"])
    for i, (W, b) in enumerate(policy["actor_ops"]):
        h = h @ W.T + b
        if i < n - 1:
            h = np.maximum(h, 0.0)  # ReLU (MimicKit base_model default)
    return h


def normalize(policy, obs, which="obs"):
    m, s = policy[f"{which}_mean"], policy[f"{which}_std"]
    return np.clip((obs - m) / s, -10.0, 10.0)


def unnormalize(policy, x, which="a"):
    m, s = policy[f"{which}_mean"], policy[f"{which}_std"]
    return x * s + m


# ------------------------------------------------------------------ model io
def parse_x1_xml():
    """Joint order/axis/kp/kd/limits + action bounds from x1.xml (training
    asset; the tree matches x1_sim.xml)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(X1_ASSET).getroot()
    joints = []
    body = root.find("worldbody").find("body")  # base_link
    def walk(b):
        for j in b.findall("joint"):
            joints.append(dict(
                name=j.attrib["name"],
                axis=np.fromstring(j.attrib.get("axis", "0 0 1"), sep=" "),
                range=np.fromstring(j.attrib.get("range", "-3.15 3.15"),
                                    sep=" "),
                kp=float(j.attrib.get("stiffness", 0)),
                kd=float(j.attrib.get("damping", 0)),
            ))
        for c in b.findall("body"):
            walk(c)
    walk(body)

    efforts = {}
    for m in root.find("actuator").findall("motor"):
        efforts[m.attrib["joint"]] = float(m.attrib["gear"])

    low = np.array([j["range"][0] for j in joints])
    high = np.array([j["range"][1] for j in joints])
    mid = 0.5 * (high + low)
    scale = np.maximum(np.abs(high - mid), np.abs(low - mid)) * 1.4
    return dict(
        joints=joints,
        names=[j["name"] for j in joints],
        kp=np.array([j["kp"] for j in joints]),
        kd=np.array([j["kd"] for j in joints]),
        effort=np.array([efforts[j["name"]] for j in joints]),
        vel_lim=np.array(VELOCITY_LIMITS),
        a_low=mid - scale, a_high=mid + scale,
    )


# URDF velocity limits (f1.urdf) in X1_DOF_ORDER
VELOCITY_LIMITS = [  # rad/s, filled by parse_urdf when available
]


def parse_urdf_velocity():
    urdf = REPO_ROOT / "X1_29DOF/urdf/f1.urdf"
    import re
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf)
    out = {}
    for j in tree.iter("joint"):
        if j.attrib.get("type") != "revolute":
            continue
        lim = j.find("limit")
        out[j.attrib["name"]] = float(lim.attrib["velocity"])
    return out


# ------------------------------------------------------------------ sim loop
class X1Sim:
    def __init__(self, spec):
        import mujoco
        self.spec = spec
        self.m = mujoco.MjModel.from_xml_path(str(X1_SIM))
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = SIM_DT
        self.qadr = np.array([self.m.joint(n).qposadr[0]
                              for n in spec["names"]])
        self.vadr = np.array([self.m.joint(n).dofadr[0]
                              for n in spec["names"]])
        self.base_bid = self.m.body("base_link").id
        self.key_bids = [self.m.body(n).id for n in KEY_BODIES]
        self.sole_gids = {s: self.m.geom(f"{s}_ankle_roll_link_sole").id
                          for s in ("left", "right")}
        self.foot_bids = {s: self.m.body(f"{s}_ankle_roll_link").id
                          for s in ("left", "right")}

    def set_init(self, frame):
        import mujoco
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = frame[0:3]
        exp = frame[3:6]
        ang = np.linalg.norm(exp)
        if ang < 1e-8:
            q = np.array([1.0, 0, 0, 0])
        else:
            axis = exp / ang
            q = axis_angle_to_quat(axis, ang)
        self.d.qpos[3:7] = q  # wxyz freejoint
        self.d.qpos[self.qadr] = frame[6:6 + 29]
        mujoco.mj_forward(self.m, self.d)

    def state(self):
        d, m = self.d, self.m
        root_pos = d.qpos[0:3].copy()
        root_rot = d.qpos[3:7].copy()  # wxyz
        # world-frame root velocities (Isaac Gym convention)
        root_vel = d.qvel[0:3].copy()
        # ang vel of base in world frame
        R = d.xmat[self.base_bid].reshape(3, 3)
        root_ang_vel = R @ d.qvel[3:6].copy()
        dof_pos = d.qpos[self.qadr].copy()
        dof_vel = d.qvel[self.vadr].copy()
        body_pos = np.array([d.xpos[b] for b in self.key_bids])
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos

    def build_obs(self):
        (root_pos, root_rot, root_vel, root_ang_vel,
         dof_pos, dof_vel, key_body_pos) = self.state()
        joint_quats = np.array([
            axis_angle_to_quat(j["axis"], dof_pos[i])
            for i, j in enumerate(self.spec["joints"])])
        joint_tn = np.concatenate(
            [quat_to_tan_norm(q) for q in joint_quats], axis=-1)
        key_pos = (key_body_pos - root_pos[None, :]).reshape(-1)
        obs = np.concatenate([
            [root_pos[2]],
            quat_to_tan_norm(root_rot),
            root_vel,
            root_ang_vel,
            joint_tn,
            dof_vel,
            key_pos,
        ])
        return obs

    def apply_action(self, a):
        a = np.clip(a, self.spec["a_low"], self.spec["a_high"])
        d = self.d
        q = d.qpos[self.qadr]
        qd = d.qvel[self.vadr]
        tau = self.spec["kp"] * (a - q) - self.spec["kd"] * qd
        tau = np.clip(tau, -self.spec["effort"], self.spec["effort"])
        d.ctrl[:] = tau

    def step_sim(self):
        import mujoco
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)

    def sole_geometry(self):
        out = {}
        d = self.d
        for s in ("left", "right"):
            g = self.sole_gids[s]
            R = d.geom_xmat[g].reshape(3, 3)
            h = self.m.geom_size[g]
            c = d.geom_xpos[g]
            corners = [c + R @ np.array([sx * h[0], sy * h[1], sz * h[2]])
                       for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            zmin = min(cc[2] for cc in corners)
            yAxis = R @ np.array([0.0, 1.0, 0.0])
            tilt = math.degrees(math.acos(min(1.0, abs(yAxis[2]))))
            out[s] = dict(zmin=zmin, zcen=c[2], tilt=tilt)
        return out


# ------------------------------------------------------------------ gates
def detect_contacts(z, h=0.04):
    z = np.asarray(z)
    base_level = np.quantile(z, 0.15)
    return z < base_level + h


def strikes(contacts):
    idx = np.where(np.diff(contacts.astype(int)) == 1)[0] + 1
    if contacts[0]:
        idx = np.r_[0, idx]
    return idx


def cycle_period(strike_idx, fps):
    if len(strike_idx) < 3:
        return float("nan")
    return float(np.median(np.diff(strike_idx)) / fps)


def run_sim2sim(args):
    import mujoco

    vel_urdf = parse_urdf_velocity()
    global VELOCITY_LIMITS
    spec = parse_x1_xml()
    VELOCITY_LIMITS = [vel_urdf[n] for n in spec["names"]]
    spec["vel_lim"] = np.array(VELOCITY_LIMITS)

    policy = load_policy(args.model)

    with open(args.motion, "rb") as f:
        mot = pickle.load(f)
    frames = np.array(mot["frames"])

    sim = X1Sim(spec)
    sim.set_init(frames[0])

    n_steps = int(args.duration * FPS)
    log = dict(t=[], root_z=[], root_pos=[], sole=[], dof_pos=[],
               tau=[], pitch=[], roll=[])

    video_frames = []
    renderer = None
    if args.video:
        renderer = mujoco.Renderer(sim.m, height=480, width=640)

    fallen = False
    foot_bodies = {sim.foot_bids["left"], sim.foot_bids["right"]}
    # ankle chain bodies are part of the feet (ankle origin sits ~5 cm up)
    ankle_bodies = set()
    for s in ("left", "right"):
        ankle_bodies.add(sim.m.body(f"{s}_ankle_roll_link").id)
        ankle_bodies.add(sim.m.body(f"{s}_ankle_pitch_link").id)
    for step in range(n_steps):
        obs = sim.build_obs()
        norm_obs = normalize(policy, obs, "obs")
        norm_a = actor_forward(policy, norm_obs)
        a = unnormalize(policy, norm_a, "a")
        sim.apply_action(a)

        for _ in range(SUBSTEPS):
            sim.step_sim()

        root_pos, root_rot, _, _, dof_pos, dof_vel, _ = sim.state()
        sole = sim.sole_geometry()
        w = root_rot
        pitch = math.degrees(math.atan2(2 * (w[0] * w[2] - w[1] * w[3]),
                                        1 - 2 * (w[2] ** 2 + w[3] ** 2)))
        roll = math.degrees(math.atan2(2 * (w[0] * w[1] + w[2] * w[3]),
                                       1 - 2 * (w[1] ** 2 + w[3] ** 2)))
        log["t"].append(step / FPS)
        log["root_z"].append(root_pos[2])
        log["root_pos"].append(root_pos.copy())
        log["sole"].append(sole)
        log["dof_pos"].append(dof_pos.copy())
        log["tau"].append(sim.d.ctrl.copy())
        log["pitch"].append(pitch)
        log["roll"].append(roll)

        # S1: non-foot bodies touching ground (body origin low)
        nonfoot_contact = False
        for bid in range(sim.m.nbody):
            name = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, bid)
            if (not name or name == "world" or bid in foot_bodies
                    or bid in ankle_bodies or bid == sim.base_bid):
                continue
            if sim.d.xpos[bid][2] < 0.05:
                nonfoot_contact = True
                break
        if (nonfoot_contact or root_pos[2] < 0.35
                or abs(pitch) > 60 or abs(roll) > 60):
            fallen = True
            print(f"[sim2sim] FALLEN at t={step/FPS:.2f}s "
                  f"(z={root_pos[2]:.3f} pitch={pitch:.0f} roll={roll:.0f} "
                  f"nonfoot={nonfoot_contact})")
            break

        if renderer is not None:
            mujoco.mj_forward(sim.m, sim.d)
            cam = renderer.camera
            cam.lookat[:] = [root_pos[0] + 1.0, 0.0, 0.7]
            cam.distance = 4.5
            cam.azimuth = 90
            cam.elevation = -10
            renderer.update_scene(sim.d)
            video_frames.append(renderer.render())

    # ---------------- metrics & gates
    T = np.array(log["t"])
    Z = np.array(log["root_z"])
    P = np.array(log["root_pos"])
    S = log["sole"]
    Q = np.array(log["dof_pos"])
    TAU = np.array(log["tau"])

    # S2 rhythm from foot center heights
    lz = np.array([s["left"]["zcen"] for s in S])
    rz = np.array([s["right"]["zcen"] for s in S])
    lc = detect_contacts(lz)
    rc = detect_contacts(rz)
    ls, rs = strikes(lc), strikes(rc)
    cad = cycle_period(ls, FPS)
    all_strikes = np.sort(np.concatenate([ls, rs]))
    ls_set = set(ls.tolist())
    sides = np.array([0 if s in ls_set else 1 for s in all_strikes])
    if len(sides) >= 4:
        alternation = float(np.mean(sides[:-1] != sides[1:]))
        side_counts = np.bincount(sides, minlength=2)
        si = alternation * (1.0 - abs(int(side_counts[0]) - int(side_counts[1]))
                            / max(1, len(sides)))
    else:
        si = 0.0

    src_cad = args.src_cadence
    cad_ratio = (1.0 / cad / src_cad if src_cad else float("nan"))

    # S3 penetration
    zmin_all = min(min(s[s2]["zmin"] for s2 in ("left", "right")) for s in S)

    # S4 stance tilt
    all_tilts = []
    for s in S:
        for s2 in ("left", "right"):
            if s[s2]["zmin"] < 0.008 and s[s2]["zcen"] < 0.055:
                all_tilts.append(s[s2]["tilt"])
    if all_tilts:
        tilt_med = float(np.median(all_tilts))
        tilt_p90 = float(np.quantile(all_tilts, 0.9))
    else:
        tilt_med = tilt_p90 = float("nan")

    # S5 smoothness
    if len(Q) > 3:
        d2 = np.abs(np.diff(Q, 2, axis=0)) * 180 / math.pi
        hip_idx = [i for i, n in enumerate(spec["names"])
                   if "hip_roll" in n or "hip_yaw" in n]
        jerk_p99 = float(np.quantile(d2[:, hip_idx], 0.99))
        qdot = np.abs(np.diff(Q, axis=0)) * FPS
        vel_ratio = float((qdot / spec["vel_lim"]).max())
    else:
        jerk_p99 = vel_ratio = float("nan")

    # S6 locomotion
    if len(P) > 2:
        v_xy = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) * FPS
        v_fwd = np.diff(P[:, 0]) * FPS
        mean_speed = float(np.mean(v_xy))
        backward_frac = float(np.mean(v_fwd < 0))
    else:
        mean_speed = backward_frac = float("nan")

    gates = {
        "S1_no_fall": dict(pass_=bool(not fallen),
                           duration_s=float(T[-1]) if len(T) else 0.0,
                           fell_at_s=None if not fallen else float(T[-1])),
        "S2_rhythm": dict(
            pass_=bool(si >= 0.75
                       and (np.isnan(cad_ratio) or 0.8 <= cad_ratio <= 1.25)),
            cadence_hz=float(cad) if cad == cad else None,
            cad_ratio=None if cad != cad else cad_ratio,
            symmetry=float(si),
            n_steps_l=int(len(ls)), n_steps_r=int(len(rs))),
        "S3_no_penetration": dict(pass_=bool(zmin_all >= -0.008),
                                  min_sole_z_mm=zmin_all * 1000),
        "S4_flat_stance": dict(
            pass_=bool(tilt_med == tilt_med and tilt_med <= 10.0
                       and tilt_p90 <= 20.0),
            tilt_med_deg=None if tilt_med != tilt_med else tilt_med,
            tilt_p90_deg=None if tilt_p90 != tilt_p90 else tilt_p90),
        "S5_smooth": dict(
            pass_=bool(vel_ratio == vel_ratio and vel_ratio <= 1.1
                       and jerk_p99 <= 8.0),
            vel_ratio=vel_ratio, hip_jerk_p99_deg=jerk_p99),
        "S6_locomotion": dict(
            pass_=bool(mean_speed == mean_speed and 1.0 <= mean_speed <= 2.6
                       and backward_frac <= 0.05),
            mean_speed=mean_speed, backward_frac=backward_frac),
    }
    overall = all(g["pass_"] for g in gates.values())

    if renderer is not None and video_frames:
        import imageio.v2 as imageio
        imageio.mimwrite(args.video, video_frames, fps=int(FPS), quality=8)

    report = dict(
        model=args.model, motion=args.motion,
        duration_s=float(T[-1]) if len(T) else 0.0,
        fell=fallen, gates=gates, overall_pass=overall,
        mean_abs_torque=float(np.abs(TAU).mean()) if len(TAU) else None,
        mean_root_z=float(np.mean(Z)) if len(Z) else None,
    )
    print(json.dumps(report, indent=2, default=str))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
    return 0 if overall else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--motion",
                    default="data/motions/x1_v3/x1_run1_subject5_seg0.pkl")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--src-cadence", type=float, default=None,
                    help="source clip step cadence in Hz (from R1 report)")
    args = ap.parse_args()
    sys.exit(run_sim2sim(args))


if __name__ == "__main__":
    main()
