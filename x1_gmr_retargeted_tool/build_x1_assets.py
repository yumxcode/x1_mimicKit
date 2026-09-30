"""Build X1 29DOF assets for MimicKit (Isaac Gym MJCF + MuJoCo sim models).

Reads the original X1_29DOF mjcf (mesh-based), fits primitive collision
geometry (capsules / boxes) to mesh AABBs, applies URDF joint limits, adds
position-control PD gains (stiffness/damping) and torque-limit actuators.

Outputs:
  data/assets/x1/x1.xml       - Isaac Gym training asset (primitive geoms)
  data/assets/x1/x1_sim.xml   - MuJoCo sim2sim model (adds floor + keyframe)

Assumptions:
  * joint stiffness ~= 2.5x effort limit, damping ~= 0.1x stiffness
    (mirrors MimicKit humanoid.xml kp/effort ratios)
  * URDF limits are the authoritative hardware limits (tighter than mjcf)
  * collision geometry = capsule per long link + box per foot sole
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_MJCF = REPO_ROOT / "X1_29DOF/mjcf/xyber_x1_flat.xml"
SRC_URDF = REPO_ROOT / "X1_29DOF/urdf/f1.urdf"

# ---------------------------------------------------------------- URDF limits
def parse_urdf_limits():
    import re
    xml = SRC_URDF.read_text()
    pat = re.compile(
        r'<joint name="([^"]+)" type="revolute">.*?'
        r'<limit lower="([^"]+)" upper="([^"]+)" effort="([^"]+)" velocity="([^"]+)"',
        re.S)
    out = {}
    for name, lo, hi, eff, vel in pat.findall(xml):
        out[name] = dict(low=float(lo), high=float(hi),
                         effort=float(eff), velocity=float(vel))
    return out


# ------------------------------------------------------------ PD gain policy
def pd_gains(effort):
    """kp ~ 2.5x torque limit (cap for weak joints), kd = 0.1 kp."""
    kp = min(2.5 * effort, 500.0)
    if effort <= 10:      # wrists: weak, low gains
        kp = min(kp, 25.0)
    elif effort <= 20:    # arms
        kp = min(kp, 50.0)
    kp = float(round(kp))
    kd = float(round(0.1 * kp, 2))
    return kp, kd


ARMATURE = {"lumbar": 0.02, "arm": 0.01, "leg": 0.02, "wrist": 0.005}

def joint_class(name):
    if "wrist" in name:
        return "wrist"
    if "lumbar" in name:
        return "lumbar"
    if any(k in name for k in ("shoulder", "elbow")):
        return "arm"
    return "leg"


# --------------------------------------------------------------- mesh fitting
def fit_capsule(verts):
    """Fit capsule: axis = principal component, radius covers points."""
    c = verts.mean(axis=0)
    u, s, vt = np.linalg.svd(verts - c, full_matrices=False)
    axis = vt[0]
    t = (verts - c) @ axis
    half = max(np.abs(t).max(), 0.01)
    perp = (verts - c) - np.outer(t, axis)
    radius = float(max(np.linalg.norm(perp, axis=1).max(), 0.015))
    p0 = c - axis * half
    p1 = c + axis * half
    return p0, p1, radius


# Hand-tuned collision set: slim capsules along the kinematic chain.
# Maps link name -> (child joint direction end point in link frame [m],
#                     radius [m]). These follow humanoid.xml conventions:
# slim capsules that approximate the limb without the motor housings.
# link name -> capsule radius [m]; endpoints derived link-origin -> child-body
HAND_COLLISION_R = {
    "base_link": 0.066,                                            # pelvis
    "left_hip_yaw_link": 0.058, "right_hip_yaw_link": 0.058,      # thigh
    "left_knee_pitch_link": 0.048, "right_knee_pitch_link": 0.048,  # shin
    "lumbar_pitch_link": 0.100,                                    # torso
    "left_shoulder_yaw_link": 0.042, "right_shoulder_yaw_link": 0.042,  # upper arm
    "left_elbow_yaw_link": 0.034, "right_elbow_yaw_link": 0.034,  # forearm
}

# meshes are dropped from the emitted assets entirely (their PCA capsules
# are far too fat for collision; Isaac Gym ignores contype in some paths).
USE_MESH_GEOMS = False


def geom_local_verts(model, geom_id):
    """Mesh vertices transformed into geom local frame (before geom pos/quat)."""
    mid = model.geom_dataid[geom_id]
    vadr = model.mesh_vertadr[mid]
    vcnt = model.mesh_vertnum[mid]
    verts = model.mesh_vert[vadr:vadr + vcnt].astype(np.float64)
    return verts


def geom_body_verts(model, geom_id):
    """Mesh vertices in the geom's body frame (applies geom pos + quat)."""
    from scipy.spatial.transform import Rotation as Rot
    verts = geom_local_verts(model, geom_id)
    pos = model.geom_pos[geom_id].astype(np.float64)
    quat = model.geom_quat[geom_id].astype(np.float64)  # wxyz
    rot = Rot.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()  # xyzw
    return verts @ rot.T + pos


# --------------------------------------------------------------------- build
def build(tolerance=0.0):
    import mujoco

    urdf_lim = parse_urdf_limits()
    assert len(urdf_lim) == 29, f"expected 29 revolute joints, got {len(urdf_lim)}"

    src = mujoco.MjModel.from_xml_path(str(SRC_MJCF))

    # ET cannot resolve <include>; parse the robot file directly for the tree
    serial = SRC_MJCF.parent / "robot/xyber_x1/xyber_x1_serial.xml"
    src_tree = ET.parse(serial)
    src_root = src_tree.getroot()
    wb = src_root.find("worldbody")
    src_body_root = wb.find("body")

    # collect mesh AABB per body from source model (keyed by body name)
    body_names = [src.body(i).name for i in range(src.nbody)]
    body_meshes = {b: [] for b in body_names}
    body_spheres = {b: [] for b in body_names}
    for g in range(src.ngeom):
        b = src.body(src.geom_bodyid[g]).name
        gname = src.geom(g).name or ""
        if src.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            body_meshes[b].append(g)
        elif src.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and "collision" in gname:
            pass  # foot corner spheres handled separately below
        elif src.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            body_spheres[b].append(g)

    def fmt(arr):
        return " ".join(f"{v:.6g}" for v in np.asarray(arr).ravel())

    def emit_body(src_body, parent_el):
        name = src_body.attrib["name"]
        body_el = ET.SubElement(parent_el, "body", name=name)
        for k in ("pos", "quat", "gravcomp"):
            if k in src_body.attrib:
                body_el.set(k, src_body.attrib[k])

        # explicit inertial (copy)
        inert = src_body.find("inertial")
        if inert is not None:
            body_el.append(ET.fromstring(ET.tostring(inert)))

        # freejoint on root
        if name == "base_link":
            ET.SubElement(body_el, "freejoint", name="floating_base")
            # pelvis shell capsule fitted from base mesh
        else:
            for jx in src_body.findall("joint"):
                jname = jx.attrib["name"]
                lim = urdf_lim[jname]
                kp, kd = pd_gains(lim["effort"])
                je = ET.SubElement(body_el, "joint", {
                    "name": jname, "type": "hinge",
                    "limited": "true",  # Isaac Gym MJCF parser needs explicit
                    "pos": jx.attrib.get("pos", "0 0 0"),
                    "axis": jx.attrib["axis"],
                    "range": f"{lim['low']:.4f} {lim['high']:.4f}",
                    "stiffness": str(kp), "damping": str(kd),
                    "armature": str(ARMATURE[joint_class(jname)]),
                    "frictionloss": jx.attrib.get("frictionloss", "0"),
                })

        # collision / visual primitives from meshes
        geom_names = []
        if USE_MESH_GEOMS:
            for g in body_meshes.get(name, []):
                verts = geom_body_verts(src, g)
                p0, p1, r = fit_capsule(verts)
                gname = src.geom(g).name or f"geom_{g}"
                ET.SubElement(body_el, "geom", {
                    "name": gname, "type": "capsule",
                    "fromto": fmt(np.concatenate([p0, p1])),
                    "size": f"{r:.4f}", "density": "0",
                    "contype": "0", "conaffinity": "0",
                    "friction": "1.0 0.05 0.05", "condim": "4",
                    "rgba": "0.8 0.4 0 1", "group": "1",
                })
                geom_names.append(gname)

        # slim hand-tuned collision capsule: link origin -> first child body
        if name in HAND_COLLISION_R:
            child = src_body.findall("body")
            end = None
            if child:
                cpos = child[0].attrib.get("pos", "0 0 0")
                end = np.fromstring(cpos, dtype=float, sep=" ")
            if name == "lumbar_pitch_link" and len(child) > 1:
                # torso capsule should span up to the shoulders, not the arm
                cpos = child[1].attrib.get("pos", "0 0 0")
                end2 = np.fromstring(cpos, dtype=float, sep=" ")
                end = 0.5 * (end + end2) + np.array([0, 0, 0.02])
            if name == "base_link":
                # pelvis capsule: base origin -> midpoint of the two hips
                pts = [np.fromstring(c.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
                       for c in child if "hip_pitch" in c.attrib.get("name", "")]
                if len(pts) == 2:
                    end = 0.5 * (pts[0] + pts[1])
                    end[2] = min(end[2], -0.03)
            r = HAND_COLLISION_R[name]
            start = np.zeros(3)
            if name == "lumbar_pitch_link":
                start = np.array([0.0, 0.0, 0.04])  # keep clear of pelvis
            ET.SubElement(body_el, "geom", {
                "name": f"{name}_col", "type": "capsule",
                "fromto": fmt(np.concatenate([start, end])),
                "size": f"{r}", "density": "0",
                "contype": "1", "conaffinity": "1",
                "friction": "1.0 0.05 0.05", "condim": "4",
                "rgba": "0.9 0.5 0.1 1", "group": "1",
            })

        # foot sole box (ankle_roll links). CRITICAL: the L/R ankle_roll body
        # frames are MIRRORED (left_ankle_pitch body quat 180deg-about-y vs
        # right 180deg-about-x, both ankle_roll bodies +90deg-about-y), so
        # the sole plate lies at local y=-0.0408 on the LEFT foot but at
        # local y=+0.0408 on the RIGHT foot (measured from the STL meshes in
        # body frame: sole x extent +-0.056, z extent [-0.093,+0.103]).
        # v1 bug: both boxes at pos y=-0.043 -> right box sat on the dorsum
        # (+7.2 cm above the true sole), left box underhung the sole; the
        # box also under-covered the foot (half 0.032x0.072 vs true
        # 0.056x0.098). v3: per-side y sign, bottom face 1.5 mm proud of the
        # mesh sole plane (|y|=0.0423), full-length/full-width footprint.
        if name.endswith("ankle_roll_link"):
            y_sign = -1.0 if name.startswith("left") else +1.0
            ET.SubElement(body_el, "geom", {
                "name": f"{name}_sole", "type": "box",
                "pos": f"0 {y_sign * 0.0303:.4f} 0.005",
                "size": "0.055 0.012 0.098", "density": "0",
                "contype": "1", "conaffinity": "1",
                "friction": "1.0 0.05 0.05", "condim": "4",
                "rgba": "1 0.5 1 1", "group": "1",
            })

        for child in src_body.findall("body"):
            emit_body(child, body_el)

        # key sites for retargeting / metrics
        X1_SITES = {
            "base_link": "x_base",
            "lumbar_pitch_link": "x_torso",
            "left_ankle_roll_link": "x_lfoot",
            "right_ankle_roll_link": "x_rfoot",
            "left_wrist_roll_link": "x_lhand",
            "right_wrist_roll_link": "x_rhand",
        }
        if name in X1_SITES:
            ET.SubElement(body_el, "site", name=X1_SITES[name], size="0.005")
        return body_el

    out_root = ET.Element("mujoco", model="x1")
    ET.SubElement(out_root, "compiler", angle="radian")
    dflt = ET.SubElement(out_root, "default")
    ET.SubElement(dflt, "motor", ctrlrange="-1 1", ctrllimited="true")
    world = ET.SubElement(out_root, "worldbody")

    emit_body(src_body_root, world)
    # MimicKit convention: root body at origin (motion provides world pose)
    world.find("body").set("pos", "0 0 0")

    # ---- self-collision excludes for close kinematic pairs (tree dist <= 2)
    # (must be inserted before <actuator> per MJCF element order)
    body_parent = {"base_link": None}
    for b in src_tree.iter("body"):
        for ch in b.findall("body"):
            body_parent[ch.attrib["name"]] = b.attrib["name"]

    def tree_dist(a, b):
        pa = {a}
        p = a
        while body_parent.get(p):
            p = body_parent[p]
            pa.add(p)
        pb = {b}
        p = b
        while body_parent.get(p):
            p = body_parent[p]
            pb.add(p)
        # distance = up-steps from a to common ancestor + from b
        da, p = 0, a
        while p not in pb:
            p = body_parent[p]
            da += 1
        db, p = 0, b
        while p not in pa:
            p = body_parent[p]
            db += 1
        return da + db

    body_list = [b.attrib["name"] for b in src_tree.iter("body")]
    contact = ET.Element("contact")
    n_excl = 0
    for i in range(len(body_list)):
        for j in range(i + 1, len(body_list)):
            a, b = body_list[i], body_list[j]
            if tree_dist(a, b) <= 2:
                ET.SubElement(contact, "exclude", body1=a, body2=b)
                n_excl += 1
    print(f"[build_x1_assets] self-collision excludes: {n_excl} pairs")

    # actuators: motor gear = effort limit (Isaac Gym reads this as torque lim)
    act = ET.SubElement(out_root, "actuator")
    order = []
    for b in src_tree.iter("body"):
        for jx in b.findall("joint"):
            order.append(jx.attrib["name"])
    for jname in order:
        lim = urdf_lim[jname]
        ET.SubElement(act, "motor", name=f"motor_{jname}",
                      joint=jname, gear=str(lim["effort"]))
    # insert contact element before actuator (MJCF ordering)
    out_root.insert(list(out_root).index(act), contact)

    # pretty print
    ET.indent(out_root, space="  ")
    xml_str = ET.tostring(out_root, encoding="unicode")

    out_dir = REPO_ROOT / "data/assets/x1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "x1.xml").write_text(xml_str + "\n")

    # ---- sim2sim variant: adds floor, timestep, keyframe (MuJoCo native)
    sim_root = ET.fromstring(xml_str)
    sim_root.set("model", "x1_sim")
    sim_root.insert(0, ET.Element("option", timestep="0.005"))
    # MuJoCo sim2sim policy: explicit PD torque in the player loop.
    # Passive springs would fight the policy targets -> remove them.
    for j in sim_root.iter("joint"):
        j.set("stiffness", "0")
        j.set("damping", "0")
    # actuators take raw torque: gear=1, ctrlrange = +-effort
    for motor in sim_root.iter("motor"):
        gear = motor.attrib.get("gear", "1")
        motor.set("ctrlrange", f"-{gear} {gear}")
        motor.set("gear", "1")
    # match Isaac training (col_filter=1): disable ALL robot self-collision
    sim_contact = sim_root.find("contact")
    if sim_contact is None:
        sim_contact = ET.Element("contact")
    body_set = [b.attrib["name"] for b in sim_root.iter("body")]
    for i in range(len(body_set)):
        for j2 in range(i + 1, len(body_set)):
            ET.SubElement(sim_contact, "exclude", body1=body_set[i],
                          body2=body_set[j2])
    if sim_contact not in list(sim_root):
        sim_root.insert(list(sim_root).index(sim_root.find("actuator")), sim_contact)
    wb = sim_root.find("worldbody")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 3 0.125",
                  contype="1", conaffinity="1", friction="1.0 0.05 0.05",
                  condim="4")
    ET.SubElement(wb, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.indent(sim_root, space="  ")
    (out_dir / "x1_sim.xml").write_text(ET.tostring(sim_root, encoding="unicode") + "\n")

    # compute standing height: FK with home joints (flat feet), sole at z=0
    m = mujoco.MjModel.from_xml_path(str(out_dir / "x1_sim.xml"))
    d = mujoco.MjData(m)
    home = [0.0] * 29
    home[17:23] = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home[23:29] = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]
    d.qpos[7:] = home
    d.qpos[2] = 1.0
    mujoco.mj_forward(m, d)
    sole_z = np.inf
    for g in range(m.ngeom):
        if (m.geom(g).name or "").endswith("_sole"):
            R = d.geom_xmat[g].reshape(3, 3)
            half = m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        c = d.geom_xpos[g] + R @ np.array(
                            [sx * half[0], sy * half[1], sz * half[2]])
                        sole_z = min(sole_z, c[2])
    stand_z = 1.0 - sole_z
    kf_qpos = "0 0 {:.4f} 1 0 0 0 ".format(stand_z) + " ".join(f"{v:g}" for v in home)
    key = ET.Element("keyframe")
    ET.SubElement(key, "key", name="home", qpos=kf_qpos)
    sim_root.append(key)
    ET.indent(sim_root, space="  ")
    (out_dir / "x1_sim.xml").write_text(ET.tostring(sim_root, encoding="unicode") + "\n")

    # validate both with mujoco
    for f in ("x1.xml", "x1_sim.xml"):
        m = mujoco.MjModel.from_xml_path(str(out_dir / f))
        print(f"[build_x1_assets] {f}: nq={m.nq} nv={m.nv} nu={m.nu} "
              f"nbody={m.nbody} mass={m.body_mass.sum():.2f} kg")

    m = mujoco.MjModel.from_xml_path(str(out_dir / "x1_sim.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"[build_x1_assets] standing base z (settled) = {stand_z:.4f} m")
    m2 = mujoco.MjModel.from_xml_path(str(out_dir / "x1.xml"))
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    print(f"[build_x1_assets] x1.xml root qpos0 z = {d2.qpos[2]:.4f} (expect 0)")
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.0)
    args = ap.parse_args()
    build(args.tolerance)
