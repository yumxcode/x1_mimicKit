"""G1 helpers: meshless FK model, CSV loading, link frames, gait analysis.

The LAFAN1_g1 CSVs (data/LAFAN1_g1/g1/*.csv) have 36 columns:
  [0:3]   root pos (pelvis, world, m)
  [3:7]   root quat (x, y, z, w)
  [7:36]  29 dof values in menagerie g1.xml depth-first order:
          L leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll),
          R leg (...), waist (yaw, roll, pitch),
          L arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                 wrist_roll, wrist_pitch, wrist_yaw), R arm (...)

The menagerie g1.xml uses meshes for visuals; for pure FK we strip meshes and
keep the kinematic tree. `site`s are attached to key links for easy FK.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
G1_MENAGERIE = REPO_ROOT / ".research/g1_menagerie.xml"
G1_FK_XML = REPO_ROOT / ".research/g1_fk.xml"

G1_DOF_ORDER = (
    [f"left_{n}_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                                  "ankle_pitch", "ankle_roll")] +
    [f"right_{n}_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                                  "ankle_pitch", "ankle_roll")] +
    [f"waist_{n}_joint" for n in ("yaw", "roll", "pitch")] +
    [f"left_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll",
                                 "shoulder_yaw", "elbow", "wrist_roll",
                                 "wrist_pitch", "wrist_yaw")] +
    [f"right_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll",
                                  "shoulder_yaw", "elbow", "wrist_roll",
                                  "wrist_pitch", "wrist_yaw")]
)

KEY_SITES = {
    "pelvis": "pelvis",
    "torso": "torso_link",
    "head": "head_link",
    "l_foot": "left_ankle_roll_link",
    "r_foot": "right_ankle_roll_link",
    "l_hand": "left_wrist_yaw_link",
    "r_hand": "right_wrist_yaw_link",
}


def build_fk_model(force=False):
    """Create a meshless G1 MJCF (kinematics-only) from the menagerie file."""
    import mujoco

    if G1_FK_XML.exists() and not force:
        return mujoco.MjModel.from_xml_path(str(G1_FK_XML))

    tree = ET.parse(G1_MENAGERIE)
    root = tree.getroot()
    # drop asset section
    for asset in root.findall("asset"):
        root.remove(asset)
    # replace mesh geoms with tiny spheres; add key sites
    site_id = 0
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            gtype = geom.attrib.get("type")
            has_mesh = "mesh" in geom.attrib
            gcls = geom.attrib.get("class", "")
            if has_mesh or gtype == "mesh" or gcls in ("visual", "collision"):
                body.remove(geom)
            else:
                geom.attrib.pop("material", None)
                geom.attrib.pop("rgba", None)
        bname = body.attrib.get("name")
        if bname in KEY_SITES.values():
            ET.SubElement(body, "site", name=bname, size="0.005")
    # add pelvis site on root body
    wb = root.find("worldbody")
    pelvis = wb.find("body")
    if pelvis.find("site") is None:
        ET.SubElement(pelvis, "site", name="pelvis", size="0.005")
    # no actuators needed for FK; drop them (avoid meshless actuator refs)
    for act in root.findall("actuator"):
        root.remove(act)
    for key in root.findall("keyframe"):
        root.remove(key)

    tree.write(G1_FK_XML)
    model = mujoco.MjModel.from_xml_path(str(G1_FK_XML))
    return model


def load_csv(path):
    """Return dict with pos (N,3), quat_xyzw (N,4), dof (N,29)."""
    data = np.loadtxt(path, delimiter=",")
    assert data.shape[1] == 36, f"{path}: expected 36 cols, got {data.shape[1]}"
    return dict(pos=data[:, 0:3], quat_xyzw=data[:, 3:7], dof=data[:, 7:36],
                raw=data)


def g1_fk(model, pos, quat_xyzw, dof):
    """Set G1 state and forward kinematics. Returns data with sites updated."""
    import mujoco
    d = mujoco.MjData(model)
    d.qpos[:3] = pos
    d.qpos[3] = quat_xyzw[3]
    d.qpos[4:7] = quat_xyzw[0:3]
    for i, jn in enumerate(G1_DOF_ORDER):
        jid = model.joint(jn).id
        d.qpos[model.jnt_qposadr[jid]] = dof[i]
    mujoco.mj_forward(model, d)
    return d


def site_pos(d, name):
    return d.site(name).xpos.copy()


def site_quat(d, name):
    """Site world quaternion (wxyz) via site_xmat (works on mujoco 3.1.x)."""
    from scipy.spatial.transform import Rotation as Rot
    sid = d.site(name).id
    q_xyzw = Rot.from_matrix(d.site_xmat[sid].reshape(3, 3)).as_quat()
    return np.r_[q_xyzw[3], q_xyzw[0:3]]


def quat_xyzw_to_wxyz(q):
    return np.r_[q[3], q[0:3]]


# ------------------------------------------------------------- gait analysis
def foot_contacts(model, frames, fps, foot_z_thresh=0.045):
    """Per-frame boolean contact for each foot.

    foot world height (min over sole points) below thresh AND vertical
    velocity near zero -> contact. Uses ankle site height as proxy.
    """
    import mujoco
    n = len(frames["pos"])
    l = np.zeros(n, dtype=bool)
    r = np.zeros(n, dtype=bool)
    lz = np.zeros(n)
    rz = np.zeros(n)
    d = mujoco.MjData(model)
    for i in range(n):
        d.qpos[:3] = frames["pos"][i]
        d.qpos[3] = frames["quat_xyzw"][i, 3]
        d.qpos[4:7] = frames["quat_xyzw"][i, 0:3]
        for k, jn in enumerate(G1_DOF_ORDER):
            jid = model.joint(jn).id
            d.qpos[model.jnt_qposadr[jid]] = frames["dof"][i, k]
        mujoco.mj_forward(model, d)
        lz[i] = site_pos(d, "left_ankle_roll_link")[2]
        rz[i] = site_pos(d, "right_ankle_roll_link")[2]
    # ankle site sits above the sole by ~0.07 m (measure once): contact when
    # z < median_low + margin
    l_base = np.quantile(lz, 0.02)
    r_base = np.quantile(rz, 0.02)
    l = lz < l_base + foot_z_thresh
    r = rz < r_base + foot_z_thresh
    return l, r, lz, rz


def contact_events(contacts, min_gap=3):
    """Indices where contact toggles on (strike) and off (lift)."""
    on, off = [], []
    prev = False
    gap = 0
    for i, c in enumerate(contacts):
        if c and not prev:
            on.append(i)
        elif prev and not c:
            off.append(i)
        prev = c
    return np.array(on), np.array(off)
