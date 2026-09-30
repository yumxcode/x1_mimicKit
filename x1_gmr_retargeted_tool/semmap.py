"""Semantic-frame joint mapping between G1 and X1 (anchor-based, v2.2).

X1's URDF link frames carry arbitrary 90/180 deg export rotations, so raw
link-frame matching is meaningless. We build segment frames from geometric
invariants (joint anchor positions + joint axis directions) and transfer
RELATIVE motion:

    rel(t)  = T_parent(t)^T @ T_child(t)
    H(t)    = rel(t) @ rel(stand)^-1            (motion since standing)
    X1 fit: joints q s.t. rel_x1(q) = H_g1(t) @ rel_x1(stand)

v2.2 fixes over the first anchor version:
  * reference-axis ORIENTATION canonicalization at standing: lateral-type
    axes (knee/ankle_pitch/elbow/wrist_pitch) oriented +pelvis-lateral,
    forward-type axes (ankle_roll) oriented +pelvis-forward. Without this,
    G1 elbow axis (+y) vs X1's (-y) flipped the transferred forearm motion.
  * foot frame: X1 ankle pitch/roll anchors are CO-LOCATED (0 mm), so the
    c0->c1 direction is degenerate; the foot frame uses the shank direction
    as z and the (oriented) ankle roll axis as x.
  * chain fits warm-start from the previous frame's solution and retry
    from random restarts when the residual exceeds 10 deg (2-dof ankle
    fits were falling into local minima).
"""

import numpy as np
import mujoco


def _u(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def R_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()
    return np.r_[q[3], q[0:3]]


# joint-name tables: spec key -> real joint name ({s}=left/right)
X1_A = {
    "lumbar_yaw": "lumbar_yaw_joint",
    "{s}_hip_pitch": "{s}_hip_pitch_joint",
    "{s}_knee": "{s}_knee_pitch_joint",
    "{s}_ankle_pitch": "{s}_ankle_pitch_joint",
    "{s}_ankle_roll": "{s}_ankle_roll_joint",
    "{s}_shoulder_pitch": "{s}_shoulder_pitch_joint",
    "{s}_elbow": "{s}_elbow_pitch_joint",
    "{s}_wrist_p": "{s}_wrist_pitch_joint",
}
G1_A = {
    "lumbar_yaw": "waist_yaw_joint",
    "{s}_hip_pitch": "{s}_hip_pitch_joint",
    "{s}_knee": "{s}_knee_joint",
    "{s}_ankle_pitch": "{s}_ankle_pitch_joint",
    "{s}_ankle_roll": "{s}_ankle_roll_joint",
    "{s}_shoulder_pitch": "{s}_shoulder_pitch_joint",
    "{s}_elbow": "{s}_elbow_joint",
    "{s}_wrist_p": "{s}_wrist_pitch_joint",
}

# axis orientation class per spec (see module docstring)
AXIS_CLASS = {
    "{s}_knee": "lateral",
    "{s}_ankle_pitch": "lateral",
    "{s}_elbow": "lateral",
    "{s}_wrist_p": "lateral",
    "{s}_ankle_roll": "forward",
}

# fit groups: (fit joint templates, parent_kind, child_kind, sides, mode)
# mode "abs": transfer the ABSOLUTE relative orientation parent->child
# (structural standing-pose differences ARE part of the pose to copy —
# used for the arm where G1/X1 standing abductions differ ~13 deg);
# mode "rel" (default): transfer only motion since standing (used where
# both robots' standing poses are structurally identical, i.e. legs/torso).
FIT_GROUPS = [
    (("lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint"),
     "pelvis", "torso", ("left",), "rel"),
    (("{s}_hip_pitch_joint", "{s}_hip_roll_joint", "{s}_hip_yaw_joint"),
     "pelvis", "thigh", ("left", "right"), "rel"),
    (("{s}_knee_pitch_joint",), "thigh", "shank", ("left", "right"), "rel"),
    (("{s}_ankle_pitch_joint", "{s}_ankle_roll_joint"), "shank", "foot",
     ("left", "right"), "rel"),
    (("{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint",
      "{s}_shoulder_yaw_joint"), "torso", "uarm", ("left", "right"), "abs"),
    # forearm NOT fitted as a full relative rotation: the standing physical
    # poses differ by 90 deg (G1 elbow-0 = forearm horizontal, X1 elbow-0 =
    # vertical), so the 3-dof rel-rotation target is unreachable with 2
    # joints and the LS compromise smears the bend; the forearm uses the
    # direction-fit transfer instead (see _forearm_direction_fit)
    # hands not fitted (visually negligible)
]


def _table(side, table):
    return {k.format(s=side) if "{s}" in k else k: v.format(s=side)
            for k, v in table.items()}


class SemanticMapper:
    def __init__(self, x1_model, g1_model):
        self.m = x1_model
        self.g1m = g1_model
        self.d = mujoco.MjData(x1_model)
        self.x1_tab = {s: _table(s, X1_A) for s in ("left", "right")}
        self.g1_tab = {s: _table(s, G1_A) for s in ("left", "right")}
        self._axis_sign = {}   # (robot, spec) -> +/-1 decided at standing
        self._warm = {}        # chain key -> last solution (warm start)
        self._standing(g1_model)

    # ------------------------------------------------------------- anchors
    def _anchors(self, d, model, tabs):
        A = {}
        for tab in tabs:
            for spec, real in tab.items():
                jid = model.joint(real).id
                A[spec] = dict(pos=d.xanchor[jid].copy(),
                               axis=d.xaxis[jid].copy())
        return A

    def _pelvis_frame(self, A):
        lh, rh = A["left_hip_pitch"]["pos"], A["right_hip_pitch"]["pos"]
        waist = A["lumbar_yaw"]["pos"]
        mid = 0.5 * (lh + rh)
        y = _u(lh - rh)
        z = _u(waist - mid)
        x = _u(np.cross(y, z))
        yy = np.cross(z, x)
        return np.column_stack([x, yy, z])   # columns: fwd, left, up

    def _frames(self, A, pelvis):
        out = {}
        for side in ("left", "right"):
            out[(side, "pelvis")] = pelvis
            other = "right" if side == "left" else "left"
            # torso
            ls, rs = A[f"{side}_shoulder_pitch"]["pos"], \
                A[f"{other}_shoulder_pitch"]["pos"]
            waist = A["lumbar_yaw"]["pos"]
            smid = 0.5 * (ls + rs)
            tz = _u(smid - waist)
            tx = _u(ls - rs)
            tyy = np.cross(tz, tx)
            out[(side, "torso")] = np.column_stack(
                [_u(np.cross(tyy, tz)), tyy, tz])
            segs = {
                "thigh": (f"{side}_hip_pitch", f"{side}_knee", f"{side}_knee"),
                "shank": (f"{side}_knee", f"{side}_ankle_pitch",
                          f"{side}_ankle_pitch"),
                "uarm": (f"{side}_shoulder_pitch", f"{side}_elbow",
                         f"{side}_elbow"),
                "farm": (f"{side}_elbow", f"{side}_wrist_p",
                         f"{side}_wrist_p"),
            }
            for kind, (c0, c1, ax) in segs.items():
                out[(side, kind)] = self._semframe(
                    A[c0]["pos"], A[c1]["pos"], A[ax], ax)
            # foot: co-located ankle anchors on X1 -> shank-direction z
            out[(side, "foot")] = self._footframe(
                A[f"{side}_ankle_pitch"]["pos"], A[f"{side}_ankle_roll"],
                A[f"{side}_knee"]["pos"])
        return out

    def _semframe(self, c0, c1, axis_info, ax_spec):
        # x = reference (joint) axis EXACTLY; z = limb direction projected
        # perpendicular to x. Rotations about the joint axis are then pure
        # rotations about frame-x (projecting the axis instead would smear
        # them with spurious twist when limb and axis are not orthogonal).
        a = np.asarray(axis_info["axis"]) * self._axis_sign.get(
            (axis_info["robot"], ax_spec), 1)
        x = _u(a)
        v = np.asarray(c1) - np.asarray(c0)
        z = v - x * (v @ x)
        if np.linalg.norm(z) < 1e-6:
            z = np.array([0.0, 0.0, -1.0]) - x * x[2]
        z = _u(z)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def _footframe(self, ankle_pos, roll_info, knee_pos):
        # x = ankle roll axis exactly; z = shank dir projected perp to x
        a = np.asarray(roll_info["axis"]) * self._axis_sign.get(
            (roll_info["robot"], "{s}_ankle_roll"), 1)
        x = _u(a)
        v = np.asarray(ankle_pos) - np.asarray(knee_pos)
        z = v - x * (v @ x)
        z = _u(z)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    # ----------------------------------------------------------- standing
    def _standing(self, g1_model):
        gd = mujoco.MjData(g1_model)
        mujoco.mj_forward(g1_model, gd)
        mujoco.mj_forward(self.m, self.d)

        def prep(d, model, tabs, robot):
            A = self._anchors(d, model, [tabs["left"], tabs["right"]])
            for k, v in A.items():
                v["robot"] = robot
            pelvis = self._pelvis_frame(A)
            return A, pelvis

        Ag, Pg = prep(gd, g1_model, self.g1_tab, "g1")
        Ax, Px = prep(self.d, self.m, self.x1_tab, "x1")

        # axis orientation canonicalization at standing: lateral +y_p,
        # forward +x_p — applied to BOTH robots' axes via sign map
        for spec, cls in AXIS_CLASS.items():
            for side in ("left", "right"):
                key = spec.format(s=side)
                for robot, A, P in (("g1", Ag, Pg), ("x1", Ax, Px)):
                    a = A[key]["axis"]
                    ref = P[:, 1] if cls == "lateral" else P[:, 0]
                    self._axis_sign[(robot, spec.format(s=side))] = \
                        1.0 if a @ ref >= 0 else -1.0
        self.g1_stand = self._frames(Ag, Pg)
        self.x1_stand = self._frames(Ax, Px)
        self.g1_rel_stand = self._rels(self.g1_stand)
        self.x1_rel_stand = self._rels(self.x1_stand)

    @staticmethod
    def _rels(fr):
        rels = {}
        for side in ("left", "right"):
            for pk, ck in (("pelvis", "torso"), ("pelvis", "thigh"),
                           ("thigh", "shank"), ("shank", "foot"),
                           ("torso", "uarm"), ("uarm", "farm")):
                rels[(side, pk, ck)] = fr[(side, pk)].T @ fr[(side, ck)]
        return rels

    # ------------------------------------------- forearm direction transfer
    def _farm_dir(self, d, model, side, robot):
        """World unit direction elbow -> wrist, from joint anchors."""
        el = ("elbow_joint" if robot == "g1"
              else "elbow_pitch_joint")
        wp = ("wrist_pitch_joint")
        a = d.xanchor[model.joint(f"{side}_{el}").id]
        b = d.xanchor[model.joint(f"{side}_{wp}").id]
        return _u(b - a)

    def _forearm_direction_fit(self, q_arm, gd, fr_g):
        """Solve X1 elbow pitch/yaw so the farm direction RELATIVE TO THE
        UARM matches G1's (absolute, in each uarm semantic frame).

        Rationale: the elbow zeros are structurally different (G1 elbow-0
        = forearm horizontal ~90-deg bend; X1 elbow-0 = forearm straight
        along the upper arm). Transferring the motion-since-standing left
        X1's forearm ~88 deg away from G1's in world while the 1-D
        angle-from-standing gates still passed. What a viewer perceives is
        HOW the forearm is bent w.r.t. the upper arm (bend angle + bend
        plane), so that absolute relative direction is the transfer
        target: 2 dof (direction on sphere) vs X1's elbow pitch+yaw
        (2 dof). Twist about the farm axis itself is visually negligible
        and dropped. The uarm semantic frames correspond frame-to-frame
        (x = canonicalized lateral elbow axis, z = shoulder->elbow).
        """
        from scipy.optimize import least_squares
        m, d = self.m, self.d
        out = {}
        for side in ("left", "right"):
            v_tgt = fr_g[(side, "uarm")].T @ self._farm_dir(
                gd, self.g1m, side, "g1")
            jp = m.joint(f"{side}_elbow_pitch_joint")
            jy = m.joint(f"{side}_elbow_yaw_joint")
            lo = np.array([m.jnt_range[jp.id][0], m.jnt_range[jy.id][0]])
            hi = np.array([m.jnt_range[jp.id][1], m.jnt_range[jy.id][1]])
            sh = [f"{side}_shoulder_pitch_joint",
                  f"{side}_shoulder_roll_joint",
                  f"{side}_shoulder_yaw_joint"]
            sh_q = [float(q_arm.get(j, 0.0)) for j in sh]

            def set_state(vals):
                d.qpos[:] = 0
                d.qpos[:3] = 0
                d.qpos[3:7] = (1.0, 0, 0, 0)
                for jn, v in zip(sh, sh_q):
                    d.qpos[m.joint(jn).qposadr[0]] = v
                d.qpos[jp.qposadr[0]] = vals[0]
                d.qpos[jy.qposadr[0]] = vals[1]
                mujoco.mj_forward(m, d)

            def uarm_frame():
                A = self._anchors(d, m, [self.x1_tab["left"],
                                         self.x1_tab["right"]])
                for vv in A.values():
                    vv["robot"] = "x1"
                return self._frames(A, self._pelvis_frame(A))[(side, "uarm")]

            # uarm frame depends only on the shoulder values (elbow anchor
            # is fixed on the uarm body) — compute once
            set_state(np.zeros(2))
            F = uarm_frame()

            def resid(vals):
                set_state(np.clip(vals, lo + 1e-4, hi - 1e-4))
                wx = self._farm_dir(d, m, side, "x1")
                return F.T @ wx - v_tgt

            wkey = ("farm", side)
            starts = []
            if wkey in self._warm:
                starts.append(self._warm[wkey])
            starts.append(np.array([1.2, 0.0]))  # typical running bend
            best = None
            for x0 in starts:
                sol = least_squares(resid, np.clip(x0, lo + 2e-4, hi - 2e-4),
                                    jac="2-point",
                                    bounds=(lo + 1e-4, hi - 1e-4),
                                    max_nfev=60)
                if best is None or sol.cost < best.cost:
                    best = sol
                if np.linalg.norm(best.fun) < 1e-3:
                    break
            for _ in range(4):
                if np.linalg.norm(best.fun) < 1e-3:
                    break
                rng = np.random.default_rng(11)
                x0 = rng.uniform(lo, hi)
                sol = least_squares(resid, x0, jac="2-point",
                                    bounds=(lo + 1e-4, hi - 1e-4),
                                    max_nfev=60)
                if sol.cost < best.cost:
                    best = sol
            self._warm[wkey] = best.x.copy()
            out[f"{side}_elbow_pitch_joint"] = float(
                np.clip(best.x[0], lo[0], hi[0]))
            out[f"{side}_elbow_yaw_joint"] = float(
                np.clip(best.x[1], lo[1], hi[1]))
        return out

    # --------------------------------------------------------------- solve
    def q_ref(self, gd):
        """X1 joint dict {name: value} for the G1 frame gd."""
        from scipy.optimize import least_squares
        from retarget_v2 import quat_log_diff
        from lib_g1 import G1_DOF_ORDER as _G1_ORDER
        g1_dof = {jn: float(gd.qpos[self.g1m.joint(jn).qposadr[0]])
                  for jn in _G1_ORDER}
        Ag = self._anchors(gd, self.g1m, [self.g1_tab["left"],
                                          self.g1_tab["right"]])
        for v in Ag.values():
            v["robot"] = "g1"
        fr = self._frames(Ag, self._pelvis_frame(Ag))
        rels = self._rels(fr)
        m, d = self.m, self.d
        q = {}

        def set_chain(fit_joints, vals):
            d.qpos[:] = 0
            d.qpos[:3] = 0
            d.qpos[3:7] = (1.0, 0, 0, 0)
            for jn, v in zip(fit_joints, vals):
                d.qpos[m.joint(jn).qposadr[0]] = v
            mujoco.mj_forward(m, d)

        def eval_err(fit_joints, vals, side, pk, ck, target):
            set_chain(fit_joints, np.clip(vals, lo + 1e-4, hi - 1e-4))
            Ax = self._anchors(d, m, [self.x1_tab["left"],
                                      self.x1_tab["right"]])
            for v in Ax.values():
                v["robot"] = "x1"
            frx = self._frames(Ax, self._pelvis_frame(Ax))
            got = frx[(side, pk)].T @ frx[(side, ck)]
            return quat_log_diff(np.array([1.0, 0, 0, 0]),
                                 R_to_quat_wxyz(target.T @ got))

        rng = np.random.default_rng(7)
        for fits, pk, ck, sides, mode in FIT_GROUPS:
            for side in sides:
                real_fits = [j.format(s=side) if "{s}" in j else j
                             for j in fits]
                lo = np.array([m.jnt_range[m.joint(j).id][0]
                               for j in real_fits])
                hi = np.array([m.jnt_range[m.joint(j).id][1]
                               for j in real_fits])
                if (hi - lo < 2e-4).all():
                    for j, v in zip(real_fits, 0.5 * (lo + hi)):
                        q[j] = float(v)
                    continue
                if mode == "abs":
                    target = rels[(side, pk, ck)].copy()
                else:
                    H = rels[(side, pk, ck)] @ \
                        self.g1_rel_stand[(side, pk, ck)].T
                    target = H @ self.x1_rel_stand[(side, pk, ck)]
                ckey = (side, ck)

                def resid(vals):
                    return eval_err(real_fits, vals, side, pk, ck, target)

                starts = []
                if ckey in self._warm:
                    starts.append(self._warm[ckey])
                starts.append(np.zeros(len(real_fits)))
                best = None
                for x0 in starts:
                    sol = least_squares(
                        resid, np.clip(x0, lo + 2e-4, hi - 2e-4),
                        jac="2-point", bounds=(lo + 1e-4, hi - 1e-4),
                        max_nfev=80)
                    if best is None or sol.cost < best.cost:
                        best = sol
                    if np.linalg.norm(best.fun) < np.radians(10):
                        break
                for _ in range(6):  # residual-restart loop
                    if np.linalg.norm(best.fun) < np.radians(10):
                        break
                    x0 = rng.uniform(lo, hi)
                    sol = least_squares(
                        resid, np.clip(x0, lo + 2e-4, hi - 2e-4),
                        jac="2-point", bounds=(lo + 1e-4, hi - 1e-4),
                        max_nfev=80)
                    if sol.cost < best.cost:
                        best = sol
                # bend-only fallback when the full 2-dof manifold fit
                # fails (X1 elbow_yaw axis is vertical at the straight-arm
                # zero pose, making generic bend+twist targets unreachable)
                if (ck == "farm"
                        and np.linalg.norm(best.fun) > np.radians(15)):
                    keep = [j for j in real_fits if "elbow_yaw" not in j]
                    klo = np.array([m.jnt_range[m.joint(j).id][0]
                                    for j in keep])
                    khi = np.array([m.jnt_range[m.joint(j).id][1]
                                    for j in keep])
                    order = [real_fits.index(j) for j in keep]

                    def resid_k(vals):
                        full = np.zeros(len(real_fits))
                        for i, oi in enumerate(order):
                            full[oi] = vals[i]
                        return eval_err(real_fits, full, side, pk, ck,
                                        target)

                    sol = least_squares(
                        resid_k, np.clip(np.zeros(len(keep)), klo + 2e-4,
                                         khi - 2e-4),
                        jac="2-point", bounds=(klo + 1e-4, khi - 1e-4),
                        max_nfev=80)
                    if sol.cost < best.cost:
                        full = np.zeros(len(real_fits))
                        for i, oi in enumerate(order):
                            full[oi] = sol.x[i]
                        best_full = np.clip(full, lo, hi)
                        self._warm[ckey] = best_full.copy()
                        for j, v in zip(real_fits, best_full):
                            q[j] = float(v)
                        continue
                self._warm[ckey] = best.x.copy()
                for j, v in zip(real_fits, np.clip(best.x, lo, hi)):
                    q[j] = float(v)
        q.update(self._forearm_direction_fit(q, gd, fr))
        return q
