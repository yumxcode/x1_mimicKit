"""Decisive state-write path diagnostic.

Answers, in ONE run:
  1. Which write methods exist on the runtime Articulation + signatures.
  2. Does writing obj.data.joint_pos + engine.step() change the SIM state?
  3. Does the keyword API write_joint_state_to_sim_index(position=...)
     change the SIM state?

Verdict printed as WRITE_RESULT lines.
"""
import glob
import inspect
import os
import sys


def find_repo_root():
    starts = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for start in starts:
        d = start
        for _ in range(6):
            if (os.path.isdir(os.path.join(d, "mimickit"))
                    and os.path.isdir(os.path.join(d, "tools"))):
                return d
            p = os.path.dirname(d)
            if p == d:
                break
            d = p
    for d in sorted(glob.glob("/workspace/*")) + ["/workspace"]:
        if (os.path.isdir(os.path.join(d, "mimickit"))
                and os.path.isdir(os.path.join(d, "tools"))):
            return d
    raise RuntimeError("repo root not found")


def main():
    repo = find_repo_root()
    os.chdir(repo)
    sys.path.insert(0, os.path.join(repo, "mimickit"))
    sys.path.insert(0, repo)

    import subprocess
    usd = os.path.join(repo, "data/assets/x1/x1.usd")
    if not os.path.exists(usd):
        subprocess.run([sys.executable, "scripts_remote/convert_x1_usd.py"])

    import numpy as np
    import torch
    import envs.env_builder as env_builder

    env = env_builder.build_env(
        "data/envs/smp_x1_probe_env.yaml",
        "data/engines/isaac_lab_engine.yaml",
        num_envs=1, device="cuda:0", visualize=False, record_video=False)
    eng = env._engine
    char_id = env._get_char_id()
    obj = eng._objs[char_id]
    dev = "cuda:0"

    print("WRITE methods present:", flush=True)
    for name in ("write_joint_position_to_sim", "write_joint_position_to_sim_index",
                 "write_joint_velocity_to_sim", "write_joint_state_to_sim_index",
                 "write_root_link_pose_to_sim", "write_root_link_pose_to_sim_index",
                 "write_data_to_sim", "reset"):
        print("WRITE %-42s %s" % (name, hasattr(obj, name)), flush=True)
        if hasattr(obj, name):
            try:
                print("WRITE   sig:", str(inspect.signature(getattr(obj, name)))[:200], flush=True)
            except Exception as e:
                print("WRITE   sig fail:", e, flush=True)

    MARK = 0.123  # distinctive joint value
    home = torch.zeros(29, dtype=torch.float32, device=dev)

    def sim_truth(marker):
        # advance one physics substep WITHOUT control writes, then read
        # the data buffer refreshed from sim
        obj.data.update(1.0 / 120.0)
        q = obj.data.joint_pos[0].detach().clone()
        print("WRITE %-12s q[:5]=%s q[7]=%.4f" % (
            marker, [round(float(v), 4) for v in q[:5]], float(q[7])), flush=True)
        return q

    # --- test A: engine set_dof_pos + engine.step
    eng.set_dof_pos(None, char_id, home + MARK)
    eng._flag_obj_needs_reset(None, char_id)
    eng.step()  # resets propagate + 4 substeps, no new targets
    qa = sim_truth("A engine")

    # --- test B: keyword API direct
    pos = (home + 2 * MARK).unsqueeze(0)
    try:
        obj.write_joint_state_to_sim_index(position=pos, velocity=torch.zeros_like(pos))
        print("WRITE B keyword call OK", flush=True)
    except Exception as e:
        print("WRITE B keyword call FAILED: %s: %s" % (type(e).__name__, e), flush=True)
        try:
            obj.write_joint_position_to_sim_index(position=pos)
            print("WRITE B2 pos-only keyword OK", flush=True)
        except Exception as e2:
            print("WRITE B2 FAILED: %s: %s" % (type(e2).__name__, e2), flush=True)
    import mujoco  # noqa: F401  (unused; keep namespace clean)
    obj.data.update(1.0 / 120.0)
    qb = obj.data.joint_pos[0].detach().clone()
    print("WRITE %-12s q[:5]=%s q[7]=%.4f" % (
        "B keyword", [round(float(v), 4) for v in qb[:5]], float(qb[7])), flush=True)

    verdict_a = "LANDED" if abs(float(qa[0]) - MARK) < 0.05 else "NOT-LANDED"
    verdict_b = "LANDED" if abs(float(qb[0]) - 2 * MARK) < 0.05 else "NOT-LANDED"
    print("WRITE_RESULT A(engine set_dof_pos+step)=%s B(keyword API)=%s"
          % (verdict_a, verdict_b), flush=True)

    # --- test C: ENGINE-level single-joint command (bypasses env.step)
    home_leg_l = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home_leg_r = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]
    home_full = np.array([0.0] * 17 + home_leg_l + home_leg_r,
                         dtype=np.float32)
    common_names = ["lumbar_yaw_joint"]
    from tools.x1_pipeline.retarget_g1_x1 import X1_DOF_ORDER
    common_names = list(X1_DOF_ORDER)

    for ci in (0, 17, 23, 6, 7):
        # reset sim to a clean airborne home via keyword API
        pos = torch.tensor(home_full, device=dev).unsqueeze(0)
        vel = torch.zeros_like(pos)
        obj.write_joint_state_to_sim_index(position=pos, velocity=vel)
        obj.write_root_link_pose_to_sim_index(
            root_pose=torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]],
                                   device=dev))
        obj.write_root_link_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=dev))

        action = torch.tensor(home_full, device=dev).unsqueeze(0).clone()
        action[0, ci] += 0.4
        eng.set_cmd(char_id, action)
        eng.step()

        obj.data.update(1.0 / 120.0)
        q = obj.data.joint_pos[0].detach().cpu().numpy()
        delta = q - home_full
        top = np.argsort(-np.abs(delta))[:3]
        print("WRITE C cmd %-28s -> %s" % (
            common_names[ci],
            " | ".join("%s %+.4f" % (common_names[j], delta[j])
                       for j in top)), flush=True)
    print("WRITE_RESULT C done", flush=True)

    # --- test D: pos-mode runtime drive gains (implicit actuator)
    act = obj.actuators["actuators"]
    for attr in ("stiffness", "damping", "effort_limit"):
        if hasattr(act, attr):
            try:
                v = getattr(act, attr)
                from engines.isaac_lab_engine import _to_torch
                t = _to_torch(v, dev)[0].cpu()
                print("WRITE D %s head: %s min/max %.1f/%.1f"
                      % (attr, t[:4].tolist(), float(t.min()), float(t.max())),
                      flush=True)
            except Exception as e:
                print("WRITE D %s read failed: %s" % (attr, e), flush=True)
    for meth in ("write_joint_stiffness_to_sim", "write_joint_damping_to_sim"):
        print("WRITE D has %s: %s" % (meth, hasattr(obj, meth)), flush=True)
    # if runtime gains are zero, write them explicitly and re-run test C once
    try:
        from engines.isaac_lab_engine import _to_torch
        kp_now = _to_torch(act.stiffness, dev)[0]
        if float(kp_now.max()) == 0.0:
            print("WRITE D zero gains detected -> writing explicit", flush=True)
            kp_common = torch.tensor(
                [375.0] * 3 + [50.0] * 14 + [375.0] * 6 + [200.0, 375, 375, 450, 80, 80] * 1,
                device=dev)[:29]
            # use MJCF-parsed gains via engine helper
            gains = eng._parse_mjcf_gains("data/assets/x1/x1.xml")
            names_g, kps_g, kds_g, effs_g = gains
            kp_sim = torch.tensor(kps_g, device=dev)
            kd_sim = torch.tensor(kds_g, device=dev)
            # map common->sim
            perm = eng._dof_order_sim2common[char_id].long()
            obj.write_joint_stiffness_to_sim(kp_sim[perm])
            obj.write_joint_damping_to_sim(kd_sim[perm])
            print("WRITE D wrote kp/kd via legacy methods", flush=True)
            # re-run single-joint test
            ci = 17
            pos = torch.tensor(home_full, device=dev).unsqueeze(0)
            obj.write_joint_state_to_sim_index(position=pos,
                                               velocity=torch.zeros_like(pos))
            obj.write_root_link_pose_to_sim_index(
                root_pose=torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]],
                                       device=dev))
            action = torch.tensor(home_full, device=dev).unsqueeze(0).clone()
            action[0, ci] += 0.4
            eng.set_cmd(char_id, action)
            eng.step()
            obj.data.update(1.0 / 120.0)
            q = obj.data.joint_pos[0].detach().cpu().numpy()
            delta = q - home_full
            top = np.argsort(-np.abs(delta))[:3]
            print("WRITE D retry cmd %-28s -> %s" % (
                common_names[ci],
                " | ".join("%s %+.4f" % (common_names[j], delta[j])
                           for j in top)), flush=True)
    except Exception as e:
        print("WRITE D explicit-write path failed: %s: %s"
              % (type(e).__name__, e), flush=True)
    print("WRITE_RESULT D done", flush=True)

    # --- test E: does set_joint_effort_target reach physics?
    try:
        pos = torch.tensor(home_full, device=dev).unsqueeze(0)
        obj.write_joint_state_to_sim_index(position=pos,
                                           velocity=torch.zeros_like(pos))
        obj.write_root_link_pose_to_sim_index(
            root_pose=torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]],
                                   device=dev))
        obj.write_root_link_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=dev))
        ci = 17
        tau = torch.zeros(1, 29, device=dev)
        tau[0, ci] = 50.0  # 50 Nm on left_hip_pitch (common idx)
        perm = eng._dof_order_sim2common[char_id].long()
        obj.set_joint_effort_target(tau[:, perm])
        eng.step()
        obj.data.update(1.0 / 120.0)
        q = obj.data.joint_pos[0].detach().cpu().numpy()
        delta = q - home_full
        top = np.argsort(-np.abs(delta))[:3]
        print("WRITE E effort50 %-24s -> %s" % (
            common_names[ci],
            " | ".join("%s %+.4f" % (common_names[j], delta[j])
                       for j in top)), flush=True)
    except Exception as e:
        print("WRITE E failed: %s: %s" % (type(e).__name__, e), flush=True)

    # --- test F: inspect collection buffers after set_cmd
    try:
        coll = obj.actuators
        pos = torch.tensor(home_full, device=dev).unsqueeze(0)
        obj.write_joint_state_to_sim_index(position=pos,
                                           velocity=torch.zeros_like(pos))
        action = torch.tensor(home_full, device=dev).unsqueeze(0).clone()
        action[0, 17] += 0.4
        eng.set_cmd(char_id, action)
        for bname in ("_joint_pos_target", "_joint_pos_target_sim",
                      "_joint_effort_target", "_joint_effort_target_sim"):
            buf = getattr(coll, bname, None)
            if buf is None:
                print("WRITE F %-24s absent" % bname, flush=True)
                continue
            try:
                t = torch.as_tensor(np.asarray(buf))[0]
                print("WRITE F %-24s head=%s nonzero=%d"
                      % (bname, [round(float(v), 3) for v in t[:5].cpu()],
                         int((t != 0).sum())), flush=True)
            except Exception as e:
                print("WRITE F %-24s read err %s" % (bname, e), flush=True)
    except Exception as e:
        print("WRITE F failed: %s: %s" % (type(e).__name__, e), flush=True)
    print("WRITE_RESULT E/F done", flush=True)

    # --- test G: dump runtime source of the command pathway
    try:
        import inspect as _insp
        cls_file = _insp.getsourcefile(type(obj))
        print("WRITE G obj class file:", cls_file, flush=True)
        src = open(cls_file).read()
        lines = src.splitlines()
        for pat, span in (("def write_data_to_sim", 90),
                          ("def set_joint_position_target_index", 60)):
            for i, l in enumerate(lines):
                if pat in l:
                    for j in range(i, min(len(lines), i + span)):
                        print("WRITE G | " + lines[j][:150], flush=True)
                    break
        # the actuator COLLECTION object (obj.actuators is a dict of groups)
        coll = obj.actuators["actuators"] if isinstance(obj.actuators, dict) else obj.actuators
        coll_cls_file = _insp.getsourcefile(type(coll))
        print("WRITE G coll class file:", coll_cls_file, flush=True)
        src2 = open(coll_cls_file).read()
        lines2 = src2.splitlines()
        for pat, span in (("def submit_commands", 60),
                          ("def compute", 40)):
            for i, l in enumerate(lines2):
                if pat in l:
                    print("WRITE G ---- coll %s @%d ----" % (pat, i + 1), flush=True)
                    for j in range(i, min(len(lines2), i + span)):
                        print("WRITE G | " + lines2[j][:150], flush=True)
                    break
    except Exception as e:
        print("WRITE G failed: %s: %s" % (type(e).__name__, e), flush=True)
    print("WRITE_RESULT G done", flush=True)


if __name__ == "__main__":
    main()
