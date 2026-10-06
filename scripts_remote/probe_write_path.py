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


if __name__ == "__main__":
    main()
