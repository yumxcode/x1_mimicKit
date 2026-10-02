"""Diagnose contact-sensor force layout + body index mapping on IsaacLab 6.

Builds the smp_x1 env (4 envs) through the real engine, then inspects:
  - ContactSensor.data.force_matrix_w shape
  - sensor.body_names
  - _sensor_body_order_sim2common values vs forces.shape[1]
  - contact_body_ids (from char_env _build_body_ids_tensor) vs force dim
Steps the env briefly to confirm stability, then reports.
"""
import glob
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
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
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
        r0 = subprocess.run([sys.executable, "scripts_remote/convert_x1_usd.py"])
        if r0.returncode != 0:
            print("DIAG RESULT: FAIL (convert)")
            sys.exit(1)

    import torch
    import envs.env_builder as env_builder

    env = env_builder.build_env(
        "data/envs/smp_x1_env.yaml",
        "data/engines/isaac_lab_engine.yaml",
        num_envs=4, device="cuda:0", visualize=False, record_video=False)

    eng = env._engine
    char_id = env._get_char_id()
    sensor = eng._ground_contact_sensors[char_id]

    fm = sensor.data.force_matrix_w
    print(f"DIAG force_matrix_w shape={tuple(fm.shape)} dtype={fm.dtype}")

    names = list(sensor.body_names)
    print(f"DIAG sensor.body_names n={len(names)} first5={names[:5]}")

    order = eng._sensor_body_order_sim2common[char_id]
    print(f"DIAG sensor_order_sim2common len={len(order)} "
          f"min={int(order.min())} max={int(order.max())}")

    body_names_engine = eng.get_obj_body_names(char_id)
    print(f"DIAG engine body_names n={len(body_names_engine)}")

    cb_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    cb_ids = env._contact_body_ids
    print(f"DIAG env._contact_body_ids tensor={cb_ids.tolist() if hasattr(cb_ids,'tolist') else cb_ids}")

    fm_sum = fm.sum(dim=-2) if fm.dim() == 4 else fm
    print(f"DIAG force dim after sum: {tuple(fm_sum.shape)}; "
          f"contact_body_ids max={int(max(cb_ids))} "
          f"in-bounds={int(max(cb_ids)) < fm_sum.shape[1]}")

    # a few steps through the failing path
    a = torch.zeros([4, 29], dtype=torch.float32, device="cuda:0")
    for i in range(3):
        obs, r, done, info = env.step(a)
        print(f"DIAG step {i} ok, done={done.tolist()}")
    print("DIAG RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
