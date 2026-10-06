"""Definitive joint-mapping audit: single-joint command tests.

For a set of joints, command +0.4 rad on EXACTLY ONE joint (others =
home), step once from a clean airborne state, and report each joint's
delta. Correct mapping => the commanded joint moves most; others show
only small gravity effects.

Also verifies the invariant sim_names[sim2common[ci]] == common_names[ci]
for every joint (the alignment the previous audit actually needed).
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
        os.environ.get("X1_PROBE_ENGINE", "data/engines/isaac_lab_engine.yaml"),
        num_envs=1, device="cuda:0", visualize=False, record_video=False)
    eng = env._engine
    char_id = env._get_char_id()
    obj = eng._objs[char_id]
    dev = "cuda:0"

    sim_names = list(obj.joint_names)
    from tools.x1_pipeline.retarget_g1_x1 import X1_DOF_ORDER
    common_names = list(X1_DOF_ORDER)
    s2c = eng._dof_order_sim2common[char_id].cpu().numpy()

    ok = sum(1 for ci in range(29) if sim_names[s2c[ci]] == common_names[ci])
    print("MAP invariant sim_names[sim2common[ci]]==common[ci]: %d/29" % ok,
          flush=True)
    for ci in range(29):
        if sim_names[s2c[ci]] != common_names[ci]:
            print("MAP MISMATCH ci=%d %s -> sim %s"
                  % (ci, common_names[ci], sim_names[s2c[ci]]), flush=True)

    home_leg_l = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home_leg_r = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]
    home = np.array([0.0] * 17 + home_leg_l + home_leg_r, dtype=np.float32)

    def clean_state():
        eng.set_root_pos(None, char_id,
                         torch.tensor([[0.0, 0.0, 0.8]], device=dev))
        eng.set_root_rot(None, char_id,
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev))
        eng.set_dof_pos(None, char_id,
                        torch.tensor(home, device=dev).unsqueeze(0))
        eng.set_dof_vel(None, char_id, 0.0)
        eng.set_root_vel(None, char_id, 0.0)
        eng.set_root_ang_vel(None, char_id, 0.0)

    # warm-up step so writes propagate
    env.step(torch.tensor(home, dtype=torch.float32, device=dev).unsqueeze(0))

    test_cis = [0, 17, 18, 23, 24, 26, 6]
    for ci in test_cis:
        clean_state()
        action = home.copy()
        action[ci] += 0.4
        q0 = eng.get_dof_pos(char_id)[0].cpu().numpy()
        env.step(torch.tensor(action, dtype=torch.float32, device=dev).unsqueeze(0))
        delta = eng.get_dof_pos(char_id)[0].cpu().numpy() - q0
        top = np.argsort(-np.abs(delta))[:3]
        print("CMD %-28s -> top movers: %s" % (
            common_names[ci],
            " | ".join("%s %+.4f" % (common_names[j], delta[j]) for j in top)),
            flush=True)
    print("MAP RESULT: DONE", flush=True)


if __name__ == "__main__":
    main()
