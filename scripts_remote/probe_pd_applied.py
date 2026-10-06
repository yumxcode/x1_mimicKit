"""Airborne PD audit v2: compare commanded vs ACTUATOR-APPLIED torques.

One control step from a clean airborne home+0.4 state; prints:
  - expected tau (engine-side manual PD, common order)
  - actuator.applied_effort (what IsaacLab actually submitted, sim order)
  - resulting joint deltas
If applied matches expected per-joint, the plumbing is correct.
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
        "data/envs/smp_x1_env.yaml",
        "data/engines/isaac_lab_engine_pd.yaml",
        num_envs=1, device="cuda:0", visualize=False, record_video=False)
    eng = env._engine
    char_id = env._get_char_id()
    obj = eng._objs[char_id]

    dev = "cuda:0"
    home_leg_l = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home_leg_r = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]
    home = np.array([0.0] * 17 + home_leg_l + home_leg_r, dtype=np.float32)

    # reset-step so state writes propagate
    env.step(torch.tensor(home, dtype=torch.float32, device=dev).unsqueeze(0))

    # clean state
    eng.set_root_pos(None, char_id, torch.tensor([[0.0, 0.0, 0.8]], device=dev))
    eng.set_root_rot(None, char_id, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev))
    eng.set_dof_pos(None, char_id, torch.tensor(home, device=dev).unsqueeze(0))
    eng.set_dof_vel(None, char_id, 0.0)
    eng.set_root_vel(None, char_id, 0.0)
    eng.set_root_ang_vel(None, char_id, 0.0)

    action = home + 0.4
    q0 = eng.get_dof_pos(char_id)[0].cpu().numpy()

    env.step(torch.tensor(action, dtype=torch.float32, device=dev).unsqueeze(0))

    q1 = eng.get_dof_pos(char_id)[0].cpu().numpy()
    delta = q1 - q0

    # expected torque in common order (what set_cmd computed)
    tau_exp = eng._pd_kp_common.cpu().numpy() * 0.4  # qd=0 at start
    tau_exp = np.clip(tau_exp, -eng._pd_eff_common.cpu().numpy(),
                      eng._pd_eff_common.cpu().numpy())

    # sim-order joint names from the articulation
    sim_names = list(obj.joint_names)
    print("AIR sim joint order head:", sim_names[:6], flush=True)

    from tools.x1_pipeline.retarget_g1_x1 import X1_DOF_ORDER
    common_names = list(X1_DOF_ORDER)
    perm = eng._dof_order_common2sim[char_id].cpu().numpy()
    print("AIR common2sim perm head:", perm[:8].tolist(), flush=True)

    act = obj.actuators["actuators"]
    from engines.isaac_lab_engine import _to_torch
    applied = None
    for attr in ("applied_effort", "computed_effort"):
        if hasattr(act, attr):
            try:
                v = getattr(act, attr)
                applied = _to_torch(v, eng._device)[0].cpu().numpy()
                print("AIR using actuator attr:", attr, flush=True)
                break
            except Exception as e:
                print("AIR attr %s failed: %s" % (attr, e), flush=True)
    # verify perm alignment explicitly
    for ci in (0, 17, 18, 23):
        si = int(perm[ci])
        print("AIR align ci=%d %-28s -> si=%d %s" %
              (ci, common_names[ci], si,
               sim_names[si] if si < len(sim_names) else "?"), flush=True)

    print("\nAIR per-joint: expected(common) vs applied(sim) vs delta(common)",
          flush=True)
    for ci in (0, 3, 6, 17, 18, 23, 24, 28):
        si = int(perm[ci])
        ap_s = applied[si] if applied is not None else float('nan')
        ap_c = applied[ci] if applied is not None else float('nan')
        print("AIR %-28s exp %+6.0f | ap[si] %+7.1f ap[ci] %+7.1f | d %+.4f"
              % (common_names[ci], tau_exp[ci], ap_s, ap_c, delta[ci]),
              flush=True)
    print("AIR delta mean %.4f" % np.abs(delta).mean(), flush=True)
    print("AIR RESULT: DONE", flush=True)


if __name__ == "__main__":
    main()
