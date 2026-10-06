"""Airborne 1-step PD response audit (IsaacLab side).

Builds the smp_x1 env, lifts the robot to z=0.8 (airborne, no contacts),
commands action = home + 0.4 rad on ALL joints, steps ONE control step,
and prints per-joint deltas + effort limits + applied torques.

Local twin: output/aircheck_mj.py. Deltas should match if the actuator
laws (kp, kd, effort clip) are aligned between engines.
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
    act = obj.actuators["actuators"]

    from engines.isaac_lab_engine import _to_torch
    kp = _to_torch(act.stiffness, eng._device)[0]
    kd = _to_torch(act.damping, eng._device)[0]
    print("AIR kp[:4]:", kp[:4].tolist(), flush=True)
    print("AIR kd[:4]:", kd[:4].tolist(), flush=True)
    try:
        eff = _to_torch(act.effort_limit, eng._device)[0]
        print("AIR effort[:4]:", eff[:4].tolist(), flush=True)
        print("AIR effort min/max: %.1f/%.1f"
              % (float(eff.min()), float(eff.max())), flush=True)
    except Exception as e:
        print("AIR effort read failed:", e, flush=True)

    # lift robot airborne, zero velocities
    home_leg_l = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home_leg_r = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]
    dev = 'cuda:0'
    home = np.array([0.0] * 17 + home_leg_l + home_leg_r, dtype=np.float32)

    eng.set_root_pos(None, char_id, torch.tensor([[0.0, 0.0, 0.8]], device=dev))
    eng.set_root_rot(None, char_id, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev))
    eng.set_dof_pos(None, char_id, torch.tensor(home, device=dev).unsqueeze(0))
    eng.set_dof_vel(None, char_id, 0.0)
    eng.set_root_vel(None, char_id, 0.0)
    eng.set_root_ang_vel(None, char_id, 0.0)

    action = home + 0.4  # uniform +0.4 rad on all 29 joints

    # step 0: applies the reset writes (state propagate); discard
    env.step(torch.tensor(home, dtype=torch.float32, device=dev).unsqueeze(0))
    rp = eng.get_root_pos(char_id)[0].cpu().numpy()
    print("AIR after reset-step root_pos:", rp.tolist(), flush=True)

    q0 = eng.get_dof_pos(char_id)[0].cpu().numpy()
    print("AIR q0 head:", q0[:5].tolist(), flush=True)
    # measured step: command the offset action
    env.step(torch.tensor(action, dtype=torch.float32, device=dev).unsqueeze(0))
    q1 = eng.get_dof_pos(char_id)[0].cpu().numpy()
    print("AIR root after step:", eng.get_root_pos(char_id)[0].cpu().numpy().tolist(), flush=True)
    delta = q1 - q0
    print("AIR delta (17 lumbar/arm, then legs):", flush=True)
    for i in (0, 3, 6, 17, 18, 23, 24, 29 - 1):
        print("AIR joint[%02d] delta %+.4f" % (i, delta[i]), flush=True)
    print("AIR delta mean %.4f max %.4f"
          % (np.abs(delta).mean(), np.abs(delta).max()), flush=True)
    print("AIR RESULT: DONE", flush=True)


if __name__ == "__main__":
    main()
