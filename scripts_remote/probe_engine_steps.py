"""In-process engine step timing probe.

Builds the x1 character in the IsaacLab engine (1 env) via the same env
configs as training, then steps the sim 40 control steps with wall-clock
timing per step. This avoids relying on a full view_motion episode and
pinpoints slow/stuck stages (build, first step, steady-state rate).

Prints PROBE_STEP lines; exits 0 if all steps complete.
"""
import glob
import os
import sys


def find_repo_root():
    starts = []
    try:
        starts.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    starts.append(os.getcwd())
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
    import time

    repo = find_repo_root()
    os.chdir(repo)
    sys.path.insert(0, os.path.join(repo, "mimickit"))
    sys.path.insert(0, repo)
    print(f"[stepprobe] repo: {repo}", flush=True)

    # ensure the IsaacLab USD asset exists (MJCF -> USD conversion)
    import subprocess
    usd = os.path.join(repo, "data/assets/x1/x1.usd")
    if not os.path.exists(usd):
        r0 = subprocess.run([sys.executable, "scripts_remote/convert_x1_usd.py"])
        if r0.returncode != 0 or not os.path.exists(usd):
            print("PROBE_STEP_RESULT: FAIL (convert)", flush=True)
            sys.exit(1)

    import torch
    import envs.env_builder as env_builder

    num_envs = int(os.environ.get("PROBE_NUM_ENVS", "4"))
    t0 = time.time()
    env = env_builder.build_env(
        "data/envs/view_motion_x1_probe_env.yaml",
        "data/engines/isaac_lab_engine.yaml",
        num_envs=num_envs, device="cuda:0", visualize=False, record_video=False)
    print(f"PROBE_STEP build_done num_envs={num_envs} t={time.time()-t0:.1f}s",
          flush=True)

    env.set_mode(1)  # EnvMode.TEST

    import numpy as np
    a = torch.zeros([num_envs, 29], dtype=torch.float32, device="cuda:0")

    n = int(os.environ.get("PROBE_STEPS", "300"))
    times = []
    n_done = 0
    for i in range(n):
        t1 = time.time()
        obs, reward, done, info = env.step(a)
        dt = time.time() - t1
        times.append(dt)
        n_done += int((done != 0).sum().item())
        if i % 25 == 0 or i == n - 1:
            env_time = env.get_env_time()
            t0v = float(env_time.flatten()[0]) if hasattr(env_time, "flatten") else float(env_time)
            print(f"PROBE_STEP i={i} dt={dt:.3f}s mean={np.mean(times):.3f}s "
                  f"obs0={float(obs[0,0]):.3f} done_total={n_done} "
                  f"env_time0={t0v:.2f}",
                  flush=True)

    print(f"PROBE_STEP done mean_dt={np.mean(times):.3f}s "
          f"max_dt={np.max(times):.3f}s", flush=True)
    print("PROBE_STEP_RESULT: PASS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
