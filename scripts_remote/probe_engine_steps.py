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

    import torch
    import envs.env_builder as env_builder

    t0 = time.time()
    env = env_builder.build_env(
        "data/envs/view_motion_x1_probe_env.yaml",
        "data/engines/isaac_lab_engine.yaml",
        num_envs=1, device="cuda:0", visualize=False, record_video=False)
    print(f"PROBE_STEP build_done t={time.time()-t0:.1f}s", flush=True)

    env.set_mode(1)  # EnvMode.TEST

    import numpy as np
    a = torch.zeros([1, 29], dtype=torch.float32, device="cuda:0")

    n = 40
    times = []
    for i in range(n):
        t1 = time.time()
        obs, reward, done, info = env.step(a)
        dt = time.time() - t1
        times.append(dt)
        if i % 5 == 0 or i == n - 1:
            print(f"PROBE_STEP i={i} dt={dt:.3f}s "
                  f"mean={np.mean(times):.3f}s obs0={float(obs[0,0]):.3f}",
                  flush=True)

    print(f"PROBE_STEP done mean_dt={np.mean(times):.3f}s "
          f"max_dt={np.max(times):.3f}s", flush=True)
    print("PROBE_STEP_RESULT: PASS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
