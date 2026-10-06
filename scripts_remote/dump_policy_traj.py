"""Dump IsaacLab policy rollouts as trajectory pkls (for LOCAL x1-mesh
video rendering).

Loads a policy (default: newest in data/models/smp_policies), runs
test-mode rollouts in the SMP env (no GSI; motion init from dataset), and
saves {root_pos, root_rot(wxyz), dof} @30 Hz per episode. Also mirrors the
pkls into the SDK-scanned dir so they are uploaded for download.

Usage (gradmotion):
  gm-run x1_mimicKit/scripts_remote/dump_policy_traj.py
Env overrides:
  X1_DUMP_POLICY - path to a specific .pt (default: newest staged policy)
  X1_DUMP_EPISODES - number of episodes (default 2)
"""
import glob
import os
import pickle
import sys
import time


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
    print(f"[dump] repo: {repo}", flush=True)

    import subprocess
    try:
        import diffusers  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "diffusers>=0.36.0"])

    usd = os.path.join(repo, "data/assets/x1/x1.usd")
    if not os.path.exists(usd):
        r = subprocess.run([sys.executable, "scripts_remote/convert_x1_usd.py"])
        if r.returncode != 0:
            print("[dump] FAIL convert")
            sys.exit(1)

    policy = os.environ.get("X1_DUMP_POLICY", "")
    if not policy:
        staged = os.path.join(repo, "data/models/smp_policies")
        import re

        def iters(p):
            m = re.search(r"(\d+)k(\d*)", os.path.basename(p))
            return int(m.group(1)) * 1000 + int(m.group(2) or 0) if m else -1
        cands = sorted(glob.glob(os.path.join(staged, "*.pt")), key=iters)
        if not cands:
            print("[dump] FAIL no policy staged")
            sys.exit(1)
        policy = cands[-1]
    print(f"[dump] policy: {policy}", flush=True)

    n_eps = int(os.environ.get("X1_DUMP_EPISODES", "2"))

    import numpy as np
    import torch
    import yaml
    import envs.env_builder as env_builder

    env = env_builder.build_env(
        "data/envs/smp_x1_env.yaml",
        engine_cfg,
        num_envs=1, device="cuda:0", visualize=False, record_video=False)

    # build agent the standard way from configs (mirrors run.py)
    from learning.smp_agent import SMPAgent
    ac = yaml.safe_load(open("data/agents/smp_x1_agent.yaml"))
    agent = SMPAgent(config=ac, env=env, device="cuda:0")
    state = torch.load(policy, map_location="cuda:0")
    agent.load_state_dict(state)
    agent._sync_optimizer()
    agent.eval()
    from learning.base_agent import AgentMode
    agent.set_mode(AgentMode.TEST)

    eng = env._engine
    char_id = env._get_char_id()

    exp_dir = os.path.join("logs", "x1_traj_dump", "exported_data",
                           time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir, exist_ok=True)

    total_eps = 0
    ep = 0
    obs, info = env.reset()
    steps = 0
    tr = dict(root_pos=[], root_rot=[], dof=[], t=[], obs=[])
    tr["obs"].append(np.asarray(obs[0].cpu().numpy()))
    tr["init"] = dict(
        root_pos=eng.get_root_pos(char_id)[0].cpu().numpy().copy(),
        root_rot=eng.get_root_rot(char_id)[0].cpu().numpy().copy(),
        dof=eng.get_dof_pos(char_id)[0].cpu().numpy().copy(),
        root_vel=eng.get_root_vel(char_id)[0].cpu().numpy().copy(),
        root_ang_vel=eng.get_root_ang_vel(char_id)[0].cpu().numpy().copy(),
        dof_vel=eng.get_dof_vel(char_id)[0].cpu().numpy().copy())
    max_steps = int(10 * 30)  # 10 s @ 30 Hz

    while total_eps < n_eps and steps < max_steps * n_eps:
        with torch.no_grad():
            norm_obs = agent._obs_norm.normalize(obs)
            dist = agent._model.eval_actor(norm_obs)
            a = agent._a_norm.unnormalize(dist.mode)
        obs, r, done, info = env.step(a)

        root_pos = eng.get_root_pos(char_id)[0].cpu().numpy()
        root_rot = eng.get_root_rot(char_id)[0].cpu().numpy()  # wxyz
        dof = eng.get_dof_pos(char_id)[0].cpu().numpy()
        tr["root_pos"].append(root_pos.copy())
        tr["root_rot"].append(root_rot.copy())
        tr["dof"].append(dof.copy())
        tr["t"].append(steps / 30.0)
        tr["obs"].append(np.asarray(obs[0].cpu().numpy()))
        steps += 1

        if (done[0] != 0).item():
            out = os.path.join(exp_dir, f"traj_ep{ep}.pt")
            with open(out, "wb") as f:
                pickle.dump({k: np.array(v) for k, v in tr.items()}, f)
            print(f"[dump] episode {ep}: {len(tr['t'])} steps -> {out}",
                  flush=True)
            ep += 1
            total_eps += 1
            tr = dict(root_pos=[], root_rot=[], dof=[], t=[], obs=[])
            obs, info = env.reset()
            tr["obs"].append(np.asarray(obs[0].cpu().numpy()))
            tr["init"] = dict(
                root_pos=eng.get_root_pos(char_id)[0].cpu().numpy().copy(),
                root_rot=eng.get_root_rot(char_id)[0].cpu().numpy().copy(),
                dof=eng.get_dof_pos(char_id)[0].cpu().numpy().copy(),
                root_vel=eng.get_root_vel(char_id)[0].cpu().numpy().copy(),
                root_ang_vel=eng.get_root_ang_vel(char_id)[0].cpu().numpy().copy(),
                dof_vel=eng.get_dof_vel(char_id)[0].cpu().numpy().copy())

    # final partial episode if any
    if tr["t"]:
        out = os.path.join(exp_dir, f"traj_ep{ep}.pt")
        with open(out, "wb") as f:
            pickle.dump({k: np.array(v) for k, v in tr.items()}, f)
        print(f"[dump] episode {ep} (partial): {len(tr['t'])} steps -> {out}",
              flush=True)

    n = len(glob.glob(os.path.join(exp_dir, "traj_ep*.pt")))
    print(f"[dump] RESULT: {'PASS' if n > 0 else 'FAIL'} ({n} episodes)",
          flush=True)
    sys.exit(0 if n > 0 else 1)


if __name__ == "__main__":
    main()
