"""Remote: render the trained X1 SMP policy in IsaacLab (X1 mesh, headless
viewport camera) and save an mp4 to the SDK-scanned logs/ directory.

Run inside the training container AFTER train_smp_policy_x1.py finishes:
  gm-run x1_mimicKit/scripts_remote/render_policy_x1.py [model_path]

Output: logs/x1_smp_policy/x1_policy_render.mp4 (uploaded as videoUrl).
Failure is isolated: exits 3 without touching weights.
"""
import os
import sys


def find_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "mimickit")) and os.path.isdir(os.path.join(d, "tools")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root not found")


def main():
    repo = find_repo_root()
    os.chdir(repo)
    # mimickit's run.py does `import envs.env_builder` -> modules live under
    # repo/mimickit/; keep both roots on the path
    sys.path.insert(0, os.path.join(repo, "mimickit"))
    sys.path.insert(0, repo)

    model_path = sys.argv[1] if len(sys.argv) > 1 else "output/x1_smp_policy/model.pt"
    if not os.path.isfile(model_path):
        print(f"[render] FAIL model not found: {model_path}", flush=True)
        sys.exit(2)

    import numpy as np
    import torch  # noqa: F401

    import util
    import util.mp_util as mp_util
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode

    if hasattr(util, 'set_rand_seed'):
        util.set_rand_seed(np.uint64(20261004))
    mp_util.init(0, 1, "cuda:0", "11555")

    env = env_builder.build_env(
        env_file="data/envs/smp_x1_env.yaml",
        engine_file="data/engines/isaac_lab_engine.yaml",
        num_envs=1, device="cuda:0",
        visualize=False, record_video=True)  # headless + enable_cameras
    print("[render] env built", flush=True)

    agent = agent_builder.build_agent("data/agents/smp_x1_agent.yaml", env, "cuda:0")
    agent.load(model_path)
    agent.set_mode(AgentMode.TEST)
    print("[render] policy loaded", flush=True)

    with torch.no_grad():
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        for _ in range(2):  # a few episodes for coverage
            agent._rollout_test(1)
    print("[render] rollout done", flush=True)

    vid = env._engine.get_video_recording()
    frames = vid.get_frames()
    if frames is None or len(frames) == 0:
        print("[render] FAIL no frames captured", flush=True)
        sys.exit(3)

    import cv2
    out_dir = os.path.join(repo, "logs", "x1_smp_policy")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "x1_policy_render.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"[render] saved {out} ({len(frames)} frames, {w}x{h})", flush=True)


if __name__ == "__main__":
    main()
