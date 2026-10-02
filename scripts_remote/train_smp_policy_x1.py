"""Remote launcher: train the X1 SMP policy with a trained prior on gradmotion.

Requires the prior artifacts mounted at
  <repo>/output/x1_smp_prior/{model.pt, diffusion_config.yaml}
(e.g. uploaded via personal storage or a resume-task checkpoint mount).

Steps:
  1. locate repo, pip install diffusers
  2. verify prior files exist at output/x1_smp_prior/
  3. python mimickit/run.py --arg_file args/smp_x1_args.txt
  4. mirror intermediate models to the SDK-scanned dir
     logs/x1_smp_policy/exported_data/<ts>/model_<iter>.pt
"""
import glob
import os
import shutil
import subprocess
import sys
import time


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
    repo = find_repo_root()
    os.chdir(repo)
    print(f"[policy] repo root: {repo}", flush=True)

    try:
        import diffusers  # noqa: F401
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "diffusers>=0.36.0"])
        assert r.returncode == 0, "pip install diffusers failed"

    prior_model = os.path.join(repo, "data/models/smp_priors/x1_prior.pt")
    prior_cfg = os.path.join(
        repo, "tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml")
    missing = [p for p in (prior_model, prior_cfg) if not os.path.isfile(p)]
    if missing:
        print(f"[policy] FAIL prior artifacts missing: {missing}", flush=True)
        sys.exit(1)

    # ensure the IsaacLab USD asset exists (MJCF -> USD conversion)
    usd = os.path.join(repo, "data/assets/x1/x1.usd")
    if not os.path.exists(usd):
        r = subprocess.run([sys.executable, "scripts_remote/convert_x1_usd.py"])
        if r.returncode != 0 or not os.path.exists(usd):
            print("[policy] FAIL mjcf->usd conversion failed", flush=True)
            sys.exit(1)

    out_dir = "output/x1_smp_policy"
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")  # surface async asserts
    r = subprocess.run(
        [sys.executable, "mimickit/run.py", "--arg_file", "args/smp_x1_args.txt"],
        env=env)
    print(f"[policy] train exit {r.returncode}", flush=True)

    # mirror all model files to the SDK-scanned dir
    run_ts = time.strftime("%Y-%m-%d_%H-%M-%S") + "x1_smp_policy"
    exp_dir = os.path.join("logs", "x1_smp_policy", "exported_data", run_ts)
    os.makedirs(exp_dir, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(out_dir, "**", "model_*.pt"),
                                recursive=True)):
        iter_name = os.path.basename(src)
        shutil.copy2(src, os.path.join(exp_dir, iter_name))
        print(f"[policy] published {src}", flush=True)
    final = os.path.join(out_dir, "model.pt")
    if os.path.exists(final):
        shutil.copy2(final, os.path.join(exp_dir, "model_final.pt"))
        print(f"[policy] published {final} -> model_final.pt", flush=True)
    has_final = os.path.exists(final)
    n_int = len(glob.glob(os.path.join(out_dir, "**", "model_*.pt"),
                          recursive=True))
    ok = r.returncode == 0 and (has_final or n_int > 0)
    print(f"[policy] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(final={has_final} int_models={n_int})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
