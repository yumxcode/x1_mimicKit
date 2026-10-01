"""Remote launcher: train the X1 SMP diffusion prior (TinyMDM) on gradmotion.

Steps:
  1. locate repo (robust to gm-run's cwd), pip install diffusers if missing
  2. python tools/diffusion_model/train_tinymdm.py --cfg_path
     tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml --out_dir
     output/x1_smp_prior
  3. copy the prior artifacts to the SDK-scanned checkpoint dir
     logs/x1_prior/exported_data/<ts>/model_final.pt so they are uploaded
     and downloadable via `gm task model list`
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
    print(f"[prior] repo root: {repo}", flush=True)

    try:
        import diffusers  # noqa: F401
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "diffusers>=0.36.0"])
        assert r.returncode == 0, "pip install diffusers failed"

    out_dir = "output/x1_smp_prior"
    r = subprocess.run(
        [sys.executable, "tools/diffusion_model/train_tinymdm.py",
         "--cfg_path", "tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml",
         "--out_dir", out_dir, "--device", "cuda"])
    print(f"[prior] train exit {r.returncode}", flush=True)

    # publish artifacts to SDK-scanned path
    run_ts = time.strftime("%Y-%m-%d_%H-%M-%S") + "x1_prior"
    exp_dir = os.path.join("logs", "x1_prior", "exported_data", run_ts)
    os.makedirs(exp_dir, exist_ok=True)
    for f in ("model.pt", "diffusion_config.yaml", "env_config.yaml"):
        src = os.path.join(out_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(exp_dir, f))
            print(f"[prior] published {src} -> {exp_dir}/{f}", flush=True)

    ok = r.returncode == 0 and os.path.exists(os.path.join(out_dir, "model.pt"))
    print("[prior] RESULT:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
