"""Remote launcher: train the X1 SMP diffusion prior (TinyMDM) on gradmotion.

Fixed upload semantics (the SDK uploads only NEWLY-DETECTED files and
ignores later overwrites of an existing name - the 1st run silently
shipped the iter-2000 snapshot):
  * watcher thread copies snapshots under UNIQUE iter-tagged names
  * final weights published as model_final.pt (unique vs model.pt)
  * post-train self-check: reload final, evaluate loss on a fresh batch,
    print FINAL_LOSS for a quick remote readout
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
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
    run_ts = time.strftime("%Y-%m-%d_%H-%M-%S") + "x1_prior"
    exp_dir = os.path.join("logs", "x1_prior", "exported_data", run_ts)
    os.makedirs(exp_dir, exist_ok=True)

    def _publish(tag):
        try:
            src = os.path.join(out_dir, "model.pt")
            if not os.path.exists(src):
                src = os.path.join(out_dir, "model_train.pt")
            if not os.path.exists(src):
                return
            # two channels: exported_data (index dir) + gm_play (SDK-doc
            # scanned pt dir)
            for base in (exp_dir,
                         os.path.join("logs", "x1_prior", "gm_play")):
                os.makedirs(base, exist_ok=True)
                dst = os.path.join(base, f"model_{tag}.pt")
                n = 0
                while os.path.exists(dst):
                    n += 1
                    dst = os.path.join(base, f"model_{tag}_{n}.pt")
                shutil.copy2(src, dst)
            print(f"[prior] published snapshot {tag}", flush=True)
        except Exception as e:
            print(f"[prior] watcher error: {e}", flush=True)

    def _watcher(stop):
        last = -1
        while not stop.is_set():
            try:
                # during training the periodic weights live in model_train.pt
                # (X1_FINAL_ONLY_SAVE); after the end they are in model.pt
                for cand in (os.path.join(out_dir, "model.pt"),
                             os.path.join(out_dir, "model_train.pt")):
                    if os.path.exists(cand):
                        mt = int(os.path.getmtime(cand))
                        if mt > last:
                            _publish(time.strftime("%H%M%S"))
                            last = mt
                        break
            except Exception as e:
                print(f"[prior] watcher error: {e}", flush=True)
            stop.wait(120)

    stop_evt = threading.Event()
    th = threading.Thread(target=_watcher, args=(stop_evt,), daemon=True)
    th.start()

    env = dict(os.environ, X1_FINAL_ONLY_SAVE="1")
    r = subprocess.run(
        [sys.executable, "tools/diffusion_model/train_tinymdm.py",
         "--cfg_path", "tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml",
         "--out_dir", out_dir, "--device", "cuda"],
        env=env)
    stop_evt.set()
    th.join(timeout=5)
    print(f"[prior] train exit {r.returncode}", flush=True)

    final = os.path.join(out_dir, "model.pt")
    ok = r.returncode == 0 and os.path.exists(final)

    # publish final under a UNIQUE name the SDK has never seen.
    # Numeric name `model_999999.pt` matches the platform checkpoint index
    # (model_{checkpoint}.pt) so it appears in `gm task model list`;
    # the fresh-dir copy rides the first-detection upload path.
    if ok:
        shutil.copy2(final, os.path.join(exp_dir, "model_999999.pt"))
        print(f"[prior] published {final} -> model_999999.pt", flush=True)
        fresh = os.path.join("output", "x1_smp_prior_final")
        os.makedirs(fresh, exist_ok=True)
        shutil.copy2(final, os.path.join(fresh, "model.pt"))
        print(f"[prior] published {final} -> {fresh}/model.pt", flush=True)

        # post-train self-check: reload final weights and evaluate loss
        check = r'''
import sys, os
sys.path.insert(0, "mimickit")
sys.path.insert(0, "tools/diffusion_model")
import torch, yaml
torch.manual_seed(0)
config = yaml.safe_load(open("tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml"))
env_config = yaml.safe_load(open(config["env_config"]))
from motion_prior_dataset import MotionPriorData
from learning.tinymdm.tinymdm_model import TinyMDMModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = MotionPriorData(config, device)
obs_space = ds.get_obs_space()
config["input_dim"] = obs_space.shape[-1]
config["input_channel"] = int(config["input_dim"] / env_config["num_disc_obs_steps"])
C = config["input_channel"]
m = TinyMDMModel(config, device)
m.load_state_dict(torch.load("''' + out_dir + r'''/model.pt", map_location=device))
m.eval()
losses = []
with torch.no_grad():
    for _ in range(3):
        B = 256
        x_raw = ds.fetch_obs_demo(B)
        x = m.normalize(x_raw.reshape(B, -1, C)).reshape(B, -1)
        losses.append(float(m(x)))
print("[prior] FINAL_LOSS %.4f (expect ~0.03; the stale iter-2k artifact was ~0.18)"
      % (sum(losses) / len(losses)), flush=True)
'''
        rc = subprocess.run([sys.executable, "-c", check])
        if rc.returncode != 0:
            print("[prior] final self-check crashed", flush=True)

    print("[prior] RESULT:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
