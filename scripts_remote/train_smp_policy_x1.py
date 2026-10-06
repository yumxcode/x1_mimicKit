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

    for mod, spec in (("diffusers", "diffusers>=0.36.0"),
                      ("gymnasium", "gymnasium")):
        try:
            __import__(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "-i",
                            "https://pypi.tuna.tsinghua.edu.cn/simple",
                            spec])

    prior_model = os.path.join(repo, "data/models/smp_priors/x1_prior.pt")
    prior_cfg = os.path.join(
        repo, "tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml")
    missing = [p for p in (prior_model, prior_cfg) if not os.path.isfile(p)]
    if missing:
        print(f"[policy] FAIL prior artifacts missing: {missing}", flush=True)
        sys.exit(1)

    # isaac_gym engine consumes the MJCF directly; no USD conversion

    out_dir = "output/x1_smp_policy"
    run_ts = time.strftime("%Y-%m-%d_%H-%M-%S") + "x1_smp_policy"
    exp_dir = os.path.join("logs", "x1_smp_policy", "exported_data", run_ts)
    os.makedirs(exp_dir, exist_ok=True)

    def _publish_once(tag):
        """Copy the newest weights into SDK-scanned dirs (unique names)."""
        try:
            src = os.path.join(out_dir, "model.pt")
            if not os.path.exists(src):
                # X1_FINAL_ONLY_SAVE: during training only int_models exist
                ints = sorted(glob.glob(
                    os.path.join(out_dir, "int_models", "model_*.pt")))
                if ints:
                    src = ints[-1]
            if not os.path.exists(src):
                return
            for base in (exp_dir,
                         os.path.join("logs", "x1_smp_policy", "gm_play")):
                os.makedirs(base, exist_ok=True)
                dst = os.path.join(base, f"model_{tag}.pt")
                n = 0
                while os.path.exists(dst):
                    n += 1
                    dst = os.path.join(base, f"model_{tag}_{n}.pt")
                shutil.copy2(src, dst)
            print(f"[policy] watcher published snapshot {tag}", flush=True)
        except Exception as e:
            print(f"[policy] watcher error: {e}", flush=True)

    def _watcher(stop):
        # publish a snapshot every 10 min so a hard kill (OOM/node loss)
        # never loses more than 10 min of training
        last = -1
        while not stop.is_set():
            try:
                src = os.path.join(out_dir, "model.pt")
                if os.path.exists(src):
                    mtime = int(os.path.getmtime(src))
                    if mtime > last:
                        _publish_once(time.strftime("%H%M%S"))
                        last = mtime
            except Exception as e:
                print(f"[policy] watcher error: {e}", flush=True)
            stop.wait(600)

    import threading
    stop_evt = threading.Event()
    th = threading.Thread(target=_watcher, args=(stop_evt,), daemon=True)
    th.start()

    # CUDA_LAUNCH_BLOCKING only when POLICY_DEBUG=1: it costs ~3x speed
    env = dict(os.environ, X1_FINAL_ONLY_SAVE="1")
    if os.environ.get("POLICY_DEBUG", "0") == "1":
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    # resume: warm-start from a checkpoint. Precedence: X1_RESUME_MODEL env,
    # a platform-mounted model_20*.pt at repo root, then the latest staged
    # policy in data/models/smp_policies/ (git-tracked relay weights).
    resume_model = os.environ.get("X1_RESUME_MODEL", "")
    if (not resume_model):
        cands = sorted(glob.glob(os.path.join(repo, "model_20*.pt")))
        if (not cands):
            staged = os.path.join(repo, "data/models/smp_policies")
            # select by iteration count parsed from the file name
            # (x1_policy_10k5.pt -> 10500); git-checkout mtimes are all
            # equal so mtime sorting degenerates to lexicographic order
            def _iters(path):
                import re as _re
                m = _re.search(r"(\d+)k(\d*)", os.path.basename(path))
                if (not m):
                    return -1
                return int(m.group(1)) * 1000 + int(m.group(2) or 0)
            cands = sorted(glob.glob(os.path.join(staged, "*.pt")), key=_iters)
        if (cands):
            resume_model = cands[-1]
    cmd = [sys.executable, "mimickit/run.py", "--arg_file",
           "args/smp_x1_args.txt"]
    if (resume_model and os.path.isfile(resume_model)):
        cmd += ["--model_file", resume_model]
        print(f"[policy] warm-start from {resume_model}", flush=True)

    r = subprocess.run(cmd, env=env)
    stop_evt.set()
    th.join(timeout=5)
    _publish_once("final")
    print(f"[policy] train exit {r.returncode}", flush=True)

    # ---- post-train survival eval (in-training physics): mean episode
    # length is the S1-equivalent metric; fall => short episode,
    # surviving 10 s => ~300 steps. Gives each relay a quality signal.
    try:
        ev = subprocess.run(
            [sys.executable, "mimickit/run.py", "--arg_file",
             "args/smp_x1_eval_args.txt", "--model_file",
             os.path.join(out_dir, "model.pt")],
            capture_output=True, text=True, timeout=1800, env=env)
        ev_out = ev.stdout + ev.stderr
        mep = [l for l in ev_out.splitlines() if "Mean Episode Length" in l]
        print(f"[policy] EVAL {' | '.join(mep) if mep else 'no metric (exit ' + str(ev.returncode) + ')'}",
              flush=True)
        if (not mep):
            keep = [l for l in ev_out.splitlines()
                    if ('File "' in l or "Error" in l)]
            print("[policy] EVAL tail: " + " | ".join(keep[-8:]), flush=True)
    except Exception as e:
        print(f"[policy] EVAL failed: {e}", flush=True)

    # mirror all model files to the SDK-scanned dir
    for src in sorted(glob.glob(os.path.join(out_dir, "**", "model_*.pt"),
                                recursive=True)):
        iter_name = os.path.basename(src)
        shutil.copy2(src, os.path.join(exp_dir, iter_name))
        print(f"[policy] published {src}", flush=True)
    final = os.path.join(out_dir, "model.pt")
    if os.path.exists(final):
        # numeric name -> indexed by gm task model list (checkpoint 999999);
        # fresh-dir copy rides the first-detection upload
        shutil.copy2(final, os.path.join(exp_dir, "model_999999.pt"))
        print(f"[policy] published {final} -> model_999999.pt", flush=True)
        fresh = os.path.join("output", "x1_smp_policy_final")
        os.makedirs(fresh, exist_ok=True)
        shutil.copy2(final, os.path.join(fresh, "model.pt"))
        print(f"[policy] published {final} -> {fresh}/model.pt", flush=True)
    has_final = os.path.exists(final)
    n_int = len(glob.glob(os.path.join(out_dir, "**", "model_*.pt"),
                          recursive=True))
    ok = r.returncode == 0 and (has_final or n_int > 0)
    print(f"[policy] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(final={has_final} int_models={n_int})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
