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
    run_ts = time.strftime("%Y-%m-%d_%H-%M-%S") + "x1_smp_policy"
    exp_dir = os.path.join("logs", "x1_smp_policy", "exported_data", run_ts)
    os.makedirs(exp_dir, exist_ok=True)
    # r5 fix: SDK only picks up TOP-LEVEL unique names under output/
    # (verified channel: output/{unique}.pt; subdirs incl. exported_data and
    # int_models are NOT scanned -> r4's 37 int_models were lost in-container)
    sdk_dir = os.path.join(repo, "output")
    os.makedirs(sdk_dir, exist_ok=True)

    # r7 fix: image-335 SDK registers only the FIRST .pt in model list
    # (empirical: r4/r6 lost all but initial model.pt; image-1 tasks like
    # TASK_20260929_110 registered 20). Git-branch relay is the proven
    # fallback (x1_policy_*.pt in main arrived this way).
    relay_dir = os.path.join(repo, "relay")
    os.makedirs(relay_dir, exist_ok=True)
    relay_state = {"pushed": set()}

    def _git(cmd):
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    def _git_relay(tag):
        """Copy current model.pt to relay/ and push branch dm/weights-relay."""
        if tag in relay_state["pushed"]:
            return
        src_f = os.path.join(out_dir, "model.pt")
        if not os.path.exists(src_f):
            return
        try:
            dst = os.path.join(relay_dir, f"model_{tag}.pt")
            shutil.copy2(src_f, dst)
            cmds = [
                ["git", "config", "user.email", "relay@gradmotion"],
                ["git", "config", "user.name", "weights-relay"],
                ["git", "add", "-f", "relay/"],
                ["git", "commit", "-m", f"weights relay: {tag}"],
                ["git", "push", "-u", "origin", f"HEAD:refs/heads/dm/weights-relay-{tag}"],
            ]
            for c in cmds:
                rc, out = _git(c)
                print(f"[relay] {' '.join(c[:3])} rc={rc} {out[:160]}", flush=True)
                if rc != 0 and c[1] != "commit":
                    print(f"[relay] FAILED at {c[1]}", flush=True)
                    return
            relay_state["pushed"].add(tag)
            print(f"[relay] pushed {dst}", flush=True)
        except Exception as e:
            print(f"[relay] error: {e}", flush=True)

    def _publish_once(tag):
        """Copy the newest model snapshot into the SDK-scanned dir with a
        UNIQUE file name each time (SDK uploads only newly-detected files;
        overwriting an existing name silently skips the upload)."""
        try:
            src = os.path.join(out_dir, "model.pt")
            if os.path.exists(src):
                top = os.path.join(sdk_dir, f"model_r5_{tag}.pt")
                shutil.copy2(src, top)
                dst = os.path.join(exp_dir, f"model_{tag}.pt")
                n = 0
                while os.path.exists(dst):
                    n += 1
                    dst = os.path.join(exp_dir, f"model_{tag}_{n}.pt")
                shutil.copy2(src, dst)
                print(f"[policy] watcher published {top} (mirror {dst})", flush=True)
        except Exception as e:
            print(f"[policy] watcher error: {e}", flush=True)

    _ticks = [0]

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
            _ticks[0] += 1
            if _ticks[0] == 9:
                _git_relay("mid")

    import threading
    stop_evt = threading.Event()
    th = threading.Thread(target=_watcher, args=(stop_evt,), daemon=True)
    th.start()

    # CUDA_LAUNCH_BLOCKING only when POLICY_DEBUG=1: it costs ~3x speed
    env = dict(os.environ)
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
            cands = sorted(glob.glob(os.path.join(staged, "*.pt")))
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
    _git_relay("final")
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

    # mirror all model files to the SDK-scanned dir (r5: top-level unique
    # names so the SDK actually uploads them)
    for src in sorted(glob.glob(os.path.join(out_dir, "**", "model_*.pt"),
                                recursive=True)):
        iter_name = os.path.basename(src)
        shutil.copy2(src, os.path.join(sdk_dir, "r5_" + iter_name))
        shutil.copy2(src, os.path.join(exp_dir, iter_name))
        print(f"[policy] published {src}", flush=True)
    final = os.path.join(out_dir, "model.pt")
    if os.path.exists(final):
        shutil.copy2(final, os.path.join(sdk_dir, "model_r5_final.pt"))
        shutil.copy2(final, os.path.join(exp_dir, "model_final.pt"))
        print(f"[policy] published {final} -> model_r5_final.pt", flush=True)
    has_final = os.path.exists(final)
    n_int = len(glob.glob(os.path.join(out_dir, "**", "model_*.pt"),
                          recursive=True))
    ok = r.returncode == 0 and (has_final or n_int > 0)
    print(f"[policy] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(final={has_final} int_models={n_int})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
