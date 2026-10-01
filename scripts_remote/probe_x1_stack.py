"""Remote probe: verify the x1 SMP training stack end-to-end on gradmotion.

Stages (each prints PROBE[n] PASS/FAIL + a one-line reason):
  1. char model:  load data/assets/x1/x1.xml via MJCFCharModel, assert
     29 hinge joints in X1_DOF_ORDER and frame size 6+29=35 vs the pkls.
  2. tinymdm:     30-iteration smoke train on dataset_x1_run.yaml (CPU ok).
  3. engine:      view_motion playback of an x1 pkl via the IsaacLab engine
     (headless). Verifies MJCF asset load, joint order, motion sampling.

Exit code 0 iff all PASS.
"""
import glob
import os
import subprocess
import sys


def find_repo_root():
    """Locate the x1_mimicKit repo root robustly.

    gm-run may copy/execute this script from /workspace or the script's own
    directory, so __file__-relative paths can be wrong. Search upward for a
    dir containing both mimickit/ and tools/, then scan /workspace/*.
    """
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
    raise RuntimeError("x1_mimicKit repo root not found from "
                       + repr(starts))


REPO = find_repo_root()
print(f"[probe] repo root: {REPO}", flush=True)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "mimickit"))
sys.path.insert(0, REPO)

RESULTS = []


def stage(name, fn):
    try:
        msg = fn()
        RESULTS.append((name, True, msg))
        print(f"PROBE[{name}] PASS {msg}", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        RESULTS.append((name, False, str(e)))
        print(f"PROBE[{name}] FAIL {e}", flush=True)


def s1_char_model():
    import torch
    import anim.motion as motion
    from anim.mjcf_char_model import MJCFCharModel

    cm = MJCFCharModel(torch.device("cpu"))
    cm.load("data/assets/x1/x1.xml")
    n_joints = cm.get_num_joints()
    dof_size = cm.get_dof_size()
    assert n_joints - 1 == 29, f"expected 29 joints, got {n_joints - 1}"
    assert dof_size == 29, f"expected dof size 29, got {dof_size}"

    import pickle
    import numpy as np
    names = [cm.get_body_names()[0]]
    with open("data/motions/x1_v3/x1_run1_subject5_seg0.pkl", "rb") as f:
        d = pickle.load(f)
    frames = np.array(d["frames"])
    assert frames.shape[-1] == 6 + 29, f"frame width {frames.shape[-1]} != 35"
    return f"29 joints, dof 29, frame width {frames.shape[-1]}, bodies={len(cm.get_body_names())}"


def s2_tinymdm():
    import yaml

    r0 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "diffusers>=0.36.0"],
        capture_output=True, text=True, timeout=900)
    if r0.returncode != 0:
        raise RuntimeError("pip install diffusers failed: "
                           + (r0.stdout + r0.stderr)[-300:])

    with open("tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["num_iterations"] = 30
    cfg["output_iter"] = 100
    cfg["batch_size"] = 128
    cfg["num_samples_stat"] = 256
    with open("/tmp/tinymdm_probe.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    r = subprocess.run(
        [sys.executable, "tools/diffusion_model/train_tinymdm.py",
         "--cfg_path", "/tmp/tinymdm_probe.yaml", "--out_dir", "/tmp/x1_prior_probe"],
        capture_output=True, text=True, timeout=1800)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    assert r.returncode == 0, "train_tinymdm exit " + str(r.returncode) + ": " + " | ".join(tail)
    return "tinymdm 30 iters OK: " + " | ".join(tail[-1:])


def s3_engine_view_motion():
    # convert MJCF -> USD for the IsaacLab engine (idempotent)
    r0 = subprocess.run(
        [sys.executable, "scripts_remote/convert_x1_usd.py"],
        capture_output=True, text=True, timeout=900)
    if r0.returncode != 0:
        keep = [l for l in (r0.stdout + r0.stderr).splitlines()
                if ("Error" in l or "error" in l or "Traceback" in l
                    or "File \"" in l or "CONVERT" in l)]
        raise RuntimeError("mjcf->usd convert failed: " + " | ".join(keep[-14:]))

    # multi-env run to exercise _build_envs cloning, _clone_obj_prim
    # positions/orientations and inter-env collision filtering
    r = subprocess.run(
        [sys.executable, "mimickit/run.py", "--arg_file",
         "args/view_motion_x1_args.txt", "--num_envs", "4"],
        capture_output=True, text=True, timeout=1200)
    out = r.stdout + r.stderr
    tail = out.strip().splitlines()[-3:]
    assert r.returncode == 0, "run.py exit " + str(r.returncode) + ": " + " | ".join(tail)
    assert "Mean Episode Length" in out, \
        "view_motion did not complete (no Mean Episode Length): " + " | ".join(tail)
    return "view_motion num_envs=4 completed"


def s4_isaacgym_available():
    try:
        import isaacgym  # noqa: F401
        return "isaacgym importable (legacy Isaac Gym present)"
    except Exception as e:
        return f"isaacgym NOT available ({type(e).__name__}: {e}) - use isaaclab"


def main():
    stage("char", s1_char_model)
    stage("tinymdm", s2_tinymdm)
    stage("engine", s3_engine_view_motion)
    print("INFO isaacgym:", s4_isaacgym_available(), flush=True)
    ok = all(r[1] for r in RESULTS)
    print("PROBE_RESULT:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
