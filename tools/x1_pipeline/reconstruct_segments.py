"""Reconstruct the exact source segment CSVs for the existing X1 pkls.

The original g1_segments CSVs (server-side) are missing for some clips; the
local re-cut mismatches because the local full-clip CSVs differ slightly
from the server's. For each pkl we know n_src = n_frames / time_scale; we
slide a window of that length over the full G1 csv and pick the start that
maximizes the correlation between the resampled G1 root-speed profile and
the X1 base-speed profile from the pkl (retarget preserves the timing
structure, idx = i / warp).

Usage:
  python tools/x1_pipeline/reconstruct_segments.py \
      [--pkl-dir x1_retargeted_motion] [--out data/LAFAN1_g1/g1_segments]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from lib_g1 import load_csv

FPS = 30.0


def smooth(v, win=FPS):
    k = np.ones(int(win)) / int(win)
    return np.convolve(v, k, mode="same")


def speed(pos, dt):
    v = np.linalg.norm(np.gradient(pos, dt, axis=0)[:, :2], axis=1)
    return v


def best_window(full, n_src, warp, pkl_frames):
    """Slide window, correlate G1 speed vs X1 base speed (both resampled)."""
    pos = full["pos"]
    N = len(pos)
    v_g_full = smooth(speed(pos, 1.0 / FPS))

    x_pos = pkl_frames[:, 0:3]
    v_x = smooth(speed(x_pos, 1.0 / FPS))

    n_out = len(pkl_frames)
    idx = np.clip(np.round(np.arange(n_out) / warp).astype(int), 0, n_src - 1)

    best_s, best_r = -1, -np.inf
    for s in range(0, N - n_src + 1):
        v_g = v_g_full[s:s + n_src][idx]
        if v_g.std() < 1e-6 or v_x.std() < 1e-6:
            continue
        r = np.corrcoef(v_g, v_x)[0, 1]
        if r > best_r:
            best_r, best_s = r, s
    return best_s, best_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", default="x1_retargeted_motion")
    ap.add_argument("--out", default="data/LAFAN1_g1/g1_segments")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted((REPO_ROOT / args.pkl_dir).glob("*.pkl"))
    for pf in pkls:
        with open(pf, "rb") as f:
            d = pickle.load(f)
        frames = np.array(d["frames"])
        warp = d.get("time_scale", 1.0)
        n_out = len(frames)
        n_src = int(round(n_out / warp))

        stem = pf.stem  # x1_run1_subject2_seg0
        seg_name = stem[3:]  # run1_subject2_seg0
        clip_name = seg_name.rsplit("_seg", 1)[0]
        full_path = REPO_ROOT / f"data/LAFAN1_g1/g1/{clip_name}.csv"
        if not full_path.exists():
            print(f"[recon] {seg_name}: source clip missing {full_path}")
            continue
        full = load_csv(full_path)

        s, r = best_window(full, n_src, warp, frames)
        if s < 0 or r < 0.5:
            print(f"[recon] {seg_name}: BAD MATCH r={r:.3f} s={s}")
            continue

        out_csv = out_dir / f"{seg_name}.csv"
        np.savetxt(out_csv, full["raw"][s:s + n_src], delimiter=",", fmt="%.6f")
        seg = full["raw"][s:s + n_src]
        v = smooth(speed(full["pos"][s:s + n_src], 1.0 / FPS), 10)
        print(f"[recon] {seg_name}: s={s} n={n_src} r={r:.3f} "
              f"v_mean={v.mean():.2f} v_max={v.max():.2f} -> {out_csv}")


if __name__ == "__main__":
    main()
