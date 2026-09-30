"""Cut steady-running segments from long LAFAN1_g1 CSVs.

A segment qualifies when the 1s-smoothed horizontal speed stays within
[lo, hi] m/s for >= min_dur seconds (G1 speed space). Writes trimmed CSVs
named <clip>_seg<k>.csv next to the outputs.

Usage:
  python tools/x1_pipeline/cut_segments.py --clips run1_subject2 ... \
      --out_dir data/LAFAN1_g1/g1_segments --lo 1.5 --hi 3.2 --min_dur 10 --max_segs 4
"""

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def cut(csv, lo, hi, min_dur, max_segs, out_dir, fps=30, margin=1.0):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from lib_g1 import load_csv
    fr = load_csv(csv)
    pos = fr["pos"]
    dt = 1.0 / fps
    v = np.linalg.norm(np.gradient(pos, dt, axis=0)[:, :2], axis=1)
    k = np.ones(fps) / fps  # 1s smoothing
    v = np.convolve(v, k, mode="same")
    ok = (v >= lo) & (v <= hi)
    # find runs
    segs = []
    start = None
    for i, o in enumerate(ok):
        if o and start is None:
            start = i
        elif not o and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(ok)))
    min_frames = int(min_dur * fps)
    segs = [(s, e) for s, e in segs if e - s >= min_frames]
    segs.sort(key=lambda se: se[1] - se[0], reverse=True)
    segs = segs[:max_segs]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(csv).stem
    written = []
    for k2, (s, e) in enumerate(segs):
        # trim 1s margins to avoid boundary transients
        s2, e2 = int(s + margin * fps), int(e - margin * fps)
        out = out_dir / f"{stem}_seg{k2}.csv"
        np.savetxt(out, fr["raw"][s2:e2], delimiter=",", fmt="%.6f")
        seg_v = v[s2:e2]
        written.append((str(out), e2 - s2, float(seg_v.mean()),
                        float(seg_v.max())))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out_dir", default="data/LAFAN1_g1/g1_segments")
    ap.add_argument("--lo", type=float, default=1.5)
    ap.add_argument("--hi", type=float, default=3.2)
    ap.add_argument("--min_dur", type=float, default=10.0)
    ap.add_argument("--max_segs", type=int, default=4)
    ap.add_argument("--margin", type=float, default=1.0)
    args = ap.parse_args()
    for c in args.clips:
        p = REPO_ROOT / "data/LAFAN1_g1/g1" / f"{c}.csv"
        for out, n, mv, xv in cut(p, args.lo, args.hi, args.min_dur,
                                  args.max_segs, REPO_ROOT / args.out_dir, margin=args.margin):
            print(f"{out}: {n} frames ({n/30:.1f}s) mean {mv:.2f} max {xv:.2f} m/s")


if __name__ == "__main__":
    main()
