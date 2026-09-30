"""Batch: retarget all run/sprint segments with the v3 pipeline, validate,
and report. Runs N clips in parallel (FK/IK is single-thread per process).

Usage:
  python tools/x1_pipeline/batch_v3.py [--only name1,name2]
      [--out-dir data/motions/x1_v3] [--jobs 4] [--skip-validate]
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEGS = [
    "run1_subject2_seg0", "run1_subject5_seg0", "run1_subject5_seg1",
    "run1_subject5_seg2", "run2_subject1_seg0", "run2_subject1_seg1",
    "run2_subject1_seg2", "run2_subject4_seg0", "run2_subject4_seg1",
    "run2_subject4_seg2", "sprint1_subject2_seg0", "sprint1_subject4_seg0",
    "sprint1_subject4_seg1", "sprint1_subject4_seg2", "sprint1_subject4_seg3",
    "sprint1_subject4_seg6",
]


def run_one(s, out_dir, skip_validate):
    csv = f"data/LAFAN1_g1/g1_segments/{s}.csv"
    pkl = f"{out_dir}/x1_{s}.pkl"
    r = subprocess.run(
        [".venv/bin/python", "tools/x1_pipeline/retarget_v3.py",
         "--csv", csv, "--out", pkl],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        return s, "retarget-crash", r.stdout[-1500:] + r.stderr[-1500:]
    if skip_validate:
        return s, "RETAGETED(not validated)", ""
    v = subprocess.run(
        [".venv/bin/python", "tools/x1_pipeline/validate_retarget_v3.py",
         "--csv", csv, "--pkl", pkl, "--sample_step", "1"],
        cwd=REPO, capture_output=True, text=True)
    tag = "PASS" if v.returncode == 0 else "FAIL"
    return s, tag, v.stdout[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out-dir", default="data/motions/x1_v3")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    segs = ([s for s in SEGS if s in args.only.split(",")]
            if args.only else SEGS)
    print(f"batch_v3: {len(segs)} clips, jobs={args.jobs}")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(run_one, s, args.out_dir, args.skip_validate): s
                   for s in segs}
        results = {}
        for fut in sorted(futures, key=lambda f: segs.index(futures[f])):
            s = futures[fut]
            name, tag, log = fut.result()
            results[name] = tag
            print(f"[batch] {name}: {tag}", flush=True)
            if tag not in ("PASS",):
                print(log[-1200:], flush=True)
    print("\n===== SUMMARY =====")
    for s, r in results.items():
        print(f"  {s}: {r}")
    ok = sum(1 for r in results.values() if r == "PASS")
    print(f"  {ok}/{len(results)} PASS")


if __name__ == "__main__":
    main()
