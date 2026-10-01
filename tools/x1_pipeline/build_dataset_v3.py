"""Build dataset_x1_run_v3.yaml from the gate-PASS X1 v3 segments.

Usage:
  python tools/x1_pipeline/build_dataset_v3.py [--out data/datasets/dataset_x1_run_v3.yaml]
Only clips that PASS validate_retarget_v3 (R1-R9 + J1-J3) enter the
dataset; sprint clips get weight 1.5, run clips 1.0.
"""
import subprocess
import sys
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


def main():
    out = "data/datasets/dataset_x1_run_v3.yaml"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    lines = ["motions:"]
    n_pass = 0
    for s in SEGS:
        pkl = Path(f"data/motions/x1_v3/x1_{s}.pkl")
        if not pkl.exists():
            print(f"[dataset] {s}: MISSING")
            continue
        v = subprocess.run(
            [".venv/bin/python", "tools/x1_pipeline/validate_retarget_v3.py",
             "--csv", f"data/LAFAN1_g1/g1_segments/{s}.csv",
             "--pkl", str(pkl), "--sample_step", "1"],
            cwd=REPO, capture_output=True, text=True)
        if v.returncode != 0:
            print(f"[dataset] {s}: FAIL (excluded)")
            continue
        w = 1.5 if s.startswith("sprint") else 1.0
        lines.append(f'  - file: "data/motions/x1_v3/x1_{s}.pkl"')
        lines.append(f"    weight: {w}")
        n_pass += 1
        print(f"[dataset] {s}: PASS w={w}")
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"[dataset] {out}: {n_pass} clips")


if __name__ == "__main__":
    main()
