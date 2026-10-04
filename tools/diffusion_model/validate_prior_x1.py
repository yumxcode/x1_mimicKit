"""Validate the X1 TinyMDM SMP prior: accuracy + data sufficiency.

Local (CPU). Steps:
  1. load prior config + dataset + EMA weights (data/models/smp_priors/x1_prior.pt)
  2. sample N windows (sample_ema, same API as train_tinymdm.test)
  3. convert to motion frames; FK via MJCFCharModel
  4. Gates:
     P1 diversity   - pairwise std of DOF trajectories (mode collapse check)
     P2 dist match  - per-DOF mean |angle diff| generated vs training < 0.3 rad
     P3 FK sanity   - root height in [0.35, 1.1] m; foot z in [-0.05, 0.35] m
     P4 foot gait   - foot clearance oscillation present (max-min >= 4 cm in
                      windows containing a swing)
  5. dump sample pcls + report json

Usage: .venv/bin/python tools/diffusion_model/validate_prior_x1.py \
          [--samples 24] [--out output/x1_prior_val]
"""
import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mimickit"))
sys.path.insert(0, str(REPO / "tools/diffusion_model"))
os.chdir(REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--out", default="output/x1_prior_val")
    ap.add_argument("--prior", default="data/models/smp_priors/x1_prior.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    torch.manual_seed(0)
    import yaml

    device = torch.device("cpu")

    with open("tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml") as f:
        config = yaml.safe_load(f)
    with open(config["env_config"]) as f:
        env_config = yaml.safe_load(f)

    from motion_prior_dataset import MotionPriorData
    dataset = MotionPriorData(config, device)
    obs_space = dataset.get_obs_space()
    num_obs_steps = env_config["num_disc_obs_steps"]
    config["input_dim"] = obs_space.shape[-1]
    config["input_channel"] = int(config["input_dim"] / num_obs_steps)
    num_frames = obs_space.shape[-1] // config["input_channel"]
    print(f"[prior-val] obs dim {config['input_dim']} channel "
          f"{config['input_channel']} frames/window {num_frames}")

    from learning.tinymdm.tinymdm_model import TinyMDMModel
    model = TinyMDMModel(config, device)
    sd = torch.load(args.prior, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[prior-val] loaded {args.prior} ({len(sd)} tensors)")

    # ---------- training data reference stats
    train_pkls = []
    for mf in yaml.safe_load(open("data/datasets/dataset_x1_run.yaml"))["motions"]:
        train_pkls.append(mf["file"])
    train_dofs, train_root_z, train_feet = [], [], []
    for p in train_pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        fr = np.array(d["frames"])
        train_dofs.append(fr[:, 6:35])
        train_root_z.append(fr[:, 2])
    train_dofs = np.concatenate(train_dofs, 0)
    train_dof_mean = train_dofs.mean(0)
    train_dof_std = train_dofs.std(0)
    train_root_z_all = np.concatenate(train_root_z)
    print(f"[prior-val] training data: {len(train_pkls)} clips, "
          f"{train_dofs.shape[0]} frames, "
          f"{train_dofs.shape[0]/30:.1f}s, root z "
          f"[{train_root_z_all.min():.2f},{train_root_z_all.max():.2f}]")

    # ---------- sample
    with torch.no_grad():
        gen = model.sample_ema(shape=obs_space.shape,
                               batch_size=args.samples, device=device)
        gen = model.unnormalize(gen.reshape([args.samples, num_frames, -1]))
        gen_dofs, gen_roots, gen_quats = [], [], []
        for s in gen:
            frames = dataset.convert_sample_to_frames(s).detach().numpy()
            gen_dofs.append(frames[:, 6:35])
            gen_roots.append(frames)
        gen_dofs = np.stack(gen_dofs)  # (N, T, 29)
    print(f"[prior-val] generated {gen_dofs.shape}")

    # ---------- gates
    flat = gen_dofs.reshape(-1, 29)

    # P1 diversity: std across samples of per-window DOF means
    per_sample_mean = gen_dofs.mean(axis=1)          # (N, 29)
    diversity = float(per_sample_mean.std(axis=0).mean())
    p1 = diversity > 0.02

    # P2 distribution match
    dof_diff = np.abs(flat.mean(0) - train_dof_mean)
    dof_std_ratio = flat.std(0) / np.maximum(train_dof_std, 1e-6)
    p2_mean = float(dof_diff.mean())
    p2 = bool(dof_diff.mean() < 0.3 and dof_diff.max() < 1.0)

    # P3/P4 FK sanity on generated frames
    root_z = np.stack([g[:, 2] for g in gen_roots])  # (N, T)
    p3 = bool(root_z.min() > 0.35 and root_z.max() < 1.1)

    fk = dataset._motion_lib._kin_char_model
    lfoot_z, rfoot_z = [], []
    for g in gen_roots:
        rp = torch.tensor(g[:, 0:3], dtype=torch.float32)
        rr = torch.tensor(g[:, 3:6], dtype=torch.float32)
        fr = torch.tensor(g, dtype=torch.float32)
        root_rot = torch_util_exp(rr)
        joint_rot = fk.dof_to_rot(torch.tensor(g[:, 6:35], dtype=torch.float32))
        body_pos, _ = fk.forward_kinematics(root_pos=rp, root_rot=root_rot,
                                            joint_rot=joint_rot)
        names = fk.get_body_names()
        lfoot_z.append(body_pos[:, names.index("left_ankle_roll_link"), 2].numpy())
        rfoot_z.append(body_pos[:, names.index("right_ankle_roll_link"), 2].numpy())
    lfoot_z = np.stack(lfoot_z)
    rfoot_z = np.stack(rfoot_z)
    foot_min = float(min(lfoot_z.min(), rfoot_z.min()))
    foot_max = float(max(lfoot_z.max(), rfoot_z.max()))
    p3_fk = foot_min > -0.05 and foot_max < 0.35
    p3 = bool(p3 and p3_fk)

    swing_amp = float(np.maximum((lfoot_z.max(1) - lfoot_z.min(1)).mean(),
                                 (rfoot_z.max(1) - rfoot_z.min(1)).mean()))
    p4 = swing_amp >= 0.04

    report = dict(
        prior=args.prior,
        samples=args.samples,
        train=dict(clips=len(train_pkls), seconds=round(train_dofs.shape[0] / 30, 1)),
        P1_diversity=dict(pass_=p1, cross_sample_std=round(diversity, 4)),
        P2_dist_match=dict(pass_=p2, mean_abs_diff_rad=round(p2_mean, 4),
                           max_abs_diff_rad=round(float(dof_diff.max()), 4),
                           std_ratio_median=round(float(np.median(dof_std_ratio)), 3)),
        P3_fk_sanity=dict(pass_=p3, root_z=[round(float(root_z.min()), 3),
                                            round(float(root_z.max()), 3)],
                          foot_z=[round(foot_min, 3), round(foot_max, 3)]),
        P4_foot_gait=dict(pass_=p4, swing_amplitude_m=round(swing_amp, 4)),
    )
    report["overall_pass"] = bool(p1 and p2 and p3 and p4)
    print(json.dumps(report, indent=2))

    with open(os.path.join(args.out, "prior_val.json"), "w") as f:
        json.dump(report, f, indent=2)

    # dump a few generated samples as motion pkls (mesh-renderable)
    for i in range(min(3, len(gen_roots))):
        with open(os.path.join(args.out, f"gen_{i:02d}.pkl"), "wb") as f:
            pickle.dump(dict(fps=30, frames=gen_roots[i].tolist(),
                             source="tinymdm_sample"), f)
    print(f"[prior-val] wrote samples + report to {args.out}")
    return 0 if report["overall_pass"] else 1


def torch_util_exp(exp):
    import torch
    import util.torch_util as tu
    return tu.exp_map_to_quat(exp)


if __name__ == "__main__":
    sys.exit(main())
