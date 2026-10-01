# X1 29DOF SMP training (jogging only)

Train the SMP diffusion prior (TinyMDM) on the gate-passing X1 jogging
dataset, then train the SMP policy with the prior, and finally run MuJoCo
sim2sim locally.

## Pipeline

1. Build assets (already done, rerunnable):
   ```
   python tools/x1_pipeline/build_x1_assets.py
   ```
   -> data/assets/x1/x1.xml (Isaac Gym), x1_sim.xml (MuJoCo sim2sim)

2. Data: `data/datasets/dataset_x1_run.yaml` — only clips that pass
   `tools/x1_pipeline/validate_retarget_v3.py` (gates R1-R9 + J1-J3)
   enter the dataset.

3. Train prior (remote GPU or local):
   ```
   python tools/diffusion_model/train_tinymdm.py --cfg_path tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml --out_dir output/x1_smp_prior
   ```

4. Train policy (remote):
   ```
   python mimickit/run.py --arg_file args/smp_x1_args.txt
   ```

5. Test / sim2sim:
   ```
   python mimickit/run.py --arg_file args/smp_x1_args.txt --num_envs 1 --visualize false --mode test --model_file <policy.pt>
   ```
