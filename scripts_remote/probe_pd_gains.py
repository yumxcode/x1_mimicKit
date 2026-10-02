"""Verify PD gains actually configured in the training articulation + USD.

Builds the smp_x1 env (4 envs, motors enabled) and prints:
  - actuator.stiffness / damping / effort limits head rows
  - USD DriveAPI joint count + sample stiffness/damping values
Exit 0 iff every joint has nonzero stiffness.
"""
import glob
import os
import sys


def find_repo_root():
    starts = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for start in starts:
        d = start
        for _ in range(6):
            if (os.path.isdir(os.path.join(d, "mimickit"))
                    and os.path.isdir(os.path.join(d, "tools"))):
                return d
            p = os.path.dirname(d)
            if p == d:
                break
            d = p
    for d in sorted(glob.glob("/workspace/*")) + ["/workspace"]:
        if (os.path.isdir(os.path.join(d, "mimickit"))
                and os.path.isdir(os.path.join(d, "tools"))):
            return d
    raise RuntimeError("repo root not found")


def main():
    repo = find_repo_root()
    os.chdir(repo)
    sys.path.insert(0, os.path.join(repo, "mimickit"))
    sys.path.insert(0, repo)

    import subprocess
    usd = os.path.join(repo, "data/assets/x1/x1.usd")
    if not os.path.exists(usd):
        r = subprocess.run([sys.executable, "scripts_remote/convert_x1_usd.py"])
        if r.returncode != 0:
            print("PD RESULT: FAIL (convert)")
            sys.exit(1)

    # ---- USD side: DriveAPI audit
    from pxr import Usd, UsdPhysics
    st = Usd.Stage.Open(usd)
    n_drive = 0
    sample = []
    for prim in st.Traverse():
        drv = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drv and prim.IsA(UsdPhysics.RevoluteJoint):
            n_drive += 1
            if len(sample) < 5:
                kp = drv.GetStiffnessAttr().Get()
                kd = drv.GetDampingAttr().Get()
                mx = drv.GetMaxForceAttr().Get()
                sample.append((prim.GetName(), kp, kd, mx))
    print(f"PD usd_drive_joints={n_drive}")
    for name, kp, kd, mx in sample:
        print(f"PD usd {name}: kp={kp} kd={kd} maxforce={mx}")

    # ---- runtime side: build env and read actuator gains
    import torch
    import envs.env_builder as env_builder
    env = env_builder.build_env(
        "data/envs/smp_x1_env.yaml",
        "data/engines/isaac_lab_engine.yaml",
        num_envs=4, device="cuda:0", visualize=False, record_video=False)

    eng = env._engine
    char_id = env._get_char_id()
    obj = eng._objs[char_id]
    act = obj.actuators["actuators"]
    kp = eng._to_torch(act.stiffness, eng._device)[0]
    kd = eng._to_torch(act.damping, eng._device)[0]
    print(f"PD runtime kp head: {kp[:8].tolist()}")
    print(f"PD runtime kd head: {kd[:8].tolist()}")
    print(f"PD runtime kp min/max: {float(kp.min())}/{float(kp.max())}")

    dof_low, dof_high = eng.get_obj_dof_limits(0, char_id)
    print(f"PD dof_limits low head: {dof_low[:5].tolist()}")
    print(f"PD dof_limits high head: {dof_high[:5].tolist()}")

    ok = float(kp.min()) > 0 and n_drive >= 29
    print(f"PD RESULT: {'PASS' if ok else 'FAIL'} (kp_min={float(kp.min())}, drives={n_drive})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
