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
        print("PD pre-convert", flush=True)
        r = subprocess.run([sys.executable, "scripts_remote/convert_x1_usd.py"])
        if r.returncode != 0:
            print("PD RESULT: FAIL (convert)")
            sys.exit(1)
        print("PD post-convert", flush=True)
    else:
        print("PD usd-exists", flush=True)

    # ---- USD side: DriveAPI audit IN A SUBPROCESS. Importing pxr before
    # AppLauncher boots corrupts Kit's USD runtime (documented) -> the env
    # build later in THIS process would crash natively.
    audit_code = r'''
from pxr import Usd, UsdPhysics
import sys
st = Usd.Stage.Open(sys.argv[1])
n_drive = 0
sample = []
for prim in st.Traverse():
    drv = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drv and prim.IsA(UsdPhysics.RevoluteJoint):
        n_drive += 1
        if len(sample) < 5:
            sample.append((prim.GetName(), drv.GetStiffnessAttr().Get(),
                           drv.GetDampingAttr().Get(),
                           drv.GetMaxForceAttr().Get()))
print(f"PD usd_drive_joints={n_drive}")
for name, kp, kd, mx in sample:
    print(f"PD usd {name}: kp={kp} kd={kd} maxforce={mx}")
'''
    r_audit = subprocess.run([sys.executable, "-c", audit_code, usd],
                             capture_output=True, text=True)
    print(r_audit.stdout.strip())

    # ---- runtime side: build env and read actuator gains (with 1 retry -
    # occasional native Kit crashes happen at app startup on some nodes)
    import torch
    import envs.env_builder as env_builder
    print("PD pre-env", flush=True)
    env = None
    for attempt in range(2):
        try:
            env = env_builder.build_env(
                "data/envs/smp_x1_env.yaml",
                "data/engines/isaac_lab_engine.yaml",
                num_envs=4, device="cuda:0", visualize=False, record_video=False)
            break
        except Exception as e:
            print(f"PD env-build attempt {attempt} failed: {e}", flush=True)
            if attempt == 1:
                raise
    print("PD post-env", flush=True)

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

    # USD gains are expected to be 0 (converter drops them); the ENGINE
    # injection is what matters at runtime
    ok = float(kp.min()) > 0
    print(f"PD RESULT: {'PASS' if ok else 'FAIL'} "
          f"(runtime kp_min={float(kp.min())}, kp_max={float(kp.max())})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
