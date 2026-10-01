"""Convert data/assets/x1/x1.xml (MJCF) to x1.usd for the IsaacLab engine.

The converter output is a multi-file asset: a top-level usda whose physics
lives behind `prepend payload = @./payloads/...@` references, plus a
payloads/ directory. This script publishes BOTH to data/assets/x1/.

Run inside an IsaacLab environment. Idempotent: skips if x1.usd AND
payloads/ already exist unless --force.
"""
import glob
import os
import subprocess
import sys


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


CONVERT_SNIPPET = r'''
import os
import shutil
import sys
import traceback

repo = sys.argv[1]

try:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app

    import isaaclab.sim as sim_utils

    cfg_cls = conv_cls = None
    for mod in ("isaaclab.sim.converters",):
        try:
            m = __import__(mod, fromlist=["MjcfConverterCfg"])
            cfg_cls = m.MjcfConverterCfg
            conv_cls = m.MjcfConverter
            break
        except (ImportError, AttributeError):
            continue
    if cfg_cls is None:
        raise ImportError("MjcfConverterCfg not found")

    target_dir = os.path.join(repo, "data/assets/x1")
    target = os.path.join(target_dir, "x1.usd")
    from pxr import Usd, UsdPhysics

    def count_on(stage):
        n_rb = n_joint = n_art = 0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                n_rb += 1
            if prim.IsA(UsdPhysics.Joint):
                n_joint += 1
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                n_art += 1
        return n_rb, n_joint, n_art

    def publish(out_path):
        # copy the top-level file AND the payloads/ dir it references
        src_dir = os.path.dirname(out_path)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(out_path, target)
        src_pl = os.path.join(src_dir, "payloads")
        dst_pl = os.path.join(target_dir, "payloads")
        if os.path.isdir(src_pl):
            if os.path.isdir(dst_pl):
                shutil.rmtree(dst_pl)
            shutil.copytree(src_pl, dst_pl)
        # open the PUBLISHED asset (references resolve relative to it)
        st = Usd.Stage.Open(target)
        dp = st.GetDefaultPrim()
        if dp and "Physics" in dp.GetVariantSets().GetNames():
            vset = dp.GetVariantSets().GetVariantSet("Physics")
            names = vset.GetVariantNames()
            order = [n for n in ("physx", "physics", "mujoco") if n in names]
            order += [n for n in names if n not in order and n != "none"]
            for name in order:
                vset.SetVariantSelection(name)
                rbs, jns, arts = count_on(st)
                print(f"CONVERT_VARIANT {name}: ({rbs},{jns},{arts})", flush=True)
                if rbs >= 30 and jns >= 29:
                    break
            else:
                raise RuntimeError("no physics variant with bodies")
            st.GetRootLayer().Save()
            # FLATTEN: bake payloads+variants into one self-contained file
            # so spawning/reference/instancing cannot drop the physics tree
            flat_tmp = target + ".flat.usda"
            st.Export(flat_tmp)
            shutil.move(flat_tmp, target)
            st = Usd.Stage.Open(target)
        return count_on(st), st

    base_kwargs = dict(
        asset_path=os.path.join(repo, "data/assets/x1/x1.xml"),
        usd_dir=target_dir,
        usd_file_name="x1.usd",
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=True,
    )
    modes = [
        {},
        {"run_multi_physics_conversion": False},
        {"run_multi_physics_conversion": False, "run_asset_transformer": False},
    ]

    def make_cfg(extra):
        kw = dict(base_kwargs)
        kw.update(extra)
        optional = ["joint_drive", "merge_fixed_joints", "fix_base",
                    "run_multi_physics_conversion", "run_asset_transformer",
                    "collision_from_visuals", "parse_mjcf"]
        for drop in range(len(optional) + 1):
            trial = dict(kw)
            for k in optional[:drop]:
                trial.pop(k, None)
            try:
                return cfg_cls(**trial)
            except TypeError:
                continue
        raise TypeError("MjcfConverterCfg kwargs failed: " + str(list(kw)))

    n_rb = n_joint = 0
    stage = None
    for mode in modes:
        try:
            cfg = make_cfg(mode)
            converter = conv_cls(cfg)
            out_path = converter.usd_path
            if not os.path.isfile(out_path):
                for meth in ("convert", "run"):
                    if hasattr(converter, meth) and callable(getattr(converter, meth)):
                        getattr(converter, meth)()
                        break
                out_path = converter.usd_path
            if not os.path.isfile(out_path):
                raise FileNotFoundError("no USD produced: " + str(out_path))
            (n_rb, n_joint, n_art), stage = publish(out_path)
            print(f"CONVERT_TRY mode={mode} -> rigid_bodies={n_rb} "
                  f"joints={n_joint} art={n_art}", flush=True)
            if n_rb >= 30 and n_joint >= 29:
                break
        except Exception as e:
            print(f"CONVERT_TRY mode={mode} failed: {e}", flush=True)

    if n_rb < 30 or n_joint < 29:
        # fallback: URDF conversion + per-joint PD drive authoring
        print("CONVERT_FALLBACK urdf route", flush=True)
        from isaaclab.sim.converters import (
            UrdfConverter as UConv, UrdfConverterCfg as UCfg)
        ukw = dict(
            asset_path=os.path.join(repo, "X1_29DOF/urdf/f1.urdf"),
            usd_dir=target_dir,
            usd_file_name="x1.usd",
            fix_base=False,
            merge_fixed_joints=False,
        )
        uc = None
        for drop in range(4):
            trial = dict(ukw)
            for k in ["merge_fixed_joints", "fix_base", "usd_file_name"][:drop]:
                trial.pop(k, None)
            try:
                uc = UCfg(**trial)
                break
            except TypeError:
                continue
        assert uc is not None, "UrdfConverterCfg kwargs failed"
        uconv = UConv(uc)
        usrc = uconv.usd_path
        if not os.path.isfile(usrc):
            raise RuntimeError("URDF converter produced no USD: " + str(usrc))

        (n_rb, n_joint, n_art), stage = publish(usrc)
        print(f"CONVERT_URDF published rigid_bodies={n_rb} joints={n_joint}",
              flush=True)

        # author per-joint PD gains from data/assets/x1/x1.xml
        import xml.etree.ElementTree as ET
        xroot = ET.parse(os.path.join(repo, "data/assets/x1/x1.xml")).getroot()
        gains, efforts = {}, {}
        for j in xroot.iter("joint"):
            gains[j.attrib["name"]] = (float(j.attrib.get("stiffness", 0)),
                                       float(j.attrib.get("damping", 0)))
        for m in xroot.find("actuator").findall("motor"):
            efforts[m.attrib["joint"]] = float(m.attrib["gear"])

        n_set = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                name = prim.GetName()
                if name in gains:
                    kp, kd = gains[name]
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    # USD angular drive is per-degree; rad gains -> deg
                    drive.GetStiffnessAttr().Set(kp * 57.29578)
                    drive.GetDampingAttr().Set(kd * 57.29578)
                    drive.GetMaxForceAttr().Set(efforts.get(name, 100.0))
                    n_set += 1
        stage.GetRootLayer().Save()
        n_rb, n_joint, n_art = count_on(stage)
        print(f"CONVERT_URDF drives_set={n_set} rigid_bodies={n_rb} "
              f"joints={n_joint}", flush=True)
        if n_rb < 30 or n_joint < 29 or n_set < 29:
            raise RuntimeError(f"URDF fallback lacks physics ({n_rb},{n_joint},{n_set})")

    print(f"CONVERT_DONE {target} rigid_bodies={n_rb} joints={n_joint}",
          flush=True)
    app.close()
except Exception:
    traceback.print_exc()
    print("CONVERT_FAILED", flush=True)
    sys.exit(1)
'''


def main():
    repo = find_repo_root()
    os.chdir(repo)
    usd = os.path.join(repo, "data/assets/x1/x1.usd")
    payloads = os.path.join(repo, "data/assets/x1/payloads")
    force = "--force" in sys.argv
    if os.path.exists(usd) and os.path.isdir(payloads) and not force:
        print(f"[convert] {usd} + payloads already exist, skip")
        sys.exit(0)

    tmp = os.path.join(repo, "scripts_remote/_convert_x1_usd_inner.py")
    with open(tmp, "w") as f:
        f.write(CONVERT_SNIPPET)
    r = subprocess.run([sys.executable, tmp, repo] + (["--force"] if force else []))
    ok = r.returncode == 0 and os.path.exists(usd)
    print(f"[convert] RESULT: {'PASS' if ok else 'FAIL'} ({usd})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
