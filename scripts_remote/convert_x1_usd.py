"""Convert data/assets/x1/x1.xml (MJCF) to x1.usd for the IsaacLab engine.

Run inside an IsaacLab environment. Idempotent: skips if the USD already
exists unless --force.
"""
import glob
import os
import subprocess
import sys
import traceback


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


CONVERT_SNIPPET = r"""
import os
import sys
import traceback

repo = sys.argv[1]
out_usd = os.path.join(repo, "data/assets/x1/x1.usd")

try:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app

    import isaaclab.sim as sim_utils

    cfg_cls = None
    for mod, name in (
            ("isaaclab.sim.converters", "MjcfConverterCfg"),
            ("isaaclab.sim.converters.mjc_converter", "MjcfConverterCfg"),
            ("isaaclab.sim.converters.mjcf_converter", "MjcfConverterCfg")):
        try:
            cfg_cls = getattr(__import__(mod, fromlist=[name]), name)
            conv_cls = getattr(__import__(mod, fromlist=["MjcfConverter"]),
                               "MjcfConverter")
            break
        except (ImportError, AttributeError):
            continue

    if cfg_cls is None:
        raise ImportError("MjcfConverterCfg not found in isaaclab.sim.converters")

    base_kwargs = dict(
        asset_path=os.path.join(repo, "data/assets/x1/x1.xml"),
        usd_dir=os.path.join(repo, "data/assets/x1"),
        usd_file_name="x1.usd",
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=True,
    )

    # try several conversion modes: the multi-physics pipeline can produce
    # empty physics layers for custom MJCFs; the single-physics legacy path
    # applies the XML's per-joint stiffness/damping directly.
    modes = [
        {"run_multi_physics_conversion": False},
        {"run_multi_physics_conversion": False, "run_asset_transformer": False},
        {},
        {"collision_from_visuals": False},
    ]

    def make_cfg(extra):
        kw = dict(base_kwargs)
        kw.update(extra)
        optional = ["joint_drive", "merge_fixed_joints", "fix_base",
                    "parse_mjcf", "create_particle_system",
                    "run_multi_physics_conversion", "run_asset_transformer",
                    "collision_from_visuals"]
        for drop in range(len(optional) + 1):
            trial = dict(kw)
            for k in optional[:drop]:
                trial.pop(k, None)
            try:
                return cfg_cls(**trial)
            except TypeError:
                continue
        raise TypeError(f"MjcfConverterCfg accepts none of: {list(kw)}")

    target = os.path.join(repo, "data/assets/x1/x1.usd")
    from pxr import Usd, UsdPhysics

    def _count_on(stage):
        n_rb = n_joint = n_art = 0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                n_rb += 1
            if prim.IsA(UsdPhysics.Joint):
                n_joint += 1
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                n_art += 1
        return n_rb, n_joint, n_art

    def _dump(stage, max_n=45):
        from collections import Counter
        types = Counter(prim.GetTypeName() for prim in stage.Traverse())
        top = " ".join(f"{k}:{v}" for k, v in types.most_common(12))
        dp = stage.GetDefaultPrim()
        root_layer = stage.GetRootLayer()
        sublayers = root_layer.subLayerPaths
        head = ""
        try:
            with open(target, errors="ignore") as f:
                head = f.read(1200).replace("\n", " ")[:1200]
        except Exception:
            pass
        print(f"CONVERT_DUMP types[{top}] default={dp.GetPath() if dp else None}"
              f" sublayers={list(sublayers)[:4]}", flush=True)
        print(f"CONVERT_HEAD {head}", flush=True)

    def _count_file(path):
        st = Usd.Stage.Open(path)
        return _count_on(st) + (st,)

    last_err = None
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
                raise FileNotFoundError(f"no USD produced at {out_path}")

            if os.path.abspath(out_path) != os.path.abspath(target):
                import shutil
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(out_path, target)

            n_rb, n_joint, n_art, stage = _count_file(target)
            print(f"CONVERT_TRY mode={mode} -> rigid_bodies={n_rb} "
                  f"joints={n_joint} articulation_roots={n_art}", flush=True)
            if n_rb < 30 or n_joint < 29:
                _dump(stage)
            if n_rb >= 30 and n_joint >= 29:
                break

            # try to fix the physics variant selection (count on the same
            # in-memory stage; save when a variant works)
            dp = stage.GetDefaultPrim()
            if dp and "Physics" in dp.GetVariantSets().GetNames():
                vset = dp.GetVariantSets().GetVariantSet("Physics")
                original = vset.GetVariantSelection()
                stats = {}
                for name in vset.GetVariantNames():
                    vset.SetVariantSelection(name)
                    rbs, jns, arts = _count_on(stage)
                    stats[name] = (rbs, jns, arts)
                    if rbs >= 30 and jns >= 29:
                        break
                else:
                    vset.SetVariantSelection(original)
                stage.GetRootLayer().Save()
                n_rb, n_joint, n_art = _count_on(stage)
                print(f"CONVERT_VARIANTS stats={stats} "
                      f"final=({n_rb},{n_joint},{n_art})", flush=True)
                if n_rb >= 30 and n_joint >= 29:
                    break
        except Exception as e:
            last_err = e
            print(f"CONVERT_TRY mode={mode} failed: {e}", flush=True)

    if not os.path.isfile(target):
        raise RuntimeError(f"all conversion modes failed; last: {last_err}")
    print("CONVERT_DONE", target, f"rigid_bodies={n_rb} joints={n_joint}",
          flush=True)
    if n_rb < 30 or n_joint < 29:
        # -------- fallback: URDF conversion + per-joint drive authoring
        print("CONVERT_FALLBACK urdf route", flush=True)
        try:
            from isaaclab.sim.converters import (
                UrdfConverter as UConv, UrdfConverterCfg as UCfg)
        except ImportError:
            raise RuntimeError(f"converted USD lacks physics ({n_rb},{n_joint}) "
                               "and URDF converter unavailable")
        ukw = dict(
            asset_path=os.path.join(repo, "X1_29DOF/urdf/f1.urdf"),
            usd_dir=os.path.join(repo, "data/assets/x1"),
            usd_file_name="x1_from_urdf.usd",
            fix_base=False,
            merge_fixed_joints=False,
        )
        uc = None
        for drop in range(5):
            trial = dict(ukw)
            for k in list(trial.keys())[3:3 + drop]:
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
            raise RuntimeError(f"URDF converter produced no USD at {usrc}")

        import shutil
        shutil.copy2(usrc, target)

        # author per-joint PD gains from data/assets/x1/x1.xml
        import xml.etree.ElementTree as ET
        xroot = ET.parse(os.path.join(repo, "data/assets/x1/x1.xml")).getroot()
        gains, efforts = {}, {}
        for j in xroot.iter("joint"):
            gains[j.attrib["name"]] = (float(j.attrib.get("stiffness", 0)),
                                       float(j.attrib.get("damping", 0)))
        for m in xroot.find("actuator").findall("motor"):
            efforts[m.attrib["joint"]] = float(m.attrib["gear"])

        stage = Usd.Stage.Open(target)
        _dump(stage)

        # select the physics variant if present (URDF importer too)
        dp = stage.GetDefaultPrim()
        if dp and "Physics" in dp.GetVariantSets().GetNames():
            vset = dp.GetVariantSets().GetVariantSet("Physics")
            for name in vset.GetVariantNames():
                vset.SetVariantSelection(name)
                rbs, jns, arts = _count_on(stage)
                print(f"CONVERT_URDF_VARIANT {name}: ({rbs},{jns},{arts})",
                      flush=True)
                if rbs >= 30 and jns >= 29:
                    break
            stage.GetRootLayer().Save()

        n_set = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                name = prim.GetName()
                if name in gains:
                    kp, kd = gains[name]
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    # USD angular drive: degrees; rad gains -> deg gains
                    drive.GetStiffnessAttr().Set(kp * 57.29578)
                    drive.GetDampingAttr().Set(kd * 57.29578)
                    drive.GetMaxForceAttr().Set(efforts.get(name, 100.0))
                    n_set += 1
        stage.GetRootLayer().Save()
        n_rb, n_joint, n_art = _count_on(stage)
        print(f"CONVERT_URDF drives_set={n_set} rigid_bodies={n_rb} "
              f"joints={n_joint}", flush=True)
        if n_rb < 30 or n_joint < 29 or n_set < 29:
            raise RuntimeError("URDF fallback also lacks physics "
                               f"({n_rb},{n_joint},{n_set})")
    print("CONVERT_DONE", target, flush=True)
    app.close()
except Exception:
    traceback.print_exc()
    print("CONVERT_FAILED", flush=True)
    sys.exit(1)
"""


def main():
    repo = find_repo_root()
    os.chdir(repo)
    usd = os.path.join(repo, "data/assets/x1/x1.usd")
    if os.path.exists(usd) and "--force" not in sys.argv:
        print(f"[convert] {usd} already exists, skip")
        sys.exit(0)

    tmp = os.path.join(repo, "scripts_remote/_convert_x1_usd_inner.py")
    with open(tmp, "w") as f:
        f.write(CONVERT_SNIPPET)
    r = subprocess.run([sys.executable, tmp, repo] +
                       (["--force"] if "--force" in sys.argv else []))
    ok = r.returncode == 0 and os.path.exists(usd)
    print(f"[convert] RESULT: {'PASS' if ok else 'FAIL'} ({usd})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
