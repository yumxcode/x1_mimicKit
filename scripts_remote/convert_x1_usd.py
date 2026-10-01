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

    def make_cfg():
        # newer configs dropped some kwargs; degrade one at a time
        optional = ["joint_drive", "merge_fixed_joints", "fix_base",
                    "parse_mjcf", "create_particle_system"]
        for drop in range(len(optional) + 1):
            kw = dict(base_kwargs)
            for k in optional[:drop]:
                kw.pop(k, None)
            try:
                return cfg_cls(**kw)
            except TypeError:
                continue
        raise TypeError("MjcfConverterCfg accepts none of the tried kwargs")

    cfg = make_cfg()

    converter = conv_cls(cfg)
    # newer converters run on construction / first usd_path access
    out_path = converter.usd_path
    if not os.path.isfile(out_path):
        # some versions convert lazily via a method
        for meth in ("convert", "run", "__call__"):
            if hasattr(converter, meth) and callable(getattr(converter, meth)):
                getattr(converter, meth)()
                break
        out_path = converter.usd_path

    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"converter produced no USD at {out_path}")

    target = os.path.join(repo, "data/assets/x1/x1.usd")
    if os.path.abspath(out_path) != os.path.abspath(target):
        import shutil
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(out_path, target)
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
