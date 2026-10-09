"""Twin ground-friction sweep for a contact-DR-trained policy.

The policy was trained across a friction family (0.7-1.4x). If the
MuJoCo twin at nominal 1.0 is simply the WORST point of that family for
it, survival at some other friction would show the policy did acquire
contact robustness that the nominal criterion hides. Sweeps multistart
mean survival vs twin ground+foot friction scale.

Run: .venv/bin/python tools/sim2sim/sweep_twin_friction.py <policy.pt>
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.sim2sim.sim2sim_x1 import (  # noqa: E402
    X1Sim, parse_x1_xml, parse_urdf_velocity, load_policy,
    normalize, unnormalize, actor_forward)
import tools.sim2sim.sim2sim_multistart as ms  # noqa: E402
import pickle  # noqa: E402


def run_friction(policy, fr_s, n_starts=8):
    import tools.sim2sim.sim2sim_x1 as sx
    orig_init = X1Sim.__init__

    def patched(self, spec):
        orig_init(self, spec)
        for g in range(self.m.ngeom):
            f = self.m.geom_friction[g]
            self.m.geom_friction[g] = (fr_s * f[0], f[1], f[2])

    X1Sim.__init__ = patched
    try:
        # replicate multistart loop compactly
        pkl = REPO / "data/motions/x1_v3/x1_run1_subject5_seg0.pkl"
        frames = np.array(pickle.load(open(pkl, "rb"))["frames"])
        fps = 30.0
        idx = np.linspace(0, len(frames) - 2, n_starts).astype(int)
        surv = []
        for fi in idx:
            sim = X1Sim(policy["spec"] if "spec" in policy else
                        _spec_cache[0])
            sim.set_init(frames[fi],
                         frames[fi + 1] if fi + 1 < len(frames) else None,
                         fps=fps)
            steps = 0
            for _ in range(300):
                obs = sim.build_obs()
                na = actor_forward(policy, normalize(policy, obs, "obs"))
                sim.apply_action(unnormalize(policy, na, "a"))
                sim.step_sim()
                steps += 1
                z = sim.d.qpos[2]
                if z < 0.35:
                    break
            surv.append(steps)
        return float(np.mean(surv)) / fps, max(surv) / fps
    finally:
        X1Sim.__init__ = orig_init


_spec_cache = [None]


def main():
    path = sys.argv[1]
    policy = load_policy(path)
    spec = parse_x1_xml()
    vel = parse_urdf_velocity()
    spec["vel_lim"] = np.array([vel[n] for n in spec["names"]])
    _spec_cache[0] = spec

    for fr_s in (0.6, 0.8, 1.0, 1.2, 1.4):
        try:
            mean_s, max_s = run_friction(policy, fr_s)
            print(f"twin friction x{fr_s:.1f}: mean {mean_s:.2f}s "
                  f"max {max_s:.2f}s", flush=True)
        except Exception as e:
            print(f"twin friction x{fr_s:.1f}: FAIL {e}", flush=True)


if __name__ == "__main__":
    main()
