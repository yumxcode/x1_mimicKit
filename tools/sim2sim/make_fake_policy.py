"""Create a fake MimicKit agent state_dict for local sim2sim smoke tests.

Builds an agent-shaped state_dict with random actor weights and identity
normalizers so sim2sim_x1.py can run end-to-end without a trained model.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "mimickit")
sys.path.insert(0, ".")

from argparse import Namespace

OBS_DIM = 1 + 6 + 3 + 3 + 29 * 6 + 29 + 4 * 3  # 231
ACT_DIM = 29

torch.manual_seed(0)
W1 = torch.randn(1024, OBS_DIM) * 0.01
b1 = torch.zeros(1024)
W2 = torch.randn(512, 1024) * 0.01
b2 = torch.zeros(512)
W3 = torch.randn(ACT_DIM, 512) * 0.001
b3 = torch.zeros(ACT_DIM)

from tools.x1_pipeline.build_x1_assets import parse_urdf_limits  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

# joint order / ranges from x1.xml
root = ET.parse("data/assets/x1/x1.xml").getroot()
names, lows, highs = [], [], []
def walk(b):
    for j in b.findall("joint"):
        names.append(j.attrib["name"])
        rng = np.fromstring(j.attrib["range"], sep=" ")
        lows.append(rng[0]); highs.append(rng[1])
    for c in b.findall("body"):
        walk(c)
walk(root.find("worldbody").find("body"))
low, high = np.array(lows), np.array(highs)
mid = 0.5 * (high + low)
scale = np.maximum(np.abs(high - mid), np.abs(low - mid)) * 1.4
a_low, a_high = mid - scale, mid + scale
a_mean = mid
a_std = np.maximum(a_high - mid, mid - a_low)

sd = {
    "_model._actor_layers.0.weight": W1, "_model._actor_layers.0.bias": b1,
    "_model._actor_layers.2.weight": W2, "_model._actor_layers.2.bias": b2,
    "_model._actor_layers.4.weight": W3, "_model._actor_layers.4.bias": b3,
    "_obs_norm._mean": torch.zeros(OBS_DIM), "_obs_norm._std": torch.ones(OBS_DIM),
    "_obs_norm._mean_sq": torch.ones(OBS_DIM),
    "_a_norm._mean": torch.tensor(a_mean, dtype=torch.float32),
    "_a_norm._std": torch.tensor(a_std, dtype=torch.float32),
}
torch.save(sd, sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_x1_policy.pt")
print("saved fake policy, obs_dim", OBS_DIM)
