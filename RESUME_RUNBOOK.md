# X1 SMP 恢复 Runbook（充值后从这里开始）

> 状态快照：2026-10-07 哨兵停止时——nominal-final（唯一奖励持续爬升配置：
> pos + effort 限幅 10–180Nm + 无随机化）在 TASK_20261007_022 达 300 iters
> （smp_reward 0.20→0.24 仍在爬）后被 16/16 账号余额耗尽打断。
> 本地 warm 权重：`data/models/smp_policies/x1_policy_10k8.pt`（已在 main）。

## 0. 前置
```bash
cd /Users/yumx/code/x1_mimicKit
KEY=$(cat .gmhome/key16.txt)   # 或任一已充值账号的 key 文件
export HOME="$PWD/.gmhome" XDG_CONFIG_HOME="$PWD/.gmhome/.config"
GM_API_KEY="$KEY" gm auth whoami    # 确认在用账号
```

## 1. 接力训练（每轮 3500 iters，~2.6h，可能被抢占→重复 copy）
```bash
TID=$(GM_API_KEY="$KEY" gm task copy --data \
  '{"taskId":"TASK_20261007_022","projectId":"<该项目key下的projectId>","taskName":"x1-nominal-rN"}' \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['taskId'])")
GM_API_KEY="$KEY" gm task run --task-id "$TID"
```
> launcher 自动选 `data/models/smp_policies/` 下名字解析迭代数最大的 .pt
> （入库命名必须 `x1_policy_<累计k数>.pt` 且 > 10k8）。

## 2. 收割（每轮结束/被抢占后）
```bash
GM_API_KEY="$KEY" gm task model list --task-id "$TID"   # 取最新 checkpoint 的 policUrlDown
curl -sL "<url>" -o data/models/smp_policies/x1_policy_14k3.pt   # 累计命名
python3 -c "import torch;d=torch.load('data/models/smp_policies/x1_policy_14k3.pt',map_location='cpu');print(d['_obs_norm._count'])"  # 血统
git add data/models/smp_policies/x1_policy_14k3.pt && git commit -m "relay N" && git push
```

## 3. 判定（每轮收割后）
```bash
.venv/bin/python tools/sim2sim/sim2sim_multistart.py --model data/models/smp_policies/x1_policy_14k3.pt
```
- 目标：mean ≫0.15s 且 full-run >0/8 → 进入第 4 步
- 引擎侧阈值：smp_reward >0.3（慢跑成型；无限力矩时代站立=0.42）
- 目标总量：≥30k iters（当前累计 ~10.8k）

## 4. 达标验收（S1-S6 + 视频）
```bash
.venv/bin/python tools/sim2sim/sim2sim_x1.py --model <policy> \
  --json output/gates_final.json --video output/videos/sim2sim_final_mesh.mp4
.venv/bin/python tools/sim2sim/render_dump.py <engine_dump.pt> output/videos/isaac_final_mesh.mp4  # dump 任务产出（quat 是 xyzw，脚本内已转）
```
全 PASS → 更新 `output/X1_SMP_SIM2SIM_FINAL_REPORT.md` 与 `output/multistart.json`。

## 5. 若 30k 引擎内已慢跑但 MuJoCo 仍秒倒
换引擎 yaml 为 `data/engines/isaac_gym_contactdr.yaml`（接触参数域随机化
[0.7,1.4]×friction + restitution [0,0.2]，已实现并冒烟点：`contact rand fr`
日志行），回到第 1 步再接力 10–15k iters。
