# X1 29DOF SMP 训练 + MuJoCo Sim2Sim 最终验收报告

日期：2026-10-07　|　会话：x1_mimicKit 全新重做（用户 rev2 指令后）
结论：**sim2sim 通过标准未达成**（MuJoCo 中存活 0.10–0.15s 级全灭）；
根因链已完全定位并修复至"语义同构"，剩余瓶颈=正确物理上的训练量不足，
且 **gradmotion 号池 16 个账号余额全部耗尽**，训练无法继续。以下为证据链与交付物。

---

## 1. 通过标准 vs 实测

| 标准（用户要求） | MuJoCo 实测（最好策略 10k8） | 判定 |
|---|---|---|
| 不能倒 | multistart mean 0.10–0.15s 全灭（8 起点×全变体） | ✗ |
| 姿态顺畅自然 | 引擎内 rollout 直立稳定（z 0.62±0.007，300 步不倒）但为站立+微迈步（0.09 m/s，膝摆 0.3Hz），非慢跑 | 部分（引擎侧） |
| 脚底平稳着地 | S4 flat_stance FAIL（tilt 中位 23°） | ✗ |
| 无对地穿模 | S3 FAIL（sole 最低 -8.6mm） | ✗ |
| 跑步速度带 S6 | FAIL（倒地后无有效位移） | ✗ |

MuJoCo S1–S6 门数据存档：`.gmhome/gates_10k8.json`。

## 2. 已完成并验证的环节（全部有实证）

1. **重定向数据 7/7 过门**（R1–R9+J1–J3，含速度曲线滑窗重建 r=0.70–0.97）；
   视频验收 `output/videos/ref_motion_x1_mesh.mp4`（x1 mesh 渲染）。
2. **prior 真终版**（EMA=200000，loss 0.0283，P1–P4 过门；数据充分性论证：
   90s/2695 帧对 2.82M 参数 TinyMDM，loss 平滑饱和无过拟合——rev5 闭环）。
3. **训练引擎换代**：IsaacLab 6.1.14 指令通路损坏（3 条写入路径全断，多轮探针实证）
   → 换 isaac-gym 镜像 V000124 + 原生引擎，关节映射审计 29/29 与 MuJoCo 双胞胎一致。
4. **sim2sim 语义对齐三件套**（决定性诊断工具化）：
   - obs parity：root_rot tan-norm 残差 0.09（float32 级）；joint_tn 0.0007；
   - 执行器语义：发现并修复 **pos drive effort 无限制**（引擎肩关节 33ms 移 0.477 rad
     需 ~700Nm 等效，真机限幅 88Nm）→ 引擎强制 `dof_props['effort']=motor_effort(10–180Nm)`；
   - 开环重放（init+逐帧动作）：索引错位修正后前 3 步 mean_dof≈0.04 rad，
     13–18 步倒=接触级混沌放大（质量/摩擦/接触/初速单变量 A/B 均不改变结论）。
5. **SDK 覆盖上传 bug 修复**（X1_FINAL_ONLY_SAVE + 指纹核验），全部入库权重经
   obs_norm count 血统验证（杜绝旧会话"终版=早期快照"问题）。

## 3. 训练矩阵与 multistart 史（同一验收口径）

| 轮次 | 物理栈 | 引擎内 smp_reward | MuJoCo multistart（严判据） |
|---|---|---|---|
| f3k5 fresh | 无限力矩 | 0.42–0.49（站立拖步） | 0.15s / max 0.23 |
| r700/r2 PD-rand | 无限力矩+rand | 0.14–0.15 | 0.14–0.15s / max 0.33 |
| e3k1(10k5) | 限幅+rand | 0.12–0.13 | 0.15s / max 0.33 |
| nominal r3(14k3) | 限幅+无rand | Q4 0.276 尾 0.31 | 0.11s / max 0.17 |
| nominal r6(21k3) | 限幅+无rand | **0.30–0.31 平台（慢跑成型）**，后期 0.22–0.25 | 0.10s / max 0.13 |
| contactdr-r1(28k3) | 限幅+接触DR | 0.19 平台（随机化税） | 0.11s / max 0.20 |

宽松判据对照（仅 root_z≥0.35，tools/sim2sim/sweep_twin_friction.py）：nominal 21k3 存活
0.58–0.78s（max 1.93s）、contactdr 28k3 0.42–0.44s（摩擦平坦=DR 鲁棒性生效）——但严判据
（S1 对齐：tilt>60°/非足触地）在 1–6 步内触发，终止主因是 **tilt>60° 倾倒**（root z 尚在
0.4m、无低位体）。起始帧倾角 8.9–31.1°（正常跑步前倾），初态公平。

关键判读：PD 随机化的弱端（kp×0.4）使任务不可学（奖励塌缩），**effort 限幅+无随机化
（=MuJoCo 标称孪生同构）是唯一奖励持续爬升的配置**，但 300 iters 被平台抢占即账号耗尽。
论文量级为 30k iters 单栈；本项目累计 ~14k iters 横跨 3 个物理栈。

## 4. 交付物

- 视频三件套（x1 mesh）：`output/videos/ref_motion_x1_mesh.mp4`（参考动作）、
  `isaac_f3k5_ep0_mesh.mp4`（引擎 rollout 10s 直立）、`sim2sim_10k8_mesh.mp4`（sim2sim 尝试，倒地过程）。
- 权重：`data/models/smp_policies/`（10k5=限幅 3.5k 轮、10k8=nominal 300 轮，血统已验）。
- 诊断工具链（可复用）：`tools/sim2sim/{check_obs_parity,parity_gym_dump,replay_gym_dump,
  replay_physx_pd,sweep_servo,replay_contact_ab,replay_initvel_ab,render_dump}.py`。
- 引擎修复：`mimickit/engines/isaac_gym_engine.py`（effort 强制+可选 PD-rand）。

## 5. GMR 重定向路线判定（契约条件审查）

契约原文："如现有的重定向数据不合适，请基于 GMR 自行进行重定向"。触发条件审查：
- 现有数据 7/7 过门（R1–R9+J1–J3，含速度滑窗重建），视频验收通过；
- prior 在该数据上 loss 0.125→0.028 平滑饱和、无过拟合回升，生成样本 P2 分布差 0.0547rad——数据量与质量充分；
- 本项目 MuJoCo 未达标的根因链（执行器语义→训练量）**均与重定向数据无关**。

结论：**触发条件不成立**，现有数据合适；GMR 路线（x1_gmr_retargeted_tool/）留作数据升级备选，不改变当前瓶颈。

## 6. 资源断供证据（账号审计）

`output/balance_audit_20261007.txt`：2026-10-07 对号池全部 16 个账号逐一实测
`gm task run`（复用探针任务 TASK_20261007_026–041，脚本 scripts_remote/zero_probe.py），
**16/16 返回"账户余额不足"**，funded=0。训练接力（copy TASK_20261007_022 → 30k iters）
在充值前不可执行——这是唯一的硬外部阻塞。

## 7. 终局裁决（2026-10-10，contact-DR 轮完成后）

在累计 ~28k iters、语义对齐验证齐备的前提下：
- **引擎内**：nominal 栈已达成慢跑成型（smp_reward 0.30–0.31 平台）；
- **MuJoCo 孪生**：全部 5 个权重（10k8/10k5/14k3/21k3/28k3）multistart 0.10–0.15s 全灭
  （output/multistart.json 含逐起点原始数据）；终止主因 tilt>60° 倾倒，非初态伪影；
- **两个标准域随机化杠杆均被证伪且带奖励税**：PD-rand（kp/kd/effort U(0.4-1.5)）奖励塌至
  0.07–0.13；contact-rand（摩擦 0.7–1.4x+restitution）奖励 0.19 且 MuJoCo 存活无改善；
- **开环重放**（同 init+同动作）：前 3 步 mean_dof≈0.04 rad 后 13–18 步倾倒，接触/摩擦/伺服
  参数扫描均不改变结论——差距在 PhysX↔MuJoCo 接触动力学层，超出本算力预算可弥合范围。

**结论：sim2sim 通过标准在本技术栈（isaac_gym 隐式 pos drive 训练→MuJoCo 隐式 PD 孪生）
内未达成，且已系统性排障至引擎级差距。**

后续可行路线（按建议优先级）：
1. **显式力矩训练栈**（pd_explicit，引擎已支持且探针验证稳定）：动作=力矩，两引擎语义可
   精确对齐，从当前权重 warm 起步需 ~10–20k iters 预算；
2. IsaacLab 原生部署路径（部署侧仿真与训练同引擎，规避跨引擎鸿沟）；
3. 接受引擎内成果（引擎 rollout 视频见 output/videos/isaac_f3k5_ep0_mesh.mp4）。

## 8. 阻塞与建议下一步

**阻塞**：见 §6（16/16 账号余额耗尽，审计存档 output/balance_audit_20261007.txt）。

**multistart 原始数据**：output/multistart.json（10k8/10k5 双策略 × 8 起点逐起点
steps/s/end_z）；S1–S6 门原始输出：output/gates_10k8.json。

**恢复训练后的最小路径**（无需再排障）：
1. 充值任一账号 → copy TASK_20261007_022（nominal-final 配置已在 main，warm 10k8）；
2. 接力至 ≥30k iters，途中看 smp_reward 是否 >0.3（慢跑成型阈值，参考无限力矩时代 0.42）；
3. 每 3.5k 轮收割 multistart；判定阈值 mean ≫0.15s；
4. 若 30k 后引擎内已慢跑但 MuJoCo 仍秒倒 → 剩余为接触模型级差距，需引入接触参数域随机化
   （solref/friction 已有 A/B 工具）或降低验收口径（如实机部署侧再用 IsaacLab 微调）。
