# X1 重定向最终链路（GMR → LAFAN1/G1 → X1 v3）脚本归集

> ⚠️ 本目录是**归集副本**（2026-09-30，main @ c3c2f3b+）。正本仍在
> `tools/x1_pipeline/` 与 `tools/gmr_to_mimickit/`——远端 agent 的任务
> 启动器、md5 锚定流程和 `sim2sim_validate.py` 等仍引用原路径，
> 原件未移动、未修改。若后续远端收工、路径要正式迁移，需同步改
> 全部 import 与 launcher 引用。

## 两条数据链

### A. GMR 链（格式转换器，来自 AMP 线）
`gmr_to_mimickit.py` — GMR (General Motion Retargeting) 输出 pickle
→ MimicKit 运动格式。GMR 主链路（AMASS→SMPLX→GMR→X1）在另一仓库
（见经验库 exp_mtv3fqeu / exp_msphaxr7）；此处仅入库了终点转换器。

### B. LAFAN1→G1→X1 链（本项目 SMP 训练数据，v1→v3 演进的最终形态）
数据源：**LAFAN1**（Ubisoft La Forge；Run 主题官方定义为
"Jogging/Running"，另有 Walk/Sprint 主题）。G1 csv
（`data/LAFAN1_g1/g1/*.csv`，服务器端专用未入库）36 列：
root pos(3)/quat xyzw(4)/29dof。

## 执行顺序（v3 最终链）

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 0 | `build_x1_assets.py` | URDF→x1.xml/x1_sim.xml（**需服务器端 X1_29DOF/urdf/f1.urdf**；sole 盒按侧镜像修复 d44ec33） |
| 1 | `cut_segments.py` | LAFAN1 G1 长片切稳态段（速度窗 1.5–3.2 m/s） |
| 2 | `retarget_v3.py` | **最终重定向器**：支撑相平底约束(2dof)+地面闭合闭环(≥+1mm)+髋抖动抑制(Hampel) |
| 3 | `validate_retarget_v3.py` | 严格门 R1-R9 + J1-J3（含 `--selftest` 自证） |
| 4 | `batch_v3.py` / `build_dataset_v3.py` | 批量跑 2-3 并组装 `dataset_x1_run_v3.yaml` |
| 5 | `render_clip_v3.py` | G1/X1 对比视频 |

## 依赖闭包（同目录文件互引即可运行）

- `retarget_v3.py` → `lib_g1`, `retarget_v2`(四元数工具/速度策略), `semmap`(SemanticMapper), `build_x1_assets`(parse_urdf_limits)
- `validate_retarget_v3.py` → `validate_retarget`(v1 基类/X1Player), `validate_retarget_v2`(joint_gates), `retarget_v3`, `lib_g1`
- `retarget_g1_x1.py`：v1 重定向器，被 `validate_retarget` 依赖（X1_DOF_ORDER）；v2 自带副本
- 外部依赖：mujoco、numpy、scipy；`lib_g1` 需 `.research/g1_menagerie.xml`（服务器端）

## 已知遗留（未解决）

**慢放烘焙**：速度适配用均匀时间膨胀（pkl `time_scale` 1.06–2.78），
sprint 族 duty/腾空物理不自洽（duty 18-20%@1m/s、腾空至 1.73s）。
建议后续重训换 1.2–2.2 m/s 慢跑源并补 duty/腾空物理门
（参考经验库待审条目：物理自洽门 + 姿态标准）。

## v1→v3 演进一句话

v1 建管线（坏 sole 盒+慢放，门失明）→ v2 修关节域忠实度
（前臂方向球面拟合/只解腿 12dof IK/支撑吸附；数据在真几何下埋地 8cm
被揭示）→ v3 修对地几何（平底/闭合/抖动，12/16 strict-PASS）。
