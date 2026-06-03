# Stage 5 — 真机 A/B + transfer-chain 诊断

> 配套：`STAGE3B_RESULTS.md`(Stage 3 + 真机 de-risk）/ `STAGE4_RESULTS.md`(sim transfer audit）
> 脚本：`lerobot/common/policies/act/refine_deploy.py`(部署 wrapper) / `scripts/eval_smoothness_realrobot.py` / `scripts/stage5_transfer_diagnosis.py`
> 数据：`eval_refine_A`(纯 ACT) / `eval_refine_B`(ACT+RefineNet+blend)，各 8 episode，真机 Piper + TSND IMU，pick-place 胶带卷。

## 部署

RefineNet 经 `RefineNetDeployWrapper` drop-in 接入 control_robot：ACT 生成 100 步 chunk →
RefineNet 残差精修 + chunk 边界续接 blend → receding-horizon 逐步下发。history（action/imu）
复用部署侧 history_buffer；current_state 由 obs.state(mdeg)→rad→z-score（首跑实测 |z|max=1.4 正常）。
env 门控：`REFINE_CKPT` 设则启用(B)，不设则纯 ACT(A)。ckpt = λ_hf=0.10 best(ep68)。

## 主结果 1 — A/B 平滑对比（eval_smoothness_realrobot, fraction 口径）

| 指标 | A 纯ACT | B refined | Δ | Welch p |
|---|---|---|---|---|
| Action RMS jerk | 224.5±91 | 140.7±51 | **−37.3%** | 0.058 |
| Action hf% (>5Hz) | 0.0005 | 0.0003 | −41.6% | 0.103 |
| IMU accel RMS | 3.16±0.50 | 2.86±0.28 | −9.4% | 0.198 |
| IMU gyro RMS | 0.280 | 0.268 | −4.3% | 0.70 |
| IMU hf% (>5Hz) | 0.288±0.032 | 0.295±0.022 | +2.5% | 0.63 |

初看：action 大幅平滑，但 IMU hf% 持平 → 易误判"不传递"。**但 hf% 是 fraction，有分母陷阱**，
故改用绝对 hp_rms + 中间层重算（主结果 2）。

## 主结果 2 — transfer-chain 诊断（绝对 hp_rms, 5Hz zero-phase Butterworth, active j1/j2/j3/grip, n=8, bootstrap 95% CI）

补上中间层 joint state，把链路拆开：**commanded action → measured joint state(电机反馈) → 结构 → IMU**。

| 链路层 | full B/A | 显著段 |
|---|---|---|
| commanded **action** | **0.60** | full/approach/carry 均显著（CI 排除 0） |
| → measured **joint state** | **0.82** | carry 0.70 *、approach 0.15 * 显著 |
| → **IMU accel** | **0.93** | **carry 0.92 显著**（CI [−0.18,−0.019]） |
| → IMU gyro | 0.97 | 不显著 |

**coherence (5-15Hz)**：action-state **0.88**、state-IMU_accel **0.86** → 三层**物理强耦合**。
**IMU accel PSD**：0-5Hz 56% / 5-10Hz 31% / 10-15Hz 13%（44% 能量在 >5Hz）。

## 结论（措辞收敛，不宣告"解耦"）

1. **传递链真实但逐级衰减**：action −40%（显著）→ joint state −18~30%（显著）→ IMU accel −7~8%（**carry 段显著**）→ gyro 持平。高 coherence 证明 IMU 高频由动作/运动驱动，**非独立结构噪声**。
2. **"IMU hf% 持平"是 fraction 分母陷阱**：绝对 IMU accel hp_rms 确实下降（carry 显著）；fraction 因总能量同变而未动。
3. **瓶颈 = 双级衰减**：
   - action→state（0.60→0.82）：**低层控制器/执行**吞掉约一半收益（控制器未完全跟踪被平滑的指令）；
   - state→IMU（0.82→0.93）：**结构/传感器**路径再衰减。
4. **chunk 边界**仍是隐患：boundary 段 B/A>1（refined 略差，方差大不显著），deploy-blend 在真机上未完全压住——与闭环 eval 一致。

## 统计与 guardrail 注记

- n=8、A/B 非配对录制 → 用 bootstrap CI / Welch，措辞为"趋势/在该段显著"，不宣称全局严格显著。
- action jerk −37% (p=0.058) = 强趋势；IMU hf% +2.5% (p=0.63) = **无改善证据，但不等于证明无作用**。
- grasp 成功率需人工确认（脚本仅给 gripper 事件数 proxy）——作为任务能力 guardrail，B 不应明显低于 A。

## 论文可写结论（候选）

> RefineNet 在冻结 ACT 之上显著平滑指令动作（jerk −37%），该平滑**可测量地逐级传递**到真实关节运动（−18~30%，显著）直至 IMU 加速度（carry 段 −8%，显著），三层高 coherence 证明物理耦合。但传递**强烈衰减**，瓶颈在**低层控制器跟踪**与**结构/传感器路径**：指令平滑不自动等量转化为 IMU 平滑，**低层执行动力学是不可忽略的瓶颈**。闭环部署还需显式处理 chunk 边界。

## 下一步（待定）

- 暂不追加实验、不重构为"解耦"。
- 可选：(i) 补录至 n≈15-20 使 full-episode IMU accel 达显著；(ii) 诊断低层控制器跟踪误差（action vs state 残差谱）；(iii) 强化 chunk 边界处理（训练期 L_boundary）。

## 主结果 3 — 低层控制器跟踪诊断（2026-06-03, `scripts/stage5_controller_diagnosis.py`）

把 Stage5 的"action→state 丢一半"做频率分辨(臂关节 j1/j2/j3, commanded action 与 measured state 均换算到 rad):

| 频段 | 控制器增益 √(Ps/Pa)（A组） | coh²(指令解释比) | 指令无关地板占 state |
|---|---|---|---|
| 0–5Hz | 1.008（完美跟踪） | 0.34 | 66% |
| 5–10Hz | **0.198** | 0.09 | **91%** |
| 10–15Hz | **0.115** | 0.08 | **92%** |

A→B: 指令高频(5-15Hz) B/A=**0.282**（RefineNet 砍 72%），但 measured-state 高频 B/A=**0.852**（仅降 15%）。

**机制定论（两铁证）**:
1. **低层位置控制器是强低通**: <5Hz 增益≈1（完美跟踪 bulk motion），>5Hz 增益仅 0.11–0.20 → 指令高频到达关节前被控制器衰减。
2. **执行态高频 ~90% 指令无关**: 5–15Hz measured-state 仅 8–9% 能被指令线性解释，91–92% 是指令无关地板（编码器量化/控制器抖/传感器）。

→ **指令平滑无法等量转化为执行/IMU 高频下降；瓶颈是低层执行动力学（控制器低通 + 指令无关地板），非精修方法。** 要进一步降 IMU 振动需在控制器层面（力矩控制/更高跟踪带宽/抑制控制器抖动）——明确 Future Work。RefineNet 在指令空间的平滑（高频 −72%）真实有效，只是被该硬件低层执行所限。

## Document History
| 日期 | 内容 |
|---|---|
| 2026-06-01 | Stage 5 真机 A/B（各8ep）+ transfer-chain 诊断：部分传递+逐级衰减（非解耦），IMU accel carry 段显著降，fraction 陷阱已纠正，瓶颈=低层执行+结构 |
| 2026-06-03 | 低层控制器跟踪诊断：控制器强低通(>5Hz增益0.11-0.20)+执行态高频~90%指令无关 → 指令平滑被 action→state 层吞掉，瓶颈机制定论为低层执行（控制器层 Future Work） |
