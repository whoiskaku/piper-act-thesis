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

## #67 真机三方 A/B — 拆开 blend vs 学到的残差(2026-06-11)

解耦 `REFINE_MODE` 后真机录制三条件(各 3 ep, **单轮非交错**, absboth ckpt=k4 Hattn):
A 裸 ACT / B 纯 blend(无残差) / C blend+absboth 残差。分析 `scripts/stage5_3way_realrobot.py`,
权威 filtfilt 5Hz hp_rms + #66 的 2-4.5Hz 中频带。图 `outputs/eval/stage5_3way_perep.png`。

**动作空间(稳健, 逐集低方差)**:
- blend 砍指令 hp_rms **−53%**(每集一致, B<A 全 3 对); 残差对指令平滑**无增量**(C≈B);
- 残差注入 2-4.5Hz 中频 **+18%** → **#66 预测的中频放大副作用在真机指令上兑现**。

**IMU 空间(N=3 欠功效)**:原始均值 B−A accel −7.6%/gyro −3.1%, C−B accel +5.9%/gyro +3.2%,
但**逐集 std(0.18–0.24)远大于差异、B/C 范围完全重叠**,且每条件 **ep0 冷启动离群**(accel 1.1–1.2 vs 0.7)。
去 ep0(N=2)趋势锐化:blend 给真机 IMU 收益(**accel −10.5% / gyro −8.6% / 中频 −19.5%**),
**残差零增量(C−B ≈0~+4%, 中频略差)**。

**结论(与全 sim 故事完全自洽)**:
1. **确定性边界 blend 是真机有效的平滑机制**(指令 −53%, IMU accel ~−10%);
2. **学到的 absboth 残差在真机 IMU 上不提供超过 blend 的增量收益** → #66 的"执行流地板/控制器瓶颈"在真机兑现(指令空间平滑真实, 但被低层执行封顶, 同 2026-06-03 诊断);
3. 残差的 **#66 中频(2-4.5Hz)放大副作用在真机 IMU 可见**(C 比 B 略高);
4. **fraction 再次误导**: action fraction 说 C 优于 B(−4%), 绝对 hp 说 C 差于 B(+4%); IMU fraction 说 blend/残差都变差(+4.8%/+9.3%) — 真机再次坐实只能信绝对口径。

**caveat**: N=3 单轮非交错 + ep0 冷启动混入; "blend 的 IMU 收益"与"残差零增量"都需 **≥9-12 交错集**确认(本次为初步)。但方向与 sim(#66 执行地板 + 中频放大)自洽。

---

## #69-real 真机四方 token 对比(residual-only)— 残差无 blend 有害(2026-06-17)

录 A(裸ACT)/ 1q(Hattn)/ M-q / RLT-AE 各 12 集,`REFINE_MODE=residual`(无 blend,= 用户"主对比 ACT vs ACT+残差"框架)。S0 未录。分析 drop 各块 ep0(冷启动),N=11。图 `outputs/eval/stage5_token_real.png`,数据 `stage5_token_real.json`。

| 指标 | A | 1q | M-q | RLT-AE |
|---|---|---|---|---|
| 指令 hp_rms | 0.0027 | +57% | +31% | +25% |
| IMU accel hp(5Hz) | 0.717 | +26% | +17% | +15% |
| IMU gyro hp(5Hz) | 0.073 | +28% | +20% | +13% |
| IMU accel 2–4.5Hz | 0.392 | +31% | +21% | +8% |
| IMU gyro 2–4.5Hz | 0.057 | +32% | +23% | +3% |

**核心结论**:
1. **无一残差变体改善 IMU;全部中性到更差**(+3~+32%)。指令 hp 也升(+25~57%)——残差在**部署流**上增加高频而非降低。
2. **变体排序 RLT-AE(伤害最小,≈裸ACT,gyro 2–4.5Hz 仅 +3% 在噪声内)< M-q < 1q(伤害最大)= 精确匹配 sim 的"保守↔激进"权衡线**:残差改动越大,真机越糟;RLT-AE 近恒等→几乎不动 IMU。
3. **根因 = 无 blend → 残差搞坏 chunk 边界**:sim 所有评估是逐 100 步 chunk 内(无边界);真机 residual-only 让相邻 chunk 边界跳变更大(smoke: 0.08–0.10 vs 裸 ACT 0.06),拼接后边界注入宽带高频**压倒** chunk 内平滑 → 净 hp 升。**坐实 blend 不是可选而是残差落地前提**(呼应 #67)。可能叠加 sim→real 输入分布漂移。
4. **方法学**:per-chunk sim 指标(#64–#69)**不预测部署流真机 IMU**;公平的"残差是否有用"必须 **blend 持平对照**(ACT+blend vs ACT+blend+残差),本批未录。
5. **任务成功率(用户手记,12 集/条件)**:**A 裸ACT 12/12(满分基线)** = RLT-AE **12/12** > M-q 11/12(失1) > 1q 10/12(失2)。**对满分基线: RLT-AE 不伤任务但近恒等无收益; M-q/1q 主动拉低抓取(且 IMU 同时恶化)。****第三根轴,排序与 sim 激进度、真机 IMU 伤害完全一致**(RLT<M-q<1q)→ **残差改动越大,IMU 越糟 + 抓取越易失败,单调耦合;residual-only 无"既降IMU又保任务"的工作点**。RLT-AE 安全(12/12+IMU近裸ACT)因近恒等几乎不干活。**token 价值重定义:RLT 重构 token→保守保任务残差(真实贡献),但不带来 IMU 平滑;1q 在真机最差**。(裸 ACT 基线成功率待补。)三轴收敛是核心证据。

**caveat**:N=11 单块非交错(各变体 std 0.08–0.20,1q/M-q 与 A 差异部分超噪声、RLT-AE 多在噪声内);未含 blend 持平对照;S0 未录。

## #70 改进 I 部署流训练 — 残差从"有害"变"安全"但非"更好";边界需跨chunk状态(2026-06-17)

针对 #69-real(per-chunk 残差真机搞坏边界)的改进 I:`sim/train_refine_stream.py` 把损失从 per-chunk 改为相邻两 chunk(K=2, deploy cadence)拼成 200 步流(含边界),absboth 在流上算。Hattn 1q cut2.0,1859 train/448 val 对。val 部署流对比:

| 模型 | 边界跳变/ACT | stream_hp/ACT | within-chunk hp/ACT | L1 |
|---|---|---|---|---|
| per-chunk Hattn(#69-real) | **2.92** | **1.42** | 0.935 | 0.0087 |
| stream Hattn(改进I) | **0.97** | **1.03** | 1.045 | 0.0092 |
| 裸 ACT | 1.0 | 1.0 | 1.0 | 0.0084 |

1. **量化了 #69-real 的根因**:per-chunk 残差让边界跳变恶化 **2.92×**、部署流 hp **+42%** → 这就是真机 IMU +13~31% 变差之源。
2. **改进 I 修好了"有害"**:stream 训练把边界拉回 0.97(≈ACT)、stream_hp 1.42→1.03 → **残差从有害变无害/安全**。
3. **但没变"更好"**:net ≈ 裸 ACT(stream_hp 1.03),且为修边界**牺牲了 chunk 内平滑**(0.935→1.045)。训练轨迹: 边界比/stream_hp 在 ep40 触底(0.94)后随 L1 暴涨(0.0095→0.0148)回升 = 过拟合,BC-guard best 落在 ep4。
4. **根本机制**:per-chunk 残差**看不到上一 chunk 末动作 → 架构上无法协调边界**;"不恶化边界"迫使"别动边缘"→ 连带丢 chunk 内平滑 → 净≈恒等。**边界连续性本质需跨 chunk 状态(上一步动作)= 确定性 blend 用 `_last_action` 免费提供**。
5. **闭环结论**:残差只该管 chunk 内平滑,边界交给 blend;**学到的 per-chunk 残差打不过确定性 blend**(改进I是blend的劣质学习替代)。要让残差net-helpful须加"上一步动作"输入(=学习版blend),但其相对"blend+残差"(#67已测≈blend)上限很小,ROI低。

## Document History
| 日期 | 内容 |
|---|---|
| 2026-06-01 | Stage 5 真机 A/B（各8ep）+ transfer-chain 诊断：部分传递+逐级衰减（非解耦），IMU accel carry 段显著降，fraction 陷阱已纠正，瓶颈=低层执行+结构 |
| 2026-06-03 | 低层控制器跟踪诊断：控制器强低通(>5Hz增益0.11-0.20)+执行态高频~90%指令无关 → 指令平滑被 action→state 层吞掉，瓶颈机制定论为低层执行（控制器层 Future Work） |
| 2026-06-11 | #67 真机三方 A/B(解耦 blend/残差)：动作空间 blend −53%、残差零增量+中频放大(#66兑现)；IMU N=3 欠功效但去冷启动后 blend accel −10.5%/残差零增量。结论：blend 有效、学到的残差不提供 IMU 增量、控制器瓶颈真机兑现。需 ≥9-12 交错集确认 |
