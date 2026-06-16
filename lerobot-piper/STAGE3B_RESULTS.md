# Stage 3b — RefineNet 训练实验记录

> 配套文档：`RESEARCH_PLAN_v4.md` §5 Stage 3b / §4 T3-T4 / §7 Risk Register
> 代码提交：`c747dd4` (Stage 3b: RefineNet model + dataset + training script)

---

## 真机 headroom de-risk（2026-05-30，最重要的战略数据）

闭环 deploy 发现 Phase 1 自由空间 headroom 偏小后，用 `scripts/eval_smoothness_realrobot.py`
对比 **demo 示教（piper_pick_place_imu_v3, 30eps）vs C1-v3 ACT 真机部署（eval_C1_v3, 5eps）**，
量化真机上的真实平滑 headroom：

| 指标 | demo（地板） | ACT 部署 | Delta |
|---|---|---|---|
| Action RMS jerk | 113.7 | 182.5 | ↑60.5% |
| Action hf% (>5Hz) | 0.0002 | 0.0006 | ↑214% |
| 真机 IMU accel RMS (m/s²) | 3.47 | 3.78 | ↑8.9% |
| 真机 IMU gyro RMS (rad/s) | 0.337 | 0.370 | ↑9.8% |
| **真机 IMU hf% (>5Hz)** | **0.198** | **0.294** | **↑48.7%** |

**结论：**
1. ✅ **真机平滑问题真实可观**：ACT 部署比示教多 ~49% 高频 IMU 能量。论文前提成立。
2. 🔑 **动力学放大**：action 空间高频绝对量极小（0.0002→0.0006），传到 IMU 放大成 +48.7%。微小 action 修正可能撬动大 IMU 改善——对 RefineNet 有利。
3. ⚠️ **核心风险**：IMU hf 中 action-驱动（可修）vs 传感器/结构噪声（不可修）的占比未知。Phase 1 sim 里 executed jitter 被电机噪声地板主导（RefineNet 没怎么动 executed hf），暗示 action→IMU transfer 非必然。**只能在 Stage 5（真机）彻底回答。**

**战略含义**：Phase 1 完成"机制验证"使命；平滑卖点的兑现与 transfer 验证在 Stage 4（接触放大 IMU）/ Stage 5（真机）。

---

## Run #1 — λ=0.1 baseline（失败 / 负结果）

**结论先行**：λ=0.1 baseline **未能改善平滑度**。refined 输出的 jerk 不低于（甚至高于）frozen ACT 自身的输出，RefineNet 在部署时只会让结果变差，没有任何平滑增益。这是一个需要记录的诊断性负结果（对应 §7 Risk Register 中已预判的「Stage 3b RefineNet shows no smoothness improvement」一行）。

### 配置

| 项 | 值 |
|---|---|
| 数据 | `outputs/sim/phase1_paired_data.h5`，train 960 / val 240 |
| 模型 | RefineNet，1,969,735 参数（≈1.97M） |
| 优化器 | AdamW，lr=3e-4，weight_decay=1e-4 |
| 调度 | cosine + 5-epoch warmup |
| batch / epochs | 64 / 200 |
| λ_smooth | 0.1 |
| loss | `L1(refined, A_clean) + λ·jerk_RMS(refined)`（jerk 为 raw rad 空间三阶差分 / dt） |
| 训练耗时 | 2.6 min @ cuda |
| 输出目录 | `outputs/train/refine_net/phase1_lam0.1_lr0.0003_bs64_ep200/` |

参照基准：
- **ACT baseline jerk**（frozen C1-v3 输出本身）= **181.8**
- **demo jerk RMS baseline**（真机示教，目标）= **113.7**

### 关键结果

| 指标 | ep0（init，delta≈0） | ep24（best val_loss） | ep199（final） |
|---|---|---|---|
| train L_BC | 0.0065 | — | 0.0069 |
| train L_smooth (jerk) | 229.6 | — | **53.4** |
| val L_BC | 0.0056 | 0.0058 | 0.0065 |
| val L_smooth (jerk) | 189.3 | **180.4** | 206.1 |
| val jerk_refined vs ACT base(181.8) | 189（更差） | 180（≈持平） | 206（更差） |

best val_loss = **18.05** @ ep24。

### 观察到的失败模式

1. **refined 比 ACT 还抖**：final val jerk = 206 > ACT baseline 181.8；连 best ckpt 也只是 180 ≈ ACT baseline，没有任何净增益。距离 demo 目标 113.7 完全够不着。
2. **平滑项严重过拟合**：train L_smooth 从 229→53（暴跌），val L_smooth 却从 189→206（反升）。train/val gap ≈ 3.9×。模型在 960 个训练样本上死记硬背每个样本的平滑解，而没有学到「context → 平滑」的可泛化映射。
3. **best ckpt 出现得极早**（ep24），之后 val 一路恶化 —— 典型的过拟合曲线。

### 根因分析（待 Run #2 验证）

1. **Loss 量级失衡（首要嫌疑）**：`λ·L_smooth ≈ 0.1×180 = 18`，把 `L_BC ≈ 0.006` 压制约 **3000×**。优化器几乎只看到平滑信号，残差几乎不受 BC 约束，于是自由地去记忆训练集的平滑解。
2. **jerk² 在 raw rad 单位下** 让 loss 预算被平滑项病态主导；λ∈{0.01,0.1,1.0} 在这个尺度下其实没有覆盖到「BC 与平滑势均力敌」的区间。
3. **Decoder 容量**：100 个可学习 query 给了 decoder 足够自由度去逐样本记忆平滑，而非学习泛化映射。
4. **样本量小**：960 train 对 100-step chunk 的 jerk 泛化偏少；且 dropout 当前为 0.0，weight_decay 仅 1e-4。

### 下一步候选（未执行，待定方向）

- **(a)** 重构 loss：用 ACT-baseline jerk 对平滑项归一化（令 L_smooth≈O(1)）或改用「相对 jerk 下降率」目标 + 加 dropout / 增 weight_decay / 缩小 decoder，再跑小 sweep。
- **(b)** 先按现状跑原始 λ-scan {0.01, 1.0}，用实验确认上面的诊断。
- **(c)** 反思 noise model / paradigm（§7 给出的兜底路径）。

---

## Run #1 后续诊断（2026-05-29）— 问题定义被修正

Run #1 后做了量纲统计（`sim/analyze_loss_scales.py`），发现了一个比 loss 设计更根本的问题：**jerk 不是合适的指标，原定的 113.7 target 是聚合口径混用造成的幻象。**

### 关键测量（val，逐关节）

| 关节 | ACT jerk_RMS | A_clean jerk_RMS | ACT hf_frac | A_clean hf_frac |
|---|---|---|---|---|
| j1 | 268 | 241 | 0.0295 | 0.0002 |
| j2 | 378 | 366 | 0.0280 | 0.0001 |
| **j3** | **136** | **188** | **0.0382** | **0.0001** |
| j7 (grip) | 61 | 76 | 0.0435 | 0.003 |

### 结论

1. **jerk 没在测我们关心的高频抖动。** 铁证：j3 上 ACT 的高频能量是 A_clean 的 ~380×，但 jerk 反而更低（136 < 188）。jerk 被「大幅度的正常运动」主导，而非高频污染。用 jerk 当 Run #1 的主 loss 与目标，方向就错了。

2. **「113.7 是目标」的叙事作废。** noise_params.json 里的 `demo_jerk_rms_overall_mean=113.7` 是 **7 关节均值**（含 j4/j5/j6=0 的稀释）：per-joint=[220, 309, 165, 0, 0, 0, 102] → 均值 113.7。而训练日志里的 ACT「181.8」是 **pooled-RMS**（按元素方差加权）。二者是**不同聚合口径**，不可比；逐关节看 demo 本身（220/309/165）比 ACT（268/378/136）还抖或相当。113.7 与 A_clean 本就是同一个真机示教信号。**Stage 3b 不得以「降 jerk 到 113.7」为 headline target。**

3. **A_clean 来源确认**：`A_clean = ctrl_8_to_action_7(action_chunk)`，即 v3 真机 demo 原始 action（经 ±10% 幅度扰动），非任何滤波/平滑版本。

4. **hf_fraction 才是判别性指标**（ACT/clean 比值高达 100~380×），且暴露出真实的大改进空间。**这正是 RefineNet 要去掉的「ACT 额外引入的高频污染」。**

### 指标 / loss / 目标修正（已拍板）

- **Stage 3b headline metric** → **active-joint mean hf_fraction**（j1/j2/j3/grip，排除 j4/j5/j6=0）。jerk 仅保留为 secondary / continuity 指标。
- **成功标准用比值** `hf_frac_refined / hf_frac_ACT`：pass < 0.5，good < 0.25，excellent < 0.1。不设「压到 0.0001」的硬标准（过苛、易被 clean 的极低高频比例误导）。
- **两个 guardrail**：① L_BC(refined, A_clean) 不明显差于 ACT baseline；② |ΔA| 不过大。
- **训练 loss**（替换原 jerk 项）：
  `L_total = L_BC + λ_hf·L_hf_fraction + λ_delta·|ΔA| (+ optional 小 λ_acc·L_acc)`
  - L_hf_fraction 为平滑主力，**分母 stopgrad**（`E_high(pred)/stopgrad(E_total(pred)+ε)`）防止模型靠抬高低频能量作弊。
  - L_acc 判别力弱（ACT/clean 仅 1.17×）且可能惩罚合法轨迹曲率 → 暂不作核心项，最多小权重 ablation。
- **λ 初值范围**（按 val 量级推，令各正则项加权后 = L_BC 的 10~30%）：λ_delta ∈ [0.10, 0.30]，λ_hf ∈ [0.024, 0.072]，λ_acc ∈ [0.39, 1.16]。

### 修正后的 Run 计划

- **Run A**：L_BC only（λ 全 0）。sanity check —— RefineNet 到底有没有能力学回 A_clean，用 hf_fraction 判断是否真的去掉 ACT 高频污染。
- **Run B**：L_BC + λ_hf·hf_fraction + λ_delta·|ΔA|（主方案）。
- L_acc：optional，不作核心项。

---

## Run A — L_BC only（sanity check）

**目的**：回答根本问题——RefineNet 到底有没有能力学回 A_clean？以及纯 BC 能不能去掉 ACT 的高频污染？

**配置**：`--lambda-hf 0 --lambda-delta 0 --lambda-acc 0`，200 ep，其余同 Run #1。
输出：`outputs/train/refine_net/phase1_runA_BConly_lr0.0003_bs64_ep200/`

**结果（val，final ep199）**：

| 指标 | refined | ACT baseline | A_clean(floor) | 解读 |
|---|---|---|---|---|
| L_BC | **0.0047** | 0.0057 | 0 | ✅ 有能力拟合 A_clean，逐点精度超过 ACT |
| hf_fraction (active-joint mean) | **0.106** | 0.032 | 0.0008 | ❌ hf 被放大 |
| **hf_ratio = refined/act** | **3.29 [FAIL]** | 1.0 | 0.025 | ❌ 纯 L1 把高频放大 3.3× |
| \|ΔA\| | 0.0050 | — | — | 残差量级合理 |

逐关节 hf（refined vs act）：grip **0.25 vs 0.033 (7.6×)**、j3 0.095 vs 0.038、j1 0.047 vs 0.030、j2 0.030≈0.028。

**结论**：**问题不在模型容量，而在目标函数。** 逐点 L1 完全不约束频率，模型靠逐帧小幅修正压低平均 L1，代价是注入高频（grip 尤甚）。hf_ratio 从早期 ~6 缓降到 3.3 但始终 >1，证明 BC-only 不可能解决污染。**显式 hf 项被实验证明是必需的** → 进 Run B。

---

## Run B — L_BC + 0.05·hf + 0.1·|ΔA|（主方案，第一个正结果）

**配置**：`--lambda-hf 0.05 --lambda-delta 0.1 --lambda-acc 0`，200 ep。
输出：`outputs/train/refine_net/phase1_runB_hf0.05_dl0.1_lr0.0003_bs64_ep200/`

**结果（val，final ep199；best ckpt = ep125）**：

| 指标 | Run A (BC only) | **Run B** | best ckpt(ep125) | ACT baseline | clean floor |
|---|---|---|---|---|---|
| hf_ratio | 3.29 [FAIL] | **0.41 [pass]** | **0.26** | 1.0 | 0.025 |
| hf_fraction | 0.106 | **0.0131** | 0.0085 | 0.0322 | 0.0008 |
| L_BC (guardrail) | 0.0047 | **0.0046** ✅ | 0.0047 | 0.0057 | 0 |
| \|ΔA\| (guardrail) | 0.0050 | **0.0030** ✅ | — | — | — |

逐关节 hf（Run B refined vs ACT）：j1 0.0118 vs 0.0295、j2 0.0094 vs 0.0280、j3 0.0161 vs 0.0382、grip 0.0151 vs 0.0328 → **四个 active joint 全部降 ~2~2.5×**。

**结论**：
1. **核心命题成立** —— RefineNet 能去掉 ACT 引入的高频污染（hf 比 ACT 低 ~60%，比 BC-only 低 ~8×），**同时任务保真不降反升**（L_BC 0.0046 < ACT 0.0057），|ΔA| 仅 0.003。两个 guardrail 全过。
2. **自平衡调度兑现**：ep0 hf 主导（0.158）→ ep50 暴降到 0.011 → 后期 BC 接管，hf 从 best 0.0085 轻微回升到 final 0.0131。模型选择按 hf_ratio + BC guardrail，已保留 ep125 ckpt（ratio 0.264）。
3. **仍有上探空间**：final(0.41) > best(0.26) 的回升 + 距 good/excellent 仍有余量 → 下一步 λ_hf 上探 sweep，描绘平滑-保真 Pareto 前沿。

---

## λ_hf Pareto sweep（λ_delta=0.1, λ_acc=0, 200ep）

按拍板规则选 ckpt（`sim/sweep_pareto.py`）：primary = mean hf_ratio↓；guardrail = L_BC ≤ 1.1×baseline、mean|ΔA|<0.02、每个 active joint hf_ratio < 1.0（不许牺牲单关节换平均）。

| λ_hf | sel_ep | hf_ratio | L_BC | mean\|ΔA\| | j1 | j2 | j3 | grip |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 125 | 0.264 | 0.0047 | 0.0030 | 0.273 | 0.254 | 0.372 | 0.140 |
| **0.10** | 68 | **0.253** | 0.0053 | 0.0021 | 0.224 | 0.165 | 0.459 | 0.115 |
| 0.20 | 120 | 0.266 | 0.0051 | 0.0018 | 0.218 | 0.283 | 0.467 | 0.060 |
| 0.40 | 130 | 0.276 | 0.0051 | 0.0018 | 0.237 | 0.390 | 0.376 | 0.095 |

（ACT baseline L_BC=0.0057；clean floor hf_ratio=0.025）

**核心发现 —— 不是 trade-off 曲线，是饱和地板：**

1. **hf_ratio 在 λ_hf 8× 范围内饱和于 ~0.25–0.28。** 加大 λ_hf 既不再压低高频也不损 BC → **λ 不是瓶颈**，方法对超参鲁棒。
2. **0.40 未击穿 BC guardrail**（L_BC 0.0051<0.0057，|ΔA| 0.0018）。预期的 trade-off cliff 在此范围内不存在——stopgrad 分母 + 残差自然收缩使 BC 始终安全。**只有 plateau，没有 cliff。**
3. **地板被 j3 卡住**（ratio 0.37–0.47 跨 run 最高），grip 在高 λ 下可压到 0.06–0.14。0.25 地板距 clean 的 0.025 还差 ~10×。这是**信息/容量限制**（7 维瓶颈 + history/state 不足以预测抵消 ACT 在 j3 上的特定高频残差），非 λ 限制 → 对应 Future Work：ACT hidden state 作 RefineNet 输入 / 更强 context。

**Stage 3b 当前最佳：λ_hf=0.10**（best-by-rule，hf_ratio 0.253，L_BC 0.0053<baseline）。0.05 与之在噪声内且 BC 更优（0.0047），二者皆可作代表点。

---

## Phase 1 闭环 deploy eval — 揭示 chunk 边界问题（关键发现）

脚本：`sim/deploy_eval_phase1.py`。ckpt：λ_hf=0.10 best（ep68）。6 val eps（24–29）。
约束：ACT receding-horizon 开环（n_action_steps=100=chunk，无 temporal ensemble）；
ACT 观测 teacher-force 真机 episode（sim 无物体→ACT 不能跑 sim 渲染）；Phase 1 无物体→不测 task success。

**度量教训**：full-episode hf_fraction 被低频臂运动稀释（washout）；windowed fraction 在超平滑信号上又有分母陷阱（小分母放大残余 kink）。最终采用 **5Hz 零相位 Butterworth 高通后的 RMS（hp_rms，rad）** 作可信的绝对 jitter 度量；hf_aln（chunk 对齐窗 fraction）仅作 chunk 内训练-scale 对照。

| 条件 | hp_rms_cmd | hp_rms_exec | hf_aln | bnd_jump | fidelity |
|---|---|---|---|---|---|
| **demo（地板）** | 0.00108 | 0.00464 | 0.0002 | 0.0034 | 0 |
| ACT | 0.00161 | 0.00515 | 0.0306 | 0.0102 | 0.0079 |
| refined（无 blend） | **0.00169** | 0.00518 | 0.0086 | 0.0109 | 0.0071 |
| +blend10/20（边界续接） | **0.00151** | 0.00486 | 0.0073 | **0.0000** | 0.0072 |

**结论（用可信的 hp_rms）：**
1. **chunk 内**：refined hf_aln 0.0086 vs ACT 0.0306（0.28×）——监督式去污染在 deploy 内部完美保留。
2. **但 chunk 边界未平滑**：RefineNet 训练时只见孤立 100 步 chunk，目标无边界项。receding-horizon 部署（每 100 步重预测）下，**不做边界处理 refined 反而让 deploy jitter 变差（0.00169>0.00161, 1.046×）**——边界跳变增加抵消了 chunk 内平滑。
3. **deploy-time 边界续接 blend**（前 m 步线性消除 offset）：bnd_jump→0，hp_rms_cmd 0.00151（0.933× ACT），**消除 ACT 超额污染（=ACT−demo=0.00053）的 ~20%**，executed 0.943×，保真不降（0.0072<0.0079）。
4. **关键现实**：ACT 的 deploy 高频污染本身就小（仅比 demo 地板高 ~33%），且集中在 j1/j2；j3/grip 绝对量几乎无污染（监督式里它们的高 fraction 是小分母陷阱）。**Phase 1 自由空间的平滑 headroom 很小** → 真正的平滑收益需在 Stage 4（接触/抓取产生 IMU 抖动）与 Stage 5（真机）验证。
5. real vs closed-loop history 几乎无差异 → 反馈漂移可忽略，history 非瓶颈。

**→ 两个层面的下一步**：(a) 战术：boundary-aware（blend 已证明方向，或训练期纳入边界续接）；(b) 战略：Phase 1 已验证机制 + 边界问题 + headroom 偏小，平滑卖点应在 Stage 4/5 兑现。

---

## Stage 3a — denoising-ACT 负消融（2026-05-30）

直接 fine-tune C1-v3 ACT（解冻）+ hf 平滑惩罚（`enable_denoise_smooth`, λ_hf=0.1, 20K 步），
得 C2-denoising-ACT。与 frozen ACT / ACT+RefineNet 对比（val, active joints, `sim/eval_stage3a.py`）：

| 模型 | hf_fraction | hf_ratio/ACT | L_BC（任务保真） |
|---|---|---|---|
| C1 frozen ACT | 0.0327 | 1.00 | 0.0057 |
| **C2 denoise-ACT (λ0.1)** | 0.0047 | **0.145** | **0.0064 (↑13%)** |
| A_clean 地板 | 0.0008 | 0.026 | 0 |
| **ACT+RefineNet (Stage 3b)** | ~0.0085 | ~0.25 | **0.0046 (↓)** |

**结论（负结果成立）**：直接改 ACT 能把 hf 压得很狠（0.145×，甚至比 RefineNet 更激进），**但任务保真劣化**（L_BC 0.0057→0.0064，+13%）。而 frozen-ACT + RefineNet 做到 hf↓**且** L_BC 不升反降（0.0046）。→ 反证 frozen+RefineNet 设计：**改 ACT 本身 = 平滑/保真 trade-off；RefineNet = 两者兼得**。

注：Phase 1 无物体，L_BC 为任务保真代理；完整"task success 明显下降"证据待 Stage 4/5。

**基建坑（已记入 memory [[infra-act-finetune-crash]]）**：本环境 ACT fine-tune via train.py 需 `num_workers=0`（否则 dataloader worker 死锁）+ `use_amp=False`（否则训练循环 native double-free / core dump）。launcher：`sim/launch_stage3a.py`（直传 cfg 绕过 CLI override 无法注入新字段的限制）。

---

## 输入消融 S0–S3（2026-06-03，#61）

验证 history(action+imu 600) / current_state 是否值这个复杂度。固定 λ_hf=0.10, λ_delta=0.1, 200ep, 单 seed；best-by-rule（hf_ratio↓ 且 L_BC≤1.1×baseline）。

| 变体 | 输入 | 参数 | best hf_ratio | L_BC | j1/j2/j3/grip |
|---|---|---|---|---|---|
| **S0** | 仅 ACT_output | **1.90M** | **0.229** | 0.0052 | 0.18/0.26/0.38/0.07 |
| S1 | +state | 1.90M | 0.244 | 0.0050 | 0.20/0.34/0.36/0.07 |
| S2 | +state+hist_action | 1.94M | 0.262 | 0.0052 | 0.23/0.19/0.41/0.18 |
| S3 | 全部(=现状) | 1.97M | 0.251 | 0.0051 | 0.22/0.23/0.43/0.09 |

（ACT baseline L_BC=0.0057；四者 L_BC 均 < baseline）

**结论**：四变体 hf_ratio 全挤在 0.229–0.262（单 seed 噪声量级）；增量 +state +0.015 / +hist_action +0.018 / +hist_imu −0.011 —— **无任何输入分支带来明显改善**。**最简 S0(仅 ACT_output, 1.90M, 100 token)与完整模型统计等价且最小** → history/state 对 chunk 内平滑可砍，RefineNet 可大幅精简、部署不再依赖 history_buffer（论文奥卡姆加分）。与闭环"换 history 内容输出不变"一致。j3 在所有变体卡 0.36–0.43 → j3 地板与输入无关（信号结构/容量限制）。
**caveat**：① 单 seed，0.03 差异在噪声内（严格应补 2–3 seed 确认等价）；② chunk 边界是否需短 history 未被此 supervised 指标覆盖（边界用 deploy-blend 处理，600 帧 history 显非必需）。

---

## 输入信号对照 A_noisy vs ACT_output（2026-06-03，#62）

验证为何 RefineNet 输入用 ACT_output 而非 A_noisy(T2 原设想的去噪)。S0 架构, λ_hf=0.10, λ_delta=0.1, 均在 val(`sim/eval_input_signal.py`)。

| 模型 | 训练输入 | 评估输入 | hf:输入→精修 | hf_ratio | L_BC | \|Δ\| |
|---|---|---|---|---|---|---|
| **N0** | ACT_output | ACT_output(部署) | 0.032→0.007 | **0.229** ✓ | 0.0052 | 0.0021 |
| N1 | A_noisy | A_noisy(自身) | 0.278→0.265 | 0.953 | 0.0026 | 0.0008 |
| N1 | A_noisy | ACT_output(部署) | 0.032→0.053 | **1.644** ✗ | 0.0059 | 0.0008 |

参考(val active-joint hf): A_noisy=0.278(高), ACT_output=0.032, A_clean=0.0008。

**两个独立结论, 都支持 ACT_output:**
1. **训练/部署分布错配 → N1 部署失败**: N1 训于 A_noisy, 喂部署输入 ACT_output 时 hf 升至 1.64×(更抖)、L_BC 也差。输入必须 = 部署可得的 ACT_output。
2. **A_noisy 高频本质不可学**: N1 在自身分布上都几乎压不动 hf(ratio 0.953, 仅 |Δ|=0.0008 微调)。因 `A_noisy=demo+i.i.d.高斯电机噪声`, 白噪声不可预测→残差无法逐步抵消。反观 ACT 抖动是策略诱导、有结构、可预测→可学可去(N0 ratio 0.229)。

**结论**: "去噪 A_noisy→A_clean" 既非 well-posed(白噪声压不动)又部署错配; "精修 ACT_output→A_clean" 任务良定且部署一致。→ ACT_output 是正确输入, 数据双重佐证(作为论文 input ablation 依据)。

---

## ACT 隐状态作输入 S0 vs S0+H（2026-06-10，#63，受 RLT 启发）

测 hf~0.24/j3 地板是**信息瓶颈**(7维动作丢了感知信息)还是**信号/物理限制**。S0+H = S0 + ACT encoder 输出 mean-pool 摘要(512→32→256 单 token)。frozen ACT 挂 hook 取摘要(`sim/precompute_act_hidden.py`),其余同 #61。

| 变体 | 参数 | hf_ratio | L_BC | \|ΔA\| | j1/j2/j3/grip |
|---|---|---|---|---|---|
| S0 | 1.90M | 0.241 | 0.0050 | 0.0021 | 0.30/0.20/0.34/0.11 |
| S0+H | 1.93M | 0.244 | 0.0051 | 0.0022 | 0.25/0.23/0.31/0.17 |

效应:hf_ratio **+0.003**(无改善)、j3 −0.024(噪声内)。

**结论**:**加 ACT 内部 encoder embedding 不破地板** → hf~0.24/j3 地板**不是信息瓶颈**(7 维动作丢失的感知上下文喂回去也没用)。结合 #61(history/state 也无用)→ **RefineNet chunk 内平滑对输入不敏感**,地板是**信号结构/容量** + 下游控制器(Stage 5)限制。**证明保留极简 S0(仅 ACT_output)正确**(加 embedding 多 1.3% 参数 + 耦合 ACT 内部,零收益)。
**诚实区分**:这只否定了"ACT embedding 作监督输入"能破地板;RLT 的真正收益在**RL/奖励**部分(优化真机指标),本监督测试未覆盖——留作 Future Work。caveat: 单 seed(0.003 在噪声内);仅测了 encoder mean-pool 摘要(decoder 逐步特征/RLT 式冻结压缩器未试,因 encoder 摘要已无效,按计划不再扩)。

---

## #63(a) RLT 式注意力读出 image token（2026-06-10）—— 纠正 #63 并得正面结果

#63 的 mean-pool 是对全部 **1502** 个 encoder token 平均,其中 **1200 个是 #61 证无用的历史 token** → 感知信息被稀释。改为只对 **300 个 image token 做 learned-query 注意力读出**(RLT 式,`sim/precompute_act_img.py` + `RefineNet.use_act_img_attn`),并加 **零-image 对照 + 多 seed**:

| 变体 | 参数 | seeds | hf_ratio mean[min,max] | j1/j2/j3/grip |
|---|---|---|---|---|
| S0 | 1.90M | 3 | 0.249 [0.241,0.256] | 0.24/0.24/0.37/0.12 |
| **S0+Hattn** | 2.30M | 3 | **0.209 [0.201,0.222]** | **0.17**/0.21/0.30/0.14 |
| S0+Hattn+zero | 2.30M | 2 | 0.258 [0.251,0.265] | 0.21/0.28/0.36/0.16 |

**结论(纠正 #63)**:
1. **S0+Hattn 稳定低于 S0,range 不重叠**(0.222<0.241)→ −0.039(~16% 相对)是真效应非噪声。
2. **零-image 对照(0.258)≈ S0,而非 ≈ S0+Hattn** → 同架构同参数、image 喂零退回基线 → **增益来自 image 信息,非参数容量**。
3. → **hf 地板(部分)是信息瓶颈**:ACT 的 image/感知特征经 **RLT 式注意力读出**接入确有用;**压缩方式是关键**(mean-pool 被历史稀释→无效;注意力读出 image token→有效)。**正面验证 RLT 设计**;#63 的负结果是"压缩太粗"的伪影。
4. 增益主要在 **j1(0.24→0.17 砍半)**;j3 仍 0.30–0.37(部分仍受信号结构/控制器限,见 STAGE5)。

**caveat**:仍是 supervised chunk 内指标;真机 IMU 收益仍受控制器瓶颈(Stage 5)封顶——这是**方法层改进**(更好的输入提取),非直接真机 IMU 收益。部署需把 image-token attention readout 接进 wrapper(未做)。

---

## ⚠️ 重大 caveat — 招牌 hf_ratio 是 fraction,不等于绝对平滑(2026-06-10 发现)

应"画 ACT vs refined 输出对比图"的需求,逐样本核对时发现的**口径问题,影响 Stage 3b 全部 hf_ratio 数字的解读**:

**`hf_ratio`(训练损失 `hf_fraction_loss` + 模型选择 + 上文所有表格)用的是 fraction = E_high/E_total(>5Hz 能量*占比*,Hann-FFT,`train_refine_net.py:70-84,190`),不是绝对高频量。** 这与方法描述里宣称的"以 5Hz Butterworth 高通后 hp_rms 作权威绝对指标"**不一致**——实际跑的全程是 fraction。

全 val(240, active joints)三度量对照(S0 best ckpt):

| 度量 | ACT | S0 | S0+Hattn | refined/ACT |
|---|---|---|---|---|
| fraction(招牌口径) | 0.0322 | 0.0077 | 0.0066 | **0.24× / 0.20×** |
| 绝对 E_high(Hann-FFT,能量加权) | 6.0e-3 | 6.4e-3 | 6.0e-3 | **1.07× / 1.00×** |
| 绝对 hp_rms(filtfilt 5Hz, rad) | 0.00122 | 0.00131 | 0.00128 | **1.08× / 1.05×** |

**结论:在 sim 开环 chunk 内,RefineNet 并不降低绝对高频抖动(两个独立绝对度量一致 ~1.05–1.08×,甚至略升)。** 模型输出本身正常(std≈ACT、各关节 E_total refined/ACT=0.98–1.00、\|Δ\|仅 0.0036 rad、L1(refined,clean)<L1(ACT,clean)),Δ 把 chunk 整体往 clean 拉但**没有专门抵消高频**。

机制(高/低运动分组证实):fraction 改善是 **mean-of-ratios** 在**低运动 chunk**(fraction 高、绝对值微小)上撑出来的;主导绝对抖动的**高运动 chunk 上 refined≈ACT(E_high ratio 1.08,不降)**。即 `hf_ratio` 踩进了本仓库一直警告的**"分母陷阱"**。

**与真机 Stage 5 "-40%" 的关系(假设,未证)**:真机指令 action hp 绝对值确实降了,但 sim 开环 chunk 内不降 → 真机收益最可能来自**部署时 chunk 边界 blend + temporal ensembling(跨 chunk 连续性)**,而非学到的 chunk 内残差。需"纯边界 blend vs RefineNet 残差"对照证实。**在证实前,不应把真机平滑全归功于 RefineNet 残差。**

**待办(等"回到研究")**:① 验证边界-blend 假设;② 考虑把损失/模型选择改成绝对 hp_rms 重训。诚实绝对对比图(样本 27,abs-hp 选)已存 `outputs/eval/out_cmp_ACT_vs_S0.png` / `out_cmp_ACT_S0_S0Hattn.png`——绝对高通分量上 ACT/refined **基本重叠**(与本节一致)。
> 注:S0+Hattn 的 fraction 相对 S0 仍稳定更低(#63a 的 image-信息结论在 fraction 口径内成立),但**同样不蕴含绝对高频下降**。

---

## 🔨 定论 — 绝对高频不可监督消除(2026-06-10 决定性实验)

接上节待办②:给 `train_refine_net.py` 加 **`--hf-mode abs`** 直接优化绝对高频 `L_abs = E_high(refined)/E_high(ACT).detach()`(ratio-of-sums **能量加权**,分母固定输入 ACT → 非分母陷阱;模型选择改按 `abs_hf_ratio`)。S0,λ∈{0.5,1,2,5},200ep。

**坑中坑**:第一版 `L_abs` 误写成 mean-of-ratios `(E_high_r/E_high_a).mean()`,被近静止关节(分母≈0)主导(init 时 loss=12117!),只压低运动样本、高能量 j2 不动 → abs 卡 0.99(假平)。改 ratio-of-sums 后才暴露真相。

**train vs val 绝对高频轨迹(λ=2.0,`abs_hf_ratio`)**:

| epoch | 10 | 30 | 60 | 100 | 199 |
|---|---|---|---|---|---|
| **train_abs** | 1.01 | 0.71 | **0.23** | **0.085** | **0.036** |
| **val_abs**   | 1.03 | 1.04 | 1.21 | 1.28 | **1.36** |

**train 绝对高频 → 0.036(砍 96%),val → 1.36(比不精修还差 36%)。** 所有 λ 一致;ckpt_best(选 val 最优 + BC 护栏)的独立 val 评估 absE_high/ACT≈1.0、hp_rms 1.2–1.3。图:`outputs/eval/abs_overfitting_trainval.png`。

**结论(一锤定音)**:
1. **模型有能力消除高频(train→0.04),但纯靠记忆**;**该消除不泛化**(val 反升)。映射 ACT→clean 的 **>5Hz 残差不是可泛化函数**——clean 的高频与 ACT 的高频互不相关,唯一可泛化的信号是低频轨迹形状(L1 学到的)。
2. **绝对高频对监督式残差精修是不可降的(val 最优≈1.0,优化只会更糟)**,与损失口径无关(fraction 与正确的绝对目标都败)。呼应 #62「A_noisy 白噪声不可学 ratio0.95」——**高频本质是样本特有噪声**。
3. → **Stage 3b 招牌 fraction hf_ratio 0.25× 是度量产物**(mean-of-ratios×低运动加权),**不是真平滑**。
4. → 真机 Stage 5「指令 hp −40%」**不可能来自 chunk 内精修** → 几乎必然来自**部署时 chunk 边界 blend + temporal ensembling(跨 chunk 连续性)**。**边界-blend 假设从"待验证"升级为"强证据支持的主因"**。

**对论文叙事的影响(重大)**:RefineNet 的**贡献不在 chunk 内监督平滑**(已证无效)。真正的杠杆在:(a) **部署时跨 chunk 连续化**(边界 blend + TE),(b) **控制器层**(Stage 5,力矩/高带宽)。下一步优先级:**验证边界-blend 假设**(纯 blend vs RefineNet 残差,在 sim/真机部署流上量绝对 hp)以定位真机收益来源。
> caveat:本结论基于 960 个 sim chunk;"更多数据能否让高频抵消泛化"未测,但 train/val 鸿沟表明目标高频近噪声、希望渺茫。新代码:`--hf-mode abs`、`hf_abs_loss`、val stats 增 `abs_hf_ratio`、模型选择 `sel_key`。

### 边界-blend 假设验证(2026-06-10,`deploy_eval_phase1.py` 加 `ACT_blend10` 态)

deploy-stream(整 episode 跨 chunk 拼接)绝对 hp_rms_cmd 三方分解,图 `outputs/eval/blend_vs_residual.png`:

| 条件 | hp_rms_cmd | vs ACT | 边界跳变 |
|---|---|---|---|
| 裸 ACT | 0.00163 | 1.000 | 0.0105 |
| **ACT + 纯 blend(无残差)** | 0.00145 | **0.886 (−11%)** | **0.0000** |
| ACT + 残差(refined,无 blend) | 0.00171 | **1.046 (+5% 更差)** | 0.0111 |
| ACT + 残差 + blend | 0.00151 | 0.927 (−7%) | 0.0000 |

**部署流上全部平滑来自确定性边界 blend;学到的残差零贡献、甚至略有害**(残差单独 +5%;残差+blend 的 −7% 还不如纯 blend 的 −11%)。executed 流同(blend 0.939、残差 1.009)。残差不碰边界跳变(0.0111),blend 清零。

**→ 真机 Stage 5 的 A/B(wrapper-on=blend+残差 vs wrapper-off=裸 ACT)所测"RefineNet 收益"实为"边界-blend 收益"**;`refine_enabled` 同时开关了 blend 与残差(`refine_deploy.py:138/140`),二者一直被混淆。**学到的残差(本工作主贡献)在任何层面都不提供绝对平滑增益。** 唯一有效的是无需学习的边界 blend + 控制器低通(Stage 5)。

### abs 残差的部署消融(2026-06-10,封死"换损失能救"的退路)

用**正确绝对损失(ratio-of-sums)训出的 abs 残差**(各 λ 的 ckpt_best,val abs≈1.0)重复上面的部署分解(`--refine-ckpt`/`--out-tag` 已参数化;json: `deploy_eval_abs{λ}.json`):

| 残差来源 | 残差单独(无blend) | 残差+blend | 纯blend(无残差) |
|---|---|---|---|
| runB fraction λ=0.10 | 1.046 (−5%) | 0.927 | 0.886 |
| abs λ=0.5 | 1.092 (−9%) | 0.988 | 0.891 |
| abs λ=1.0 | 1.124 (−12%) | 1.024 | 0.887 |
| abs λ=2.0 | 1.128 (−13%) | 1.022 | 0.889 |
| abs λ=5.0 | 1.166 (−17%) | 1.060 | 0.893 |

1. **abs 残差在部署流上一致有害(−9%~−17%),且随 λ 单调恶化**——λ 越大、对(不可学的)train 噪声抵消越激进,部署时注入的伪高频越多。val abs≈1.0 的 ckpt 到部署流恶化到 1.09–1.17,因 deploy chunk 起点错位 = 额外分布漂移。
2. **残差+blend 全线 ≤ 纯 blend**——加残差只会拖累。**纯 blend 严格占优**(0.886–0.893,五行几乎不变,因为 blend 不依赖残差)。
3. → **"换正确损失重训"不能救残差**(chunk 内过拟合 + 部署有害,双重证据);残差无贡献的结论与训练口径无关,**消融闭环**。
> 基建注:循环连跑多次 deploy_eval 会触发已知的 dataloader 退出期 `double free`(同 Stage 3a),偶发静默失败;单独跑稳定,json 均已生成。

### S0 / S0+Hattn 的 fraction 增益到底是什么(2026-06-10 收尾分析)

val 按运动量对半分组(motion = active-joint AC 能量):

| 组 | 模型 | fraction ratio | 绝对 E_high ratio |
|---|---|---|---|
| LOW(近静止) | S0 | 0.214 | 1.051 |
| LOW(近静止) | S0+Hattn | **0.168** | 0.998 |
| HIGH(运动) | S0 | 0.384 | 1.079 |
| HIGH(运动) | S0+Hattn | 0.395 | 1.006 |

Hattn−S0 的 fraction 改善:LOW **+0.0025(全部)**,HIGH −0.0001(零)。

1. 模型学到的唯一可泛化平滑 = **"近静止段输出平线"**(clean 在静止段本来就平,"保持不动"可预测,不需要预测噪声;运动段高频=样本噪声不可学,fraction 卡 0.38-0.40)。
2. **#63a 的 Hattn 增益全部来自 image 信息帮模型识别任务阶段(静止 vs 运动)** → 机制结论"RLT 式 attention 读出能有效利用 ACT image 信息"**保留**(零-image 对照逻辑仍自洽);"破 hf 地板"的解读**作废**——压的是静止段 fraction,绝对能量可忽略。
3. 小真发现:Hattn 绝对 E_high≈1.00 vs S0 1.05-1.08——image 信息让残差至少不注入伪高频(不帮倒忙≠增益)。
4. fraction 掉 5× 而绝对不动 = mean-of-ratios 给静止段样本同权,大比值改善撑起招牌数字(同一度量陷阱在分组内递归出现)。

---

## 🔄 #64 大数据重训 — 可泛化绝对平滑涌现,残差复活(2026-06-10/11)

### 数据
原 960 train 实为 **240 唯一起点 × 4 扰动变体**(ACT_output/act_img 由观测决定,4 变体共享输入)。`sim/generate_dense_act_pairs.py`:stride 50→5 → **train 2339 / val 568 唯一输入(9.7×)**;无扰动变体(消除"同输入4目标"标签噪声);A_clean=raw demo 切片(无 sim);history 填零(S0/Hattn 不读);一次 ACT 前向同取 ACT_output+act_img → `phase1_dense.h5`。
> 副发现:ACT 推理输出含 dataset history 随机采样驱动的抖动(同帧两次 history 差 4.0 → ACT_output 差 0.026 rad);模型本身确定(同张量 diff=0)。

### 损失-度量军备竞赛(每代博弈与修复,论文素材)
| 代 | 损失 | 模型学到的博弈 | 修复 |
|---|---|---|---|
| 1 | fraction(Hann) | 压静止段占比(分母+加权陷阱) | → 绝对口径 |
| 2 | abs(Hann-FFT) | **边缘倾倒**:Hann 窗 t[0:5]/t[95:100] 权重≈0,垃圾倒边缘(9.3×/1.8×),内核真降 0.80-0.86 | → FIR 均匀权重 |
| 3 | absfir(FIR) | ①实现 bug:零填充+未去均值 → 边界瞬态比抖动大 5 个量级,损失只看伪迹(L1 崩);②修 bug 前还会塞 FIR 过渡带(5-7Hz) | → AC去均值+reflect 填充;Hann+FIR 组合 |
| 4 | **absboth**(0.5·Hann-abs + 0.5·FIR) | 盲区互补(频率尖锐×时间均匀),无处可倒;选模型用 absmax(两口径 worst-case) | ✅ 终版 |

### chunk 级结果(dense val 568, ckpt_best, BC-guarded)
| 模型 | Hann-abs | FIR | hp_rms | L1 |
|---|---|---|---|---|
| ACT | 1.000 | 1.000 | 1.000 | 0.00839 |
| dense S0 (fraction) | 1.040 | 1.087 | 1.089 | 0.00631 |
| dense Hattn (fraction) | 1.048 | 1.248 | 1.137 | 0.00640 |
| **dense S0 (absboth λ1.0)** | **0.789** | **0.722** | **0.946** | 0.00790 |
| **dense Hattn (absboth λ1.0)** | **0.761** | **0.712** | **0.940** | 0.00790 |
| clean(天花板) | 0.591 | 0.620 | 0.771 | — |

- **三个独立绝对口径全 <1**,per-step profile 无边缘伪影(边缘 1.14-1.17,内核 0.77-0.79),L1 优于 ACT。对 λ(0.3/1/2)不敏感。
- abs(Hann) 单独训也给出 val 0.80@ep49 的真泛化(train/val U 形,数据假设的第一证据),但有边缘倾倒 → absboth 是诚实版。
- **Hattn 在每个绝对口径都略优于 S0**(0.761/0.712/0.940 vs 0.789/0.722/0.946)→ #63a 的 image 信息增益**首次在诚实口径成立**。
- fraction 目标即使大数据也无绝对增益(1.04-1.25)→ **数据量与目标函数都曾是瓶颈,缺一不可**。

### 部署流分解(deploy_eval_abs… vs deploy_eval_dense_absboth)
| 条件 | hp_rms_cmd vs ACT |
|---|---|
| 纯 blend | 0.894 |
| 旧残差+blend(fraction, 最好) | 0.927 |
| **新 absboth 残差单独** | **0.942(首个为正的残差贡献)** |
| **新 absboth 残差+blend** | **0.821(显著超纯 blend)** |

**「纯 blend 严格占优 / 残差任何层面无贡献」的旧定论被推翻**——前提是 ①唯一输入 ~10×、②无博弈绝对损失。blend(11%)+残差(6%)≈ 合计(18%),近似可加。

### 结论与 caveat
1. ACT→clean 的高频残差**在小数据下表现为不可学噪声,在大数据+诚实损失下部分可学**(chunk 内核 ~20-28% 能量,部署 −6% 单独/−18% 合血)。此前"本质不可学"的判断是**数据饥饿的伪影**——abs 实验当时已留 caveat,现兑现。
2. 仍未到天花板(0.59-0.77),且 hp_rms 口径(逐样本等权)只到 0.94 → 大头改善在高能量样本。
3. caveat:仍 sim/teacher-forced;真机收益依旧受控制器低通封顶(Stage 5);新残差的真机 A/B 未做;dense 目标定义(raw demo)与原(sim 干净执行)略异。
4. 工件:`--hf-mode absboth`、`hf_absfir_loss`/`_fir_highpass`(AC+reflect)、`absmax_ratio` 选模型;图 `outputs/eval/dense_retrain_summary.png`。

### #65 多 seed 稳健性 + K-实现增广(2026-06-11)

**A. 多 seed(dense K1, absboth λ1.0, 3 seeds, BC-guarded best absmax)**:

| 变体 | s42 | s43 | s44 | mean [min,max] |
|---|---|---|---|---|
| S0 | 0.789 | 0.776 | 0.792 | 0.786 [0.776,0.792] |
| **Hattn** | 0.761 | 0.765 | 0.736 | **0.754 [0.736,0.765]** |

range 不重叠(0.765<0.776) → **#64 两核心结论过 seed 检验**:绝对平滑可泛化(±0.01);Hattn image 增益稳定(−0.03)。(s43 S0 的 history.json 被退出期 double-free 吞掉,从 ckpt_best 的 val_stats 恢复。)

**B. K-实现增广(`generate_dense_act_pairs.py --k-train 4`)**:利用副发现"ACT 抖动由 dataset history 随机采样驱动"——同起点重复前向 = 同 A_clean 目标、独立噪声实现(实现间 ACT_output 差 0.007-0.016 rad,目标全同)。train 2339→**9356**,val 保持 K=1 可比。结果:

| 变体 | K1 (2339) | K4 (9356) |
|---|---|---|
| S0 | 0.786±0.008 | 0.783(无增益) |
| Hattn | 0.754±0.015 | 0.739(贴 seed 下沿) |

**饱和**:同起点加噪声实现不再推进天花板(0.59/0.62)→ **剩余缺口绑定于场景多样性(仅 24 个 demo episode),不是噪声采样数**;要继续逼近需更多真机演示(新瓶颈=数据采集)。
**副发现:Hattn 的 U 形过拟合显著更慢**(ep99 absmax 0.841 vs S0 1.076)——image 条件化兼具正则化效应(RLT 式读出的又一独立益处)。
图:`outputs/eval/scaling_seeds_summary.png`(规模曲线 240→2339→9356 + K4 U 形对比)。

### #66 absboth 残差的机制特征化(2026-06-11)

把 #64「它有效」深化为「它如何有效」。频域传递函数 |FFT(refined)|/|FFT(ACT)|(dense val, active joints, Hann, DC 去除):

| 频段 | refined/ACT | clean/ACT | 解读 |
|---|---|---|---|
| 0–1 Hz(任务运动, 带宽 0.6Hz) | ~1.00 | ~0.95 | 保留主运动 ✓ |
| **1.8–4.5 Hz** | **1.15–1.57** | 0.85–1.03 | **放大!**(损失 5Hz 截止的盲区, refined 在此远离 clean) |
| 5.4–14.4 Hz(真抖动) | 0.86–0.99 | 0.27–0.43 | 一致衰减(但远未达 clean) |

1. **反相抵消是真机制**:corr(Δ_highfreq, ACT_highfreq) = **−0.509**,|Δ_hf|/|ACT_hf| RMS = **0.716** → 残差在 >5Hz **主动反相抵消** ACT 抖动(非"不添加",而是"减去"),这是 #64 可泛化平滑的来源。
2. **副作用**:损失 `HF_CUTOFF=5.0` 只惩罚 >5Hz,模型在 **2–4.5Hz 无约束地放大(1.15–1.57×)**;任务带宽仅 0.6Hz → 这是注入了不需要的中频。真机 IMU 对加速度敏感,该中频可能有害 → **待办: 降 HF_CUTOFF 到 ~2.5Hz(贴任务带宽)重训, 验证能否消除放大、谱更贴 clean**(已开始把 HF_CUTOFF 改 CLI 可配, 中断转实机)。
3. **S0 vs Hattn 传递函数几乎重合** → image 条件化不改变机制(仍是反相抵消 + 中频放大),只在 13–14Hz 略多衰减 + 抗过拟合(#65),增益机制一致。
4. **执行流地板(sim, 已有 deploy_eval 数据)**:refined+blend 执行流 hp_exec=**0.905** ≈ demo(完美平滑指令)执行流 **0.906** → **sim 里命令平滑已触执行噪声地板**(电机噪声+控制器设的下限 ~0.90),指令再压(往 0.59)对 sim 执行流无用。Stage 5 控制器瓶颈在新残差上的量化复现。→ **命令平滑的边际价值在 sim 耗尽, 真机执行流地板是否更低 = 唯一能回答的途径 = 真机三方 A/B。**
图:`outputs/eval/residual_spectrum_mechanism.png`。

### #68 降 HF_CUTOFF 消除 #66 中频放大 + 控制器透传洞见(2026-06-11)

`HF_CUTOFF` 做成 CLI 可配(`--hf-cutoff`;`hf_power` 改调用时读全局 + main 重算 `_FIR_TAPS` + tag `+cut{v}`)。dense S0/Hattn absboth 扫 cutoff,**固定口径**评估(独立于训练 cutoff:5Hz 高通 + 2–4.5Hz 带通 + L1):

| 模型 | 5–15Hz/ACT(真抖动) | 2–4.5Hz/ACT(#66放大带) | L1(vs ACT 0.0084) |
|---|---|---|---|
| S0 cut5.0 | 0.946 | **1.389** 放大 | 0.0079 |
| **S0 cut2.5** | 0.985 | **1.019** 中性 | **0.0075** |
| S0 cut2.0 | 0.924 | **0.716** 衰减 | 0.0086 |
| Hattn cut5.0 | 0.940 | 1.386 | 0.0079 |
| Hattn cut2.5 | 0.933 | 1.085 | 0.0078 |

1. **cut2.5 零代价修复**:2–4.5Hz 放大 1.39→1.02 消除,>5Hz 衰减保住(0.95→0.99),L1 反而更好。
2. **cut2.0 主动衰减该带**(→0.72,低于 clean 0.937),>5Hz 仍 0.92,但 L1 略升(0.0086>ACT)= 开始牺牲保真。
3. **控制器透传洞见(联 Stage 5)**:控制器**通过 <5Hz(增益≈1)、阻挡 >5Hz(0.11–0.20)**。故残差的 >5Hz 衰减大半被控制器吃掉(#67 真机 IMU 收益不显著之因);而 **2–4.5Hz 直接透传到 IMU** —— cut5.0 在此放大(#67 真机 IMU 中频 +10% 变差的元凶),cut2.0 在此衰减 → **cut2.0 很可能是真机 IMU 收益最大的杠杆(针对真正透传的频段)**。重定义残差的目标:不该盯被控制器阻挡的 >5Hz,而该压控制器透传的 2–5Hz。
图:`outputs/eval/cutoff_sweep_tf.png`。
**待验**:真机 A/B/C(cut2.0) ——若 C−B 在 IMU(尤其 2–4.5Hz 带)显著负,残差终于在真机赚到超 blend 的增量。

### #66b 频谱剪刀差的相干性验证(2026-06-11)— 修正 + 强化机制

用相干性(magnitude-squared coherence with demo, 跨 val chunk 平均, active joints)验证"哪些频段是任务运动 vs ACT 特有抖动"。图 `outputs/eval/coherence_bands.png`:

| band | ACT~demo | S0残差~demo | Hattn残差~demo |
|---|---|---|---|
| 0–0.6Hz 任务 | 0.994 | 0.971 | 0.993 |
| 0.6–2Hz | 0.703 | 0.581 | 0.664 |
| **2–4.5Hz 透传** | **0.266** | 0.077 | 0.141 |
| **5–15Hz 抖动** | **0.011** | 0.011 | 0.007 |

1. **>5Hz 相干 ≈0.011 = ACT 高频与 demo 完全不相关 = 纯 ACT 特有抖动** → 铁证 #64 的"不可学(是噪声)" + 剪刀差上刀刃(可去但被控制器阻挡)。
2. **修正先前过强论断**:此前用"+7% RMS"推"2–4.5Hz 大多是任务运动"是**错的**(能量相近≠结构相同)。相干仅 **0.266** → ACT 的 2–4.5Hz **只 27% 与 demo 共享,73% 是 ACT 特有中频**。
3. → **这 73% 既可去、又能传到 IMU = 真机 IMU 余量存在**(剪刀差没那么死,残差前景比"硬上限"乐观)。挑战:与 27% 任务结构纠缠,须**选择性**去除。
4. **token 的价值(正向但温和)**:2–4.5Hz 上 S0 把 demo 相干砸到 0.077,**Hattn 保住 0.141(~2×)** → token 去中频抖动时**更少破坏任务结构**,更内容感知。但两者都降(非外科手术式)→ 去 73% 抖动不可避免地伤及部分 27% 任务相干,token 只是改善这个权衡。
5. **真机命题(更新)**:bare ACT vs ACT+Hattn残差(cut2.0)——残差去掉 2–4.5Hz 的 73% ACT 特有抖动,**真机 IMU 该带是否随之下降**(且任务成功/保真不破)。任务成功 + 真机保真仍需录制验证。

### #69 token 提取四方对比(2026-06-17)— 平滑↔保真权衡, RLT-AE 无免费午餐

回答"有无更好的 token 提取(RLT 式)"。新增两件:**M-q**(8 个 learned query readout, refine_net `n_img_readout`)与 **RLT-AE**(论文式:冻结 image token 上重构预训练 encoder-decoder 压缩器 → K=8 token, `sim/train_rlt_compressor.py`,val explained-var **0.929**,RefineNet `use_act_rlt` 分支)。全 dense absboth cut2.0。评估固定口径 + 2–4.5Hz 与 demo 的相干(选择性)。图 `outputs/eval/token_tradeoff.png`。

| 变体 | 2–4.5Hz/ACT(平滑) | 5–15Hz/ACT | coh~demo(保真) | L1 |
|---|---|---|---|---|
| S0(无 token) | **0.716** | 0.924 | 0.077 | 0.00857 |
| M-q(8) | 0.875 | 0.960 | 0.099 | 0.00868 |
| 1q(#63a, 主方法) | 0.926 | 0.994 | 0.141 | 0.00819 |
| RLT-AE(重构压缩) | **0.968** | 1.013 | **0.176** | 0.00865 |
| (ACT 自身 coh~demo=0.266;clean 2–4.5Hz/ACT=0.937) | | | | |

1. **四者构成单调的"平滑↔保真"权衡线**:越压 2–4.5Hz(去抖)→ demo 相干越低(越伤 27% 任务结构)。因该带 27% 任务 + 73% 抖动**纠缠**,无 token 提取法能逃离此权衡。
2. **RLT-AE 不提供更好的选择性平滑**:它落在**保守/保真端**(几乎不平滑 0.968、最保任务 0.176、L1 最好)。重构预训练的信息完整 token(explained-var 0.93)→ 残差变**保守**(近恒等),而非更外科式。**论文式 RLT 提取在我们的监督平滑设定下并不更优**——更忠实但不更平滑,无免费午餐。
3. **S0(无 token)= 激进端**(最平滑 0.716 但最伤任务 coh 0.077)。**1q/M-q 居中**;1q(主方法)是合理工作点(适度平滑 + 适度保真)。
4. → **RLT-AE 对照达成目的**:证明不必上更复杂的重构压缩器,**简单的单 query 注意力读出(1q/Hattn)作主 token 方法是站得住的**。
**caveat**:各变体 ckpt_best 选择 epoch 不同(absmax+BC;RLT/1q 早停、S0 训到 ep141),权衡部分由训练动力学中介,但跨 epoch 定性顺序稳健。RLT 训练退出期 double-free 吞了 ep149/history(同 Stage3a),用 ep58 best。

---

## Document History

| Run | 日期 | loss | 结论 |
|---|---|---|---|
| #1 | 2026-05-29 | L_BC + 0.1·jerk_RMS | 负结果：无平滑增益、平滑项过拟合。**后续诊断发现 jerk 指标本身不可靠，问题定义被修正（见上）** |
| A | 2026-05-29 | L_BC only | sanity check：模型能拟合 A_clean（L_BC 0.0047<0.0057），但纯 L1 把 hf 放大 3.3×（ratio FAIL）。证明显式 hf 项必需 |
| B | 2026-05-29 | L_BC + 0.05·hf + 0.1·\|ΔA\| | **正结果**：hf_ratio 0.41(best 0.26) [pass]，L_BC/\|ΔA\| guardrail 全过，四关节 hf 全降 2~2.5×。RefineNet 去污染命题成立 |
| B-sweep | 2026-05-29 | +hf∈{0.05,0.1,0.2,0.4} | hf_ratio 饱和 ~0.25（j3 卡地板），无 trade-off cliff，BC 始终安全。**best-by-rule λ_hf=0.10**。地板为信息/容量限制 |
| deploy | 2026-05-29 | λ_hf=0.10 闭环 | **关键**：chunk 内 0.28×（监督式成立），但边界未平滑→无 blend 时 deploy jitter 反升 1.046×；deploy-time blend 后 0.933×（消除超额污染 20%, bnd_jump→0）。Phase1 headroom 小（ACT 仅高于 demo 33%），平滑卖点应在 Stage 4/5 |
| 3a | 2026-05-30 | denoise-ACT λ0.1 20K | 负消融：C2 hf 0.145×（比 RefineNet 更激进）但 L_BC ↑13%（保真劣化）；RefineNet 则 hf↓且 L_BC↓ → 反证 frozen+RefineNet。基建：num_workers=0+use_amp=False |
