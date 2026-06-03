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

## Document History

| Run | 日期 | loss | 结论 |
|---|---|---|---|
| #1 | 2026-05-29 | L_BC + 0.1·jerk_RMS | 负结果：无平滑增益、平滑项过拟合。**后续诊断发现 jerk 指标本身不可靠，问题定义被修正（见上）** |
| A | 2026-05-29 | L_BC only | sanity check：模型能拟合 A_clean（L_BC 0.0047<0.0057），但纯 L1 把 hf 放大 3.3×（ratio FAIL）。证明显式 hf 项必需 |
| B | 2026-05-29 | L_BC + 0.05·hf + 0.1·\|ΔA\| | **正结果**：hf_ratio 0.41(best 0.26) [pass]，L_BC/\|ΔA\| guardrail 全过，四关节 hf 全降 2~2.5×。RefineNet 去污染命题成立 |
| B-sweep | 2026-05-29 | +hf∈{0.05,0.1,0.2,0.4} | hf_ratio 饱和 ~0.25（j3 卡地板），无 trade-off cliff，BC 始终安全。**best-by-rule λ_hf=0.10**。地板为信息/容量限制 |
| deploy | 2026-05-29 | λ_hf=0.10 闭环 | **关键**：chunk 内 0.28×（监督式成立），但边界未平滑→无 blend 时 deploy jitter 反升 1.046×；deploy-time blend 后 0.933×（消除超额污染 20%, bnd_jump→0）。Phase1 headroom 小（ACT 仅高于 demo 33%），平滑卖点应在 Stage 4/5 |
| 3a | 2026-05-30 | denoise-ACT λ0.1 20K | 负消融：C2 hf 0.145×（比 RefineNet 更激进）但 L_BC ↑13%（保真劣化）；RefineNet 则 hf↓且 L_BC↓ → 反证 frozen+RefineNet。基建：num_workers=0+use_amp=False |
