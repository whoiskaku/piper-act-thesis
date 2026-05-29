# Stage 3b — RefineNet 训练实验记录

> 配套文档：`RESEARCH_PLAN_v4.md` §5 Stage 3b / §4 T3-T4 / §7 Risk Register
> 代码提交：`c747dd4` (Stage 3b: RefineNet model + dataset + training script)

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

## Document History

| Run | 日期 | loss | 结论 |
|---|---|---|---|
| #1 | 2026-05-29 | L_BC + 0.1·jerk_RMS | 负结果：无平滑增益、平滑项过拟合。**后续诊断发现 jerk 指标本身不可靠，问题定义被修正（见上）** |
