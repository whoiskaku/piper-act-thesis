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

## Document History

| Run | 日期 | λ | 结论 |
|---|---|---|---|
| #1 | 2026-05-29 | 0.1 | 负结果：无平滑增益，平滑项过拟合，root cause 待验证 |
