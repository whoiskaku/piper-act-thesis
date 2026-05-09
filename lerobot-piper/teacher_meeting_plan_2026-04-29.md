# ACT 改造方案 — Group Meeting 计划书

**日期**: 2026-04-29
**主题**: 基于老师建议的新 ACT 训练范式 — 实施方案

---

## 1. 背景与方向调整

之前基于 SAH (Spectral Action Head) 的方案已在真机完成 4 路对比 (Vanilla / α-imu-only / SAH-only / SAH+TE)，结果显示 SAH 虽然在频谱指标上有改善，但破坏了任务成功率（SAH-only: 1/15）。老师指出 SAH 类的"基函数带宽限制"未触及问题本质，建议改为：

- **训练阶段**让 ACT 接受历史 episode 的反馈信息（autoregressive across episodes）
- **训练后期**用真机闭环 fine-tune，把真机 IMU 与 GT IMU 的差异作为监督信号

本计划书是该方案的具体实施细节。

---

## 2. 老师方案核心思想（复述）

```
                ┌──────────────────────────────────────────┐
                │       ACT  (改造后)                       │
                │                                          │
   images ─────►│                                          │
   j.ang  ─────►│                                  Action  │
   a.seq(T-1) ─►│              encoder → decoder ───►Seq(T)│
   r.IMU(T-1) ─►│                                          │
                └──────────────────────────────────────────┘
                                  ↓
                               真机执行
                                  ↓
                          real IMU(T), action(T)
                                  ↓
                  Loss 1 = ||action(T) - action_GT(T)||
                  Loss 2 = ||real_IMU(T) - GT_IMU(T)||


  1st training step  : Loss 1 (BC)              — offline supervised
  2nd training step  : Loss 1 + α · Loss 2      — real robot in loop, RL
  Future Work        : DNN dynamics (action → IMU) → 完全可微，纯 supervised
```

---

## 3. 1st Training Step 实施细节（已敲定）

### 3.1 架构改造

ACT 输入新增两路（来自上一个 episode）：

| 输入 | 形状 | 说明 |
|--|--|--|
| `a.seq(T-1)` | `[L_max, 7]` | 上一个 episode 完整 action 序列 |
| `r.IMU(T-1)` | `[L_max, 6]` | 上一个 episode 完整 IMU 序列（下采样到 30Hz） |

**Encoding**：
- `a.seq`: `Linear(7 → 512)` + 学习式 positional embedding → L 个 token
- `r.IMU`: `Linear(6 → 512)` + 学习式 positional embedding → L 个 token

**融合方式**：encoder 输入端早期融合，全部 token concat：

```
Encoder input = [z, state, image_patches, a.seq tokens, IMU tokens]
              ≈ [1, 1, ~600, ~300, ~300]  =  ~1200 tokens
              + attention mask (屏蔽 zero-pad 部分)
              ↓
Transformer Encoder (4 layers, dim=512)  ← 沿用 ACT 默认结构
              ↓
Transformer Decoder (1 layer)            ← 沿用 ACT 默认结构
              ↓
Action head → [B, T_chunk=100, 7]
```

**注**：先尝试不做 pooling 的全序列拼接（最大信息保留）。若 RTX 3090 显存或速度不足，降级到 attention-pool 或 Conv1D 下采样。

### 3.2 数据集 — `piper_pick_place_imu_v3`

| 项目 | 内容 |
|--|--|
| Episodes | **50** |
| 控制频率 | 30 Hz |
| 必要字段 | `action`, `observation.state`, `observation.images.*`, **`observation.imu_real`** |
| IMU 来源 | TSND151，下采样到 30 Hz，与 action 严格时间对齐 |
| IMU 安装 | 全程固定，拍照记录（用于部署复现） |

**前置条件**：
- ✅ Gamepad axis bug（axis 2 → axis 3）已修，不会再有 joint_4 漂移污染
- ✅ `policy.reset()` + `teleop_safety_stop()` 两个 patch 已应用

**数据使用策略**：
- 训练时**随机配对** (任意 `prev_episode` → 任意 `current_episode`)
- 首个 episode (无 prev) → zero pad + attention mask
- Episode 长度不一致 → dynamic batch padding

### 3.3 训练设置

| 项目 | 值 |
|--|--|
| 初始化 | 从头训练（不加载 vanilla ckpt） |
| Loss | action L1 only（Loss 1，Loss 2 留给 2nd step） |
| Batch size | 24（RTX 3090 24GB 估算可行） |
| Training steps | 80K – 150K（依 loss 曲线） |
| Optimizer | AdamW, lr = 1e-5 |

### 3.4 1st Step 验收标准

全部满足才开始 2nd step：

1. ✅ 训练 loss 收敛
2. ✅ 验证集 action L1 ≤ vanilla ACT
3. ✅ 真机部署 5 次 task success ≥ vanilla 的 5/5

---

## 4. 2nd Training Step（待 1st 完成后细化）

### 4.1 反传方式：真机 + RL（Policy Gradient）

**为什么用 RL**：真机不可微，PyTorch 无法对 `real_IMU(action)` 求梯度。RL 用 `R = -||real_IMU - GT_IMU||` 作为标量 reward，通过 `∇θ log π_θ(a|s) × R` 估计梯度，绕开可微性问题。

**实现简化**：不给 ACT 加 σ head，而是在 rollout 时套固定噪声 `Normal(μ_ACT, σ_fixed)`，σ 可手工 schedule。架构改动量为零。

### 4.2 待决策点（拟在 1st step 验证后讨论）

| 编号 | 内容 | 候选 |
|--|--|--|
| D2 | Reward 粒度与形式 | episode/step/chunk × L1/L2/exp shaping |
| D3 | Trajectory 切分 | chunk-as-trajectory（每步独立 log_prob） |
| D4 | RL 算法 | PPO（主推） / SAC（数据更省，备选） |
| D5 | Loss 1 + Loss 2 协调 | α 调度、是否交替更新 |
| D6 | 真机安全机制 | 紧急停止、reward 异常处理 |

---

## 5. 时间表

| 阶段 | 内容 | 预计时间 |
|--|--|--|
| Week 1 | `record.py` 加 IMU 字段同步 + 采集 v3 (50 ep) | 1 周 |
| Week 1–2 | ACT 架构改造（与采集并行） | 1 周 |
| Week 2 | Sanity check 5K 步小训练，确认显存与速度 | 2 天 |
| Week 2–4 | 1st step 完整训练 80K–150K 步 | 2 周 |
| Week 4 | 1st step 离线 eval + 真机 5 次部署验证 | 2 天 |
| Week 5+ | 2nd step (RL) 设计与实施 | 待 1st 通过后展开 |

---

## 6. 待老师确认的关键问题

### Q1 — `a.seq(T-1)` 与 `r.IMU(T-1)` 的粒度

我们目前理解为 **episode 级**（即上一个完整 episode 的全序列）。
是否符合您的本意？还是应理解为 **chunk 级**（上一个 100 步 chunk）？

### Q2 — 编码方式（A3）

为最大保留信息，1st step 计划**不做 pooling**，将 `a.seq` 和 IMU 完整序列（~300 token 各）直接拼到 encoder 输入。
代价：encoder 输入 token 数从 ~600 增至 ~1200，self-attention 计算量约 ×4。
RTX 3090 24GB 估算可行（batch 24）。是否同意此方案？

### Q3 — 2nd step 的反传路线

(a) **真机 + RL**：您图上的默认方案，需引入 Policy Gradient
(b) **DNN_dynamics 提前做**：把 Future Work 的 DNN 提前训出来，用作可微 surrogate，2nd step 完全 supervised，速度快得多

哪条优先？

### Q4 — 数据量

50 episodes 是否足够？是否需要扩到 80–100 ep？

---

## 7. 风险与降级方案

| 风险 | 应对 |
|--|--|
| 显存 OOM (A3 全序列拼接) | AMP fp16 / gradient checkpointing / 降级到 attention pool |
| 训练不收敛 | 检查 attention mask、lr、初始化 |
| 真机 task success 低于 vanilla | 暂停 2nd step，debug 1st step（可能是 history 输入分支干扰） |
| RL 训练发散 | σ 减小、加 KL penalty、PPO clip 范围调小 |

---

## 8. 已完成的前置工作

- ✅ Phase 0 频率分析：GT 95% bandwidth = 0.6 Hz
- ✅ SAH 真机 4 路对比（Vanilla / α-imu-only / SAH-only / SAH+TE）
- ✅ Gamepad axis bug 修复（v1 数据集 joint_4 漂移污染的根因）
- ✅ `policy.reset()` 与 `teleop_safety_stop()` patches 应用
- ✅ v2 数据集 (30 ep) 采集与训练完成（用于对照）

---

## 附录 — 关键文件参考

- 4 路真机对比脚本: `scripts/p1v2_realrobot_compare.py`
- 当前 ACT 实现: `lerobot/common/policies/act/modeling_act.py`
- 采集脚本: `lerobot/common/robot_devices/control_utils.py`
- 老师方案原图: `~/下载/plan.pdf`, `~/下载/毕业论文（3）.pdf`
