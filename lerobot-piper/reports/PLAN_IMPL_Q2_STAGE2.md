# G→π v2 阶段二（Q2 开发检验）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 零机时，用近期三个会话（ab5 / ab6 / ab7）的 88 集补正运行做留一会话检验：比较两种响应形式（非线性差分、条件线性）与固定 H、零响应，看谁更能预测「补正造成的变化」（Q2 开发检验）。

**Architecture:** 新目录 `lerobot-piper/sim/q2/`，只读复用 `sim/q1/`（数据契约、窗口与计分、2a 的结构与训练、损失、产物工具）。链路如下：
- 数据层：δa、日志契约、冻结名单；
- δa 时间轴与固定 H：δa 的可见性规则只在这一处实现；
- 两种响应形式；
- 训练、预测与计分：名义 2a 直接复用阶段一的 `train_one`；增量在冻结的名义模型上训练；
- 命令行与产物：五个阶段、冻结闸门、截止时间、输入指纹；
- 汇总判据 → 只报告的波形图 → 正式运行脚本。

每折的名义模型只用另外两个会话的裸跑，增量只用另外两个会话的补正运行，留出会话只用于评分。正式运行前，预注册须由用户冻结。

**Tech Stack:** Python 3.10（`lerobot` conda 环境）· numpy 1.26 · scipy 1.15 · torch 2.11（CPU 单线程、确定性算法）· MuJoCo 3.9（理想 IMU 渲染、G-2 参照）· pytest 9.1 · matplotlib 3.10

**Spec:**
- `lerobot-piper/reports/PREREG_Q2_STAGE2.md`：草稿，用户 2026-10-03 已审——§13 第 1–6 项采用，第 7 项（24 小时）暂定；
- `lerobot-piper/reports/DESIGN_G_PI_V2.md` §4：「阶段一之后的用户决定」与「阶段二第 1–6 项」（用户 2026-10-03 定稿）。

⚠ **本计划授权的范围**：
- 写代码与测试；
- Task 9 的 30 步耗时测量（`timing`：不评分外层留出会话、不选步、不保存正式产物）；
- 在预注册里回填指纹、名单文件与实测耗时。

**正式运行（Task 10）必须等用户确认 24 小时上限，并冻结 `reports/PREREG_Q2_STAGE2.md`。**
- 零机时，不连接机械臂；不训练 π；不采集数据。
- 不改 `sim/q1/`：阶段一的 `code_sha=d6a823ba1ed4a862`、`config_hash=be83d2e77958e85c` 保持不变。

**用户 2026-10-03 认可的两处默认（已同步写进预注册 §7，`405a7a5`）**
1. **ab4 的 β 跨种子汇总**（只报告；Task 6 `beta_seed_mean`）：
   - 每个「折 × 种子」先汇总内积再相除；
   - 再平均有定义的种子；
   - 保留逐种子值与有效种子数；全部无定义记 N/A。
2. **保留只报告的波形图**（Task 7）：
   - 固定 seed 0；按零响应误差选中位集与最难集；
   - 只读保存的预测；
   - 不参与判据，也不进代码指纹。

---

## Global Constraints

每条都是所有任务的隐含要求。引号内是预注册（§ 号）或用户 2026-10-03 的原文。

**授权与边界**
- 「零机时；不训练 π；不采集新数据（pilot 推迟）；不修改 G-2、固定 H 与阶段一的任何产物。」（预注册抬头）
  - ⇒ 不改 `sim/q1/` 下任何文件；
  - ⇒ `outputs/step1_H/step1_H_v1.*` 与 `outputs/step4_gvnext/` 只读。
- 「代码、`code_sha`、`config_hash`、数据名单文件与实测耗时：实施后填入；冻结前不运行任何正式入口。」
- 「`timing` 只用于另行授权的短程耗时测量：不评分外层留出会话、不选步、不保存正式训练产物；它不是运行的旁路。」（§11）
- 前提：「按以后能保证维持 ab5/ab6 的装置状态来考虑；实机结果不好再重新收集。」结论只在这一前提下有意义（§9）。

**数据与时间轴**
- 数据（§1）：
  - ab5：裸跑 A 40；补正 C 40 + Csafe 3；
  - ab6：A 30；D 30；
  - ab7：Apost 15；D 15；
  - 辅助：ab4 P2 6、P3 4；「P1 不参与」。
- 「名义指令 u = 日志 chunk 的原始 ACT 指令；δa = 日志逐帧的 `da`，即日志记录的请求补正量（不是执行位移）；日志动作 a = u + δa。」a = u + δa「只证明日志里的动作组合关系，不证明设备实际接收或执行了同样的补正」（§0）。
- 标签：「`e_t = ω_real[t+3] − ω_ideal(u)[t]`。全局标定 R、b 同阶段一。」
- δa 的可见性（§2）：
  - 「过去段 t0−19 … t0：日志记录的请求补正量」；
  - 「当前 chunk t0+1 … t1：离线条件预测使用日志中的当前 chunk 补正序列。这是离线给定的条件，不代表历史部署时在起点就知道全部补正」；
  - 「超出 t1：取 0」。
- 「计分范围、逐 chunk 拼接、`P(·)`（去均值 → bandpass41 → 丢首尾 20 帧）、会话误差比 ρ：同阶段一。」
- 「数据名单冻结为新的名单文件…prepare、每个作业与汇总都核对。」

**模型（§3）**
- 名义 2a：「结构同阶段一（66,441 参数）。每个『外层折 × 种子』各一个：只用该折另外两个会话的裸跑训练，选步后冻结。」
- 非线性差分：
  - 「`ŷ = ŷ_nom(u) + [N(u, δa) − N(u, 0)]`」；
  - 「输入 = 2a 的 13 个通道 + 6 个 δa 通道，共 68,361 参数」；
  - 「输出头零初始化 ⇒ 起点 = 零响应；δa = 0 时增量严格为 0」。
- 条件线性：
  - 「`Δy_k = Σ_{τ=0}^{19} K_θ(k)[τ] · δa_{k−τ}`，`K_θ(k) = Σ_{r=1}^{4} a_r(k) · B_r`」；
  - 「固定 4 个、20 个滞后，不扫描」；
  - a_r「可正可负的普通输出，不用 softmax，输出头零初始化」；
  - 「共 70,436 参数」；
  - 基核初值「每个元素独立取 N(0, 0.05²)（归一化单位），由训练种子决定」（§13-1）。
- δa 归一化与输出尺度：
  - 「δa 归一化保留零点：每个关节只除以训练数据上 δa 的均方根，不减均值；均方根 < 1e-6 rad 时取 1e-6」；
  - 「增量输出的尺度沿用冻结名义模型的 s_ε（关节）与 s_gap（误差）」。
- 固定 H：
  - 「H = `step1_H_v1`（float64 sha256 `89d118d4b69837f0`）」；
  - 「`Δê_t = Σ_{τ=0}^{15} H[τ] · δa_{t−τ}`，不再额外平移 L = 3」；
  - 「卷积包含过去的 δa，不在每个 chunk 清零重启」。
- 「三种预测（候选、零响应、固定 H）共用同一种子的名义模型，按种子配对。」

**划分与训练（§4、§5）**
- 划分：
  - 「名义 2a 只用另外两个会话的裸跑训练并冻结；增量分支只用另外两个会话的补正运行训练。不得复用任何包含留出会话的拟合结果。」
  - 「留出会话的裸跑与补正运行都只用于最终评分，不参与归一化、选步或调参。ab5 的 C 与 Csafe 全部归入 ab5。」
- 停止规则：
  - 「上限 10000 步；检查点 0 / 250 / 500 / 1000 / 1500 / 2000 / 3000 / 4000 / 6000 / 8000 / 10000；种子 0 / 1 / 2；Adam lr 1e-3、梯度裁剪 1.0；CPU 单线程确定性训练。」
  - 「两次验证 ρ 的平均最小者为选中步数（完整精度平局取最早）。」
- 增量选步：叫「固定名义模型下的增量选步验证」。「这不是整个流程的严格内层留一，也不宜断言偏差必然偏乐观；本轮主动采用这一折中，不再增加嵌套层级。」
- 「归一化按内层划分计算：内层中新估计的 δa 尺度、响应网络统计量与损失尺度，只用那一个内层训练会话；沿用的冻结名义模型统计量是例外。」
- 选步单位：
  - 「每个『外层折 × 种子 × 分支 / 候选』分别选步」；
  - 「第 0 步是有效选择：增量选中 0 就保留零响应，不强制训练」；
  - 「选中 10000 则标注『最佳点落在搜索边界』，不追加步数」。
- 损失：「ω 项与关节项 1:1；关节项内宽带与 3–9 Hz 0.5 : 0.5」（§13-5，同阶段一 `q1_losses`）。

**判据（§6）**
- 汇总：「先对种子平均误差，得到每折的 ρ̄；再算 `d_f = (ρ̄_H,f − ρ̄_候选,f) / ρ̄_H,f`，三折等权平均。逐种子结果全部报告，不额外要求每个种子都通过。」
- 「每个候选单独判断，三条都满足才通过」：
  1. 「三折平均 d ≥ 5%」；
  2. 「每一折 ρ̄_候选 < ρ̄_H」；
  3. 「每一折 ρ̄_候选 < ρ̄_零响应」。
- 两种形式之间：
  - 「只有一个通过 ⇒ 用通过的那个」；
  - 「两个都通过 ⇒ 默认保留条件线性；只有非线性差分在三折都更好、且三折平均相对误差再降 ≥ 2%，才保留非线性差分。2% 的算法同上，分母换成条件线性的 ρ̄」；
  - 「两个都不通过 ⇒ 阶段二不通过，不进入阶段三」。
- 「不得逐折挑赢家再拼成『最佳模型』的平均成绩。」「不追加其他门槛。」

**只报告（§7）**
- Q1 复查：留出裸跑上名义 2a 的 ρ，并列 G-2。
- ab4：P2 / P3 分开；P1 不参与；「不用于选步、选模或推进判决」；β 带三处解读限定。
- 前瞻分层；逐种子结果；逐「外层折 × 种子 × 分支」的 step*。

**措辞（§9；所有报告与图都适用）**
- 「这是开发检验，不是密封确认；也不是补正因果效应的直接验证。」
- 「两种形式的对照结论只能是『两套具体响应模型谁的折外预测更好』…不能把胜负单独解释成非线性是否有用。」
- 「5% 与 2% 是事前的工程推进 / 选择阈值，不等于统计显著，也不能换算成真机改善。」
- 「通过只意味着可以进入阶段三的离线 π 研究，不代表 π 会在真机上有效。」
- 不能把结果「单独归因于对未见策略的泛化」（§4）。

**产物与完整性（§8、§11）**
- 正式入口检查冻结状态（状态行以「**状态：已冻结」开头，并核对 `code_sha` 与 `config_hash`），不接受训练参数覆盖。
- 输入指纹含：数据文件；「冻结的标定 R、b 文件哈希、固定 H 文件哈希」；依赖代码与库版本；名义与增量产物。续跑只跳过指纹一致的作业。
- 截止时间：「第一次 prepare 在名单核验通过后记录截止时间，续跑不重置；超时则拒绝运行并标记『未完成』，汇总不作通过判决。」上限暂定墙钟 ≤ 24 小时（§13-7）。
- 汇总前强制核验：「3 折 × 3 种子 × {名义, 两种增量} 齐全、名单一致、由内层评分表重算的选中步数与权重对应、全部产物在截止时间内。」
- 计划审阅补的三项执行检查（用户 2026-10-03）：
  - 「测速入口：在函数入口强制 1 ≤ steps ≤ 30，越界时在加载数据、训练之前拒绝」；
  - 「续跑完成判定：终训作业还须检查 .pt、.npz 的存在和哈希；损坏就重新排队，不能只凭 JSON 指纹跳过」；
  - 「运行日志：每次尝试保留独立日志和退出状态，避免续跑覆盖失败证据」。
- 「若修复涉及冻结代码，须重新冻结，不能直接沿用旧实验继续。」
  - ⇒ 新的 `code_sha` / `config_hash`，并换新的 `OUT` 从头运行；
  - 旧目录的截止时间记录绑定旧指纹，入口会拒绝。

**工作方式**
- 工作区里有另一会话的未提交文件，**不碰、不提交**：
  - 已修改：`reports/PLAN_INDEPENDENT_RESPONSE_DATA.md`、`reports/PLAN_PILOT_RESPONSE_14.md`、`sim/replay_probe.py`；
  - 未跟踪：`reports/RELEASE_PILOT_RESPONSE_14.json`、`reports/RUNBOOK_PILOT_RESPONSE_14.md`、`sim/make_pilot_response14.py`、`sim/run_pilot_response14.py`、`sim/test_make_pilot_response14.py`、`sim/test_replay_pilot_scope.py`、`sim/test_run_pilot_response14.py`。
  - 每次只 `git add` 本任务列出的文件，不用 `git add -A` / `git add .`。
- 在 `develop` 分支提交；提交信息不加 `Co-Authored-By`。
- 入口 `gc.disable()`，用 `python -u`（MuJoCo + scipy 的 GC 段错误会吞掉缓冲输出）。长时间训练在用户自己的终端跑，不放 harness 后台。

## Review Focus

下面五类是 spec 隐含、最可能出问题的输入或失败方式。每条后面列出锁住它的测试。

1. **留出会话或 ab4 经作业接线混入训练、归一化或选步**：记录的字段正确，实际传入的集却不对。
   → Task 5 `test_jobs_train_and_select_only_on_the_allowed_runs`（直接记录传给训练与内层评分的集）、`test_inner_split_mapping`；Task 4 `test_increment_statistics_come_only_from_the_given_training_runs`。
2. **下一 chunk 的指令或 δa 漏进候选预测**：当前 chunk 的补正是离线给定条件，超出 t1 必须取 0。
   → Task 2 `test_da_visible_zeroes_rows_after_t1_and_after_the_episode_end`；Task 3 `test_increment_ignores_next_chunk_commands_and_corrections`、`test_increment_is_causal_over_the_sequence`。
3. **固定 H 基准被悄悄改了口径**：每个 chunk 清零重启，或额外平移 L = 3。
   → Task 2 `test_h_window_matches_the_gap_axis_formula_without_shift`、`test_h_window_uses_past_corrections_and_never_future_ones`、`test_load_h_checks_the_frozen_fingerprint`；Task 4 `test_predictor_shapes_and_h_equals_the_whole_episode_convolution`。
4. **零点被破坏**：j4 的 δa 恒为 0、选中第 0 步、δa = 0 时增量不为 0。任何一种都会让候选与零响应不可比，或让 β 失去意义。
   → Task 3 `test_da_scale_preserves_zero_and_uses_rms_not_std`、`test_zero_init_is_zero_response`、`test_zero_correction_gives_exactly_zero_increment`；Task 4 `test_checkpoints_are_prefixes_and_step_zero_is_zero_response`；Task 5 `test_increment_chain_and_zero_step_keeps_zero_response`；Task 6 `test_beta_is_na_when_the_increment_is_identically_zero`。
5. **未冻结就运行、测速变成旁路、混用过期或被篡改的产物、续跑覆盖失败证据、超时后仍给判决**。
   → Task 5 `test_formal_commands_refuse_unless_frozen`、`test_timing_refuses_out_of_range_steps_before_loading_or_training`、`test_jobs_refuse_when_inputs_changed_after_prepare`、`test_passed_deadline_stops_every_formal_command`、`test_list_jobs_counts_and_requeue_on_input_change`、`test_final_jobs_requeue_when_weights_or_predictions_are_missing_or_corrupt`、`test_changed_code_cannot_continue_an_old_run`；Task 6 `test_damaged_trees_are_rejected`（8 种损坏）、`test_main_marks_incomplete_runs_and_gives_no_verdict`；Task 8 `test_each_attempt_keeps_its_own_logs_and_exit_status`、`test_a_job_stopped_by_the_deadline_stops_the_run_with_a_clear_message`。

---

## 文件结构

全部新建于 `lerobot-piper/sim/q2/`。目录不放 `__init__.py`；模块名统一加 `q2_` 前缀。`sim/q1/` 只读复用：`conftest.py` 与各入口把它加进 `sys.path`。

| 文件 | 职责 |
|---|---|
| `conftest.py` | 关 GC；把 `sim/q1` 与 `sim/q2` 加进 `sys.path`；注册 `slow` 标记 |
| `q2_data.py` | 183 集 → `Q2Episode`（阶段一的 `Q1Episode` + `fold / arm / kind / da`）；日志契约；按冻结名单核对 |
| `q2_roster.json` | 冻结的数据名单（9 组共 183 集；在代码指纹里） |
| `q2_testing.py` | 合成集工厂（测试用，不读真实数据） |
| `q2_windows.py` | δa 的可见性与固定 H 的窗口响应 —— **δa 时间轴规则的唯一实现处** |
| `q2_models.py` | 两种响应形式、δa 尺度、增量解码 |
| `q2_train.py` | 名义 2a 训练（复用 `train_one`）、增量训练、预测、计分小工具、代码与配置指纹、选步规则 |
| `q2_run.py` | 命令行 `timing / prepare / refs / list-jobs / nom-inner / nom-final / inc-inner / inc-final`；冻结闸门、截止时间、输入指纹、原子写产物 |
| `q2_report.py` | 完整性核验、先种子后折的 ρ̄、判据、两种形式之间的选择、只报告项；输出 `summary.json` 与 `summary.md` |
| `fig_q2.py` | 只报告的波形图（不在代码指纹里） |
| `run_q2_stage2.sh` | 正式运行（可重入、并行；每次尝试独立日志与退出状态）；**冻结后**在用户终端执行；不在代码指纹里 |
| `test_*.py` | 每个模块一份 |

报告：
- 修改 `reports/PREREG_Q2_STAGE2.md`：Task 9 回填，Task 10 冻结；
- 新建 `reports/RESULT_Q2_STAGE2.md`：Task 10。

产物目录 `outputs/q2_stage2/`。`outputs/` 被 `.gitignore` 忽略，不提交；报告里记哈希。

## 通用约定

- 所有命令在 `lerobot-piper/` 下执行；`PY=/home/enolab-guo/miniconda3/envs/lerobot/bin/python`。
- 测试命令：
  - 快速测试：`$PY -m pytest sim/q2 -q -m "not slow"`（91 个，约 30 s）；
  - 集成测试：`$PY -m pytest sim/q2 -q -m slow`（2 个，约 90 s：读 183 集真实数据并核对名单与日志契约；G-2 在合成集上的参照）。
- 本计划的代码已在暂存区镜像里跑通：快速测试 91 个、集成测试 2 个全部通过。镜像通过软链接指向真实的 `outputs/`、`lerobot/` 和既有 `sim/` 模块（含 `sim/q1/`）。
  - 并且按任务顺序逐个重放过：每个任务在实现前测试失败，实现后通过；最终文件与镜像逐字节相同。
  - 照抄即可。若与仓库现状不符，先查原因再改；**不要为了让测试通过而放宽测试**。
- 提交信息格式 `阶段二 Q2：<内容>`，**不加 `Co-Authored-By`**。

---

### Task 1: 数据层（183 集 → `Q2Episode`、日志契约、冻结名单）

**Files:**
- Create: `sim/q2/conftest.py`, `sim/q2/q2_data.py`, `sim/q2/q2_testing.py`, `sim/q2/q2_roster.json`（Step 4 由命令生成）
- Test: `sim/q2/test_q2_data.py`

**Interfaces:**
- Consumes（只读复用 `sim/q1`）：
  - `q1_data`：`ROOT`、`HERE`、`ROSTER`（阶段一冻结名单）、`DataContractError`、`Q1Episode`、`load_episode(path, sess)`
  - `q1_testing.synthetic_episode(seed, T, sess)`
- Produces:
  - 常量：`ROOT, FOLDS=("ab5","ab6","ab7"), BARE, CORR, AUX={"ab4": ("P2","P3")}, LOG_TOL=1e-6, ROSTER_FILE, ROSTER`
  - `class LogContractError(D1.DataContractError)`
  - `@dataclass Q2Episode(Q1Episode)`，新增字段 `fold`、`arm`、`kind ∈ {"bare","corr","aux"}`、`da (T, 6)`
  - 函数：
    - `log_da(path, T, kind) -> (T, 6)`：核对 |a − raw − δa| ≤ 1e-6；裸跑 δa ≡ 0；补正运行与探针 δa 非零
    - `from_q1(e1, fold, arm, kind, da) -> Q2Episode`、`load_episode(path, fold, arm, kind) -> Q2Episode`
    - `arm_paths(fold, arm)`、`load_arms(spec, kind)`、`spec_for(folds, kinds) -> [(spec, kind)]`
    - `load(folds=FOLDS, kinds=("bare","corr","aux"), check=True) -> list[Q2Episode]`
    - `groups(eps) -> {"fold/arm": [name]}`、`check_roster(eps, keys=None, roster=None)`、`select(eps, fold=None, kind=None)`
  - `q2_testing`：`synthetic(seed=0, fold="ab5", arm="C", kind="corr", T=587)`、`synthetic_set(n_per=2)`、`roster_of(eps)`

- [ ] **Step 1: 写失败的测试** —— `sim/q2/test_q2_data.py`

```python
import json

import numpy as np
import pytest

import q1_data as D1
import q2_data as D2
import q2_testing as T

EXPECTED = {"ab5/A": 40, "ab6/A": 30, "ab7/Apost": 15, "ab5/C": 40, "ab5/Csafe": 3,
            "ab6/D": 30, "ab7/D": 15, "ab4/P2": 6, "ab4/P3": 4}


def test_roster_is_frozen_with_the_expected_groups():
    assert {k: len(v) for k, v in D2.ROSTER.items()} == EXPECTED
    names = [n for v in D2.ROSTER.values() for n in v]
    assert len(names) == len(set(names)) == 183
    assert all(n.startswith(k.split("/")[0] + "/" + k.split("/")[1] + "_ep") for k, v in D2.ROSTER.items() for n in v)


def test_synthetic_episode_fields():
    e = T.synthetic(0, "ab6", "D", "corr")
    assert (e.fold, e.arm, e.kind) == ("ab6", "D", "corr") and e.da.shape == (e.T, 6)
    assert np.abs(e.da).max() > 0 and (e.da[:, 3] == 0).all()
    b = T.synthetic(1, "ab6", "A", "bare")
    assert (b.da == 0).all()


def _log(tmp_path, kind, tamper=0.0, zero=False):
    rng = np.random.default_rng(0)
    T_ = 120
    raw = rng.normal(0, 0.1, (T_, 7))
    da = np.zeros((T_, 6)) if (kind == "bare" or zero) else rng.normal(0, 1e-3, (T_, 6))
    a = raw.copy(); a[:, :6] += da; a[5, 0] += tamper
    pay = {"frames": [{"a": a[k].tolist(), "da": da[k].tolist()} for k in range(T_)],
           "chunks": [{"start": 0, "raw": raw[:100].tolist()}, {"start": 100, "raw": raw[100:].tolist()}]}
    p = tmp_path / f"{kind}.json"
    p.write_text(json.dumps(pay))
    return p, T_, da


@pytest.mark.parametrize("kind", ["bare", "corr", "aux"])
def test_log_da_accepts_a_consistent_log(tmp_path, kind):
    p, T_, da = _log(tmp_path, kind)
    assert np.array_equal(D2.log_da(p, T_, kind), da)


def test_log_da_rejects_inconsistent_logs(tmp_path):
    p, T_, _ = _log(tmp_path, "corr", tamper=1e-4)                  # a ≠ u + δa
    with pytest.raises(D2.LogContractError):
        D2.log_da(p, T_, "corr")
    p, T_, _ = _log(tmp_path, "corr", zero=True)                    # 补正运行的 δa 全为 0
    with pytest.raises(D2.LogContractError):
        D2.log_da(p, T_, "corr")
    p, T_, _ = _log(tmp_path, "corr")                               # 当作裸跑读：δa 非零
    with pytest.raises(D2.LogContractError):
        D2.log_da(p, T_, "bare")
    assert issubclass(D2.LogContractError, D1.DataContractError)    # 不会被加载循环（只吞 ValueError）吞掉


def test_check_roster_catches_a_swap_that_keeps_the_total():
    eps = T.synthetic_set()
    roster = T.roster_of(eps)
    D2.check_roster(eps, roster=roster)
    swapped = [e for e in eps if e.name != eps[0].name] + [T.synthetic(99, eps[0].fold, "D" if eps[0].fold != "ab5" else "C", "corr")]
    with pytest.raises(D1.DataContractError):
        D2.check_roster(swapped, roster=roster)


def test_check_roster_only_checks_the_requested_groups():
    eps = T.synthetic_set()
    roster = T.roster_of(eps)
    part = [e for e in eps if e.fold == "ab6"]
    D2.check_roster(part, keys=["ab6/A", "ab6/D"], roster=roster)
    with pytest.raises(D1.DataContractError):
        D2.check_roster(part, keys=["ab6/A", "ab6/D", "ab7/D"], roster=roster)


def test_spec_for_selects_arms_by_fold_and_kind():
    assert D2.spec_for(("ab6",), ("bare", "corr")) == [({"ab6": ("A",)}, "bare"), ({"ab6": ("D",)}, "corr")]
    assert D2.spec_for(("ab5",), ("aux",)) == [({"ab4": ("P2", "P3")}, "aux")]


@pytest.mark.slow
def test_real_data_matches_the_roster_and_contracts():
    eps = D2.load()                                                 # 核对名单与日志契约
    assert {k: len(v) for k, v in D2.groups(eps).items()} == EXPECTED
    assert all(e.da.shape == (e.T, 6) for e in eps)
    assert all((e.da == 0).all() for e in eps if e.kind == "bare")
    for f in D2.FOLDS:                                              # 近期裸跑 = 阶段一冻结名单里的同一批集
        arm = D2.BARE[f][0]
        assert sorted(e.name for e in eps if e.fold == f and e.kind == "bare") == sorted(D1.ROSTER[f"{f}/{arm}"])
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_q2_data.py -q`
Expected: 收集时报 `ModuleNotFoundError: No module named 'q1_data'`（`conftest.py` 还没建，`sim/q1` 不在搜索路径上）

- [ ] **Step 3: 实现**

`sim/q2/conftest.py`：

```python
"""`sim/q2` 的 pytest 配置。MuJoCo + scipy 共存时 GC 会 SIGSEGV ⇒ 导入即关 GC；`sim/q1` 只读复用。"""
import gc
import sys
from pathlib import Path

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 读真实数据或跑 MuJoCo/G-2 的集成测试")
```

`sim/q2/q2_data.py`：

```python
"""阶段二（Q2 开发检验）数据层 —— `reports/PREREG_Q2_STAGE2.md` §1。

近期装置三个会话（ab5 / ab6 / ab7）的裸跑与补正运行，以及 ab4 P2/P3 辅助参考 → `Q2Episode`。
标签、渲染与阶段一的数据契约全部沿用 `sim/q1/q1_data.py`（只读复用）；本模块只加：
- `da`：日志逐帧的 `da`，即**日志记录的请求补正量**（不是执行位移）；
- 日志契约：日志动作 a = 名义指令 u + δa（|差| ≤ 1e-6；只证明日志的组合关系，不证明设备执行了同样的补正）；
  裸跑的 δa 必须全为 0，补正运行与探针必须有非零 δa；
- 冻结的数据名单（`q2_roster.json`，在代码指纹里），按「会话/臂」逐集名核对。
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import q1_data as D1           # noqa: E402

ROOT = D1.ROOT
FOLDS = ("ab5", "ab6", "ab7")
BARE = {"ab5": ("A",), "ab6": ("A",), "ab7": ("Apost",)}
CORR = {"ab5": ("C", "Csafe"), "ab6": ("D",), "ab7": ("D",)}
AUX = {"ab4": ("P2", "P3")}
LOG_TOL = 1e-6
ROSTER_FILE = HERE / "q2_roster.json"
ROSTER = json.loads(ROSTER_FILE.read_text()) if ROSTER_FILE.exists() else {}


class LogContractError(D1.DataContractError):
    """日志不满足阶段二的契约（a = u + δa、裸跑 δa = 0、补正运行 δa 非零）。不被加载循环吞掉。"""


@dataclass
class Q2Episode(D1.Q1Episode):
    fold: str = ""             # 会话：ab5 / ab6 / ab7 / ab4
    arm: str = ""              # A / Apost / C / Csafe / D / P2 / P3
    kind: str = ""             # bare / corr / aux
    da: np.ndarray = None      # (T, 6) 日志记录的请求补正量（rad）


def log_da(path, T: int, kind: str) -> np.ndarray:
    """从日志读 δa，并核对日志契约 → (T, 6)。"""
    pay = json.loads(Path(path).read_text())
    fr = pay["frames"]
    a = np.asarray([f["a"] for f in fr], np.float64)[:, :6]
    da = np.asarray([f["da"] for f in fr], np.float64)
    chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
    raw = np.concatenate([np.asarray(c["raw"], np.float64) for c in chunks])[:len(fr), :6]
    if da.shape != (T, 6) or not np.isfinite(da).all():
        raise LogContractError(f"{path}: δa 形状 {da.shape} 或含非有限值")
    gap = float(np.abs(a - raw - da).max())
    if gap > LOG_TOL:
        raise LogContractError(f"{path}: 日志动作 a ≠ 名义指令 + δa（最大差 {gap:.2e}）")
    nz = bool(np.abs(da).max() > 0)
    if kind == "bare" and nz:
        raise LogContractError(f"{path}: 裸跑的 δa 应全为 0")
    if kind in ("corr", "aux") and not nz:
        raise LogContractError(f"{path}: 补正运行 / 探针的 δa 全为 0")
    return da


def from_q1(e1, fold: str, arm: str, kind: str, da) -> Q2Episode:
    base = {f.name: getattr(e1, f.name) for f in fields(D1.Q1Episode)}
    return Q2Episode(**base, fold=fold, arm=arm, kind=kind, da=np.asarray(da, np.float64))


def load_episode(path, fold: str, arm: str, kind: str) -> Q2Episode:
    """一集 → `Q2Episode`。帧数不足 / 缺 IMU 抛 `ValueError`（同阶段一筛选）；违反契约抛 `DataContractError`。"""
    e1 = D1.load_episode(path, f"{fold}/{arm}")
    return from_q1(e1, fold, arm, kind, log_da(path, e1.T, kind))


def arm_paths(fold: str, arm: str) -> list:
    return sorted((ROOT / "outputs" / fold).glob(f"{arm}_ep*.json"), key=lambda q: int(q.stem.split("ep")[-1]))


def load_arms(spec: dict, kind: str) -> list:
    out = []
    for fold, arms in spec.items():
        for arm in arms:
            for p in arm_paths(fold, arm):
                try:
                    out.append(load_episode(p, fold, arm, kind))
                except ValueError:
                    continue
    return out


def spec_for(folds=FOLDS, kinds=("bare", "corr", "aux")) -> list:
    """→ [(spec, kind)]：给定会话与类别要加载的臂。"""
    out = []
    if "bare" in kinds:
        out.append(({f: BARE[f] for f in folds if f in BARE}, "bare"))
    if "corr" in kinds:
        out.append(({f: CORR[f] for f in folds if f in CORR}, "corr"))
    if "aux" in kinds:
        out.append((AUX, "aux"))
    return out


def load(folds=FOLDS, kinds=("bare", "corr", "aux"), check=True) -> list:
    """加载指定会话与类别的集；`check` 时按冻结名单核对这些「会话/臂」组。"""
    eps = []
    for spec, kind in spec_for(folds, kinds):
        eps += load_arms(spec, kind)
    if check:
        keys = [f"{f}/{a}" for spec, _ in spec_for(folds, kinds) for f, arms in spec.items() for a in arms]
        check_roster(eps, keys)
    return eps


def groups(eps) -> dict:
    g = {}
    for e in eps:
        g.setdefault(f"{e.fold}/{e.arm}", []).append(e.name)
    return g


def check_roster(eps, keys=None, roster=None) -> None:
    """载入的集必须与冻结名单逐组、逐集名一致（`keys` 指定要核对的「会话/臂」组；缺省 = 名单里的全部组）。"""
    roster = ROSTER if roster is None else roster
    keys = sorted(roster) if keys is None else sorted(keys)
    got = groups(eps)
    bad = []
    for k in sorted(set(keys) | (set(got) - set(roster))):
        a, b = got.get(k, []), roster.get(k, [])
        if sorted(a) != sorted(b):
            bad.append(f"{k}: 缺 {sorted(set(b) - set(a))[:3]} 多 {sorted(set(a) - set(b))[:3]}（{len(a)} 集，应为 {len(b)}）")
    if bad:
        raise D1.DataContractError("数据与阶段二冻结名单不符：" + "；".join(bad))


def select(eps, fold=None, kind=None) -> list:
    return [e for e in eps if (fold is None or e.fold == fold) and (kind is None or e.kind == kind)]
```

`sim/q2/q2_testing.py`：

```python
"""测试用合成集（不读真实数据）：阶段一的合成集 + δa（补正运行 / 探针为小随机序列，裸跑为 0）。"""
from __future__ import annotations
import numpy as np
import q1_testing as T1
import q2_data as D2


def synthetic(seed: int = 0, fold: str = "ab5", arm: str = "C", kind: str = "corr", T: int = 587) -> D2.Q2Episode:
    e1 = T1.synthetic_episode(seed, T=T, sess=f"{fold}/{arm}")
    e1.name = f"{fold}/{arm}_ep{seed}"
    rng = np.random.default_rng(1000 + seed)
    da = np.zeros((e1.T, 6)) if kind == "bare" else rng.normal(0.0, 1e-3, (e1.T, 6))
    if kind != "bare":
        da[:, 3] = 0.0                                          # 同真实数据：j4 不补正
    return D2.from_q1(e1, fold, arm, kind, da)


def synthetic_set(n_per: int = 2) -> list:
    """3 个近期会话 × {裸跑, 补正} × n_per 集 + ab4 P2/P3 各 n_per 集。"""
    out, s = [], 0
    for fold in D2.FOLDS:
        for arm in D2.BARE[fold]:
            for _ in range(n_per):
                out.append(synthetic(s, fold, arm, "bare")); s += 1
        for arm in D2.CORR[fold][:1]:
            for _ in range(n_per):
                out.append(synthetic(s, fold, arm, "corr")); s += 1
    for arm in D2.AUX["ab4"]:
        for _ in range(n_per):
            out.append(synthetic(s, "ab4", arm, "aux")); s += 1
    return out


def roster_of(eps) -> dict:
    return {k: sorted(v) for k, v in D2.groups(eps).items()}
```

- [ ] **Step 4: 生成冻结名单** —— `sim/q2/q2_roster.json`

名单由真实数据生成一次，此后冻结在代码指纹里。逐组按集号排序：

```bash
$PY - <<'EOF'
import gc, json, sys; gc.disable(); sys.path[:0] = ["sim/q1", "sim/q2"]
import q2_data as D2
eps = D2.load(check=False)
g = {k: sorted(v, key=lambda n: int(n.split("ep")[-1])) for k, v in sorted(D2.groups(eps).items())}
open("sim/q2/q2_roster.json", "w").write(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
print(len(eps), {k: len(v) for k, v in g.items()})
EOF
sha256sum sim/q2/q2_roster.json
```

Expected:
- 打印 `183 {'ab4/P2': 6, 'ab4/P3': 4, 'ab5/A': 40, 'ab5/C': 40, 'ab5/Csafe': 3, 'ab6/A': 30, 'ab6/D': 30, 'ab7/Apost': 15, 'ab7/D': 15}`；
- sha256 以 `a6ee1487639137b1` 开头（镜像里生成的同一文件）。

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest sim/q2/test_q2_data.py -q -m "not slow"` → Expected: `9 passed, 1 deselected`
Run: `$PY -m pytest sim/q2/test_q2_data.py -q -m slow` → Expected: `1 passed`，约 80 s（读 183 集、核对名单与日志契约，并核对近期裸跑与阶段一冻结名单是同一批集）

- [ ] **Step 6: 提交**

```bash
git add sim/q2/conftest.py sim/q2/q2_data.py sim/q2/q2_testing.py sim/q2/q2_roster.json sim/q2/test_q2_data.py
git commit -m "阶段二 Q2：数据层（183 集 → Q2Episode；日志契约 a = u + δa；冻结名单）"
```

---

### Task 2: δa 时间轴与固定 H

**Files:**
- Create: `sim/q2/q2_windows.py`
- Test: `sim/q2/test_q2_windows.py`

**Interfaces:**
- Consumes：
  - `q1_data.ROOT`；`q1_windows.Window(t0, t1, s1)`（`n_scored`）
  - `fit_step1_candidate.causal_conv(H, x)`
  - `outputs/step1_H/step1_H_v1.npz`（键 `H`，形状 (16, 3, 6)）
- Produces:
  - 常量：`H_FILE`、`H_SHA16 = "89d118d4b69837f0"`
  - `da_visible(ep, w, upto) -> (upto+1, 6)`：超出 t1 或集尾的行取 0
  - `da_span(ep, w, lo, hi) -> (hi−lo+1, 6)`：< 0 的行取 0
  - `load_h() -> (16, 3, 6) float64`：核对冻结指纹
  - `h_full(H, ep) -> (T, 3)`：整集因果卷积，零历史只在集首
  - `h_window(H, ep, w) -> (n_scored, 3)`

- [ ] **Step 1: 写失败的测试** —— `sim/q2/test_q2_windows.py`

```python
import copy

import numpy as np
import pytest

import q1_windows as W
import q2_testing as T
import q2_windows as W2


def test_da_visible_zeroes_rows_after_t1_and_after_the_episode_end():
    e = T.synthetic(0)
    w = W.chunk_windows(e)[1]                                        # t0 = 100, t1 = 199
    v = W2.da_visible(e, w, w.t1 + 5)
    assert np.array_equal(v[:w.t1 + 1], e.da[:w.t1 + 1]) and (v[w.t1 + 1:] == 0).all()
    last = W.chunk_windows(e)[-1]                                    # t1 = 599 > T − 1
    v = W2.da_visible(e, last, last.t1)
    assert v.shape == (last.t1 + 1, 6) and np.array_equal(v[:e.T], e.da) and (v[e.T:] == 0).all()


def test_da_span_pads_frames_before_the_episode_start():
    e = T.synthetic(1)
    w = W.chunk_windows(e)[0]
    s = W2.da_span(e, w, -3, 2)
    assert (s[:3] == 0).all() and np.array_equal(s[3:], e.da[:3])


def test_h_window_matches_the_gap_axis_formula_without_shift():
    e = T.synthetic(2)
    H = np.random.default_rng(0).normal(0, 1, (16, 3, 6))
    w = W.chunk_windows(e)[2]
    got = W2.h_window(H, e, w)
    t = w.t0 + 7
    want = sum(H[tau] @ e.da[t - tau] for tau in range(16))         # Δê_t = Σ_τ H[τ] δa_{t−τ}，不再平移 L=3
    assert got.shape == (w.n_scored, 3) and np.allclose(got[7], want)


def test_h_window_uses_past_corrections_and_never_future_ones():
    e = T.synthetic(3)
    H = np.random.default_rng(1).normal(0, 1, (16, 3, 6))
    w = W.chunk_windows(e)[2]
    base = W2.h_window(H, e, w)
    e1 = copy.copy(e); d = e.da.copy(); d[w.t1 + 1:] += 0.01; e1.da = d     # 改下一 chunk 的补正 ⇒ 不变
    assert np.array_equal(W2.h_window(H, e1, w), base)
    e2 = copy.copy(e); d = e.da.copy(); d[w.t0 - 3] += 0.01; e2.da = d      # 改过去的补正 ⇒ 窗口开头变（不清零重启）
    assert not np.allclose(W2.h_window(H, e2, w)[:3], base[:3])


def test_load_h_checks_the_frozen_fingerprint():
    H = W2.load_h()
    assert H.shape == (16, 3, 6)
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_q2_windows.py -q`
Expected: `ModuleNotFoundError: No module named 'q2_windows'`

- [ ] **Step 3: 实现** —— `sim/q2/q2_windows.py`

```python
"""阶段二的 δa 时间轴规则与固定 H 的窗口响应 —— 预注册 §2、§3.5（只在这里实现）。

δa 的可见性（控制周期索引，与指令同一个时间轴）：
- 过去段 t0−19 … t0：日志记录的请求补正量；
- 当前 chunk t0+1 … t1：离线条件预测使用日志中的当前 chunk 补正序列（离线给定的条件，不代表历史部署时在起点就知道全部补正）；
- 超出 t1（以及超出集尾）：取 0。
固定 H：沿用原有的 gap 时间轴 `Δê_t = Σ_{τ=0}^{15} H[τ] · δa_{t−τ}`，不再额外平移 L = 3；卷积包含过去的 δa，不在每个 chunk 清零重启。
第 t 个误差只用到第 t 周期为止的 δa，而计分的 t ≤ t1 ⇒ 始终可见。
"""
from __future__ import annotations
import hashlib

import numpy as np

import q1_data as D1

H_FILE = D1.ROOT / "outputs/step1_H/step1_H_v1.npz"
H_SHA16 = "89d118d4b69837f0"               # float64 sha256 前 16 位（step1-linear-v1 同一个核）


def da_visible(ep, w, upto: int) -> np.ndarray:
    """δa 第 0 … upto 行；超出 t1 或超出集尾的行取 0。"""
    out = np.zeros((upto + 1, 6))
    n = min(upto, w.t1, ep.T - 1) + 1
    if n > 0:
        out[:n] = np.asarray(ep.da, np.float64)[:n]
    return out


def da_span(ep, w, lo: int, hi: int) -> np.ndarray:
    """第 lo … hi 行（含）的可见 δa；< 0 的行取 0。"""
    vis = da_visible(ep, w, max(hi, 0))
    k = np.arange(lo, hi + 1)
    out = np.zeros((len(k), 6))
    ok = k >= 0
    out[ok] = vis[k[ok]]
    return out


def load_h() -> np.ndarray:
    z = np.load(H_FILE, allow_pickle=False)
    H = np.ascontiguousarray(z["H"], np.float64)
    got = hashlib.sha256(H.tobytes()).hexdigest()[:16]
    if got != H_SHA16:
        raise RuntimeError(f"{H_FILE}: 固定 H 的指纹 {got} ≠ 冻结值 {H_SHA16}")
    return H


def h_full(H, ep) -> np.ndarray:
    """整集 gap 索引上的 `Σ_τ H[τ]·δa_{t−τ}` → (T, 3)（零历史只在集首，不在 chunk 边界）。"""
    import fit_step1_candidate as S1
    return S1.causal_conv(np.asarray(H, np.float64), np.asarray(ep.da, np.float64))


def h_window(H, ep, w) -> np.ndarray:
    """窗口内 t ∈ [t0, s1] 的 H 响应 → (n_scored, 3)。"""
    return h_full(H, ep)[w.t0:w.s1 + 1]
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q2/test_q2_windows.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q2/q2_windows.py sim/q2/test_q2_windows.py
git commit -m "阶段二 Q2：δa 时间轴（过去段 + 当前 chunk 离线条件、超出 t1 取 0）与固定 H（含过去 δa、不平移）"
```

---

### Task 3: 两种响应形式

**Files:**
- Create: `sim/q2/q2_models.py`
- Test: `sim/q2/test_q2_models.py`

**Interfaces:**
- Consumes：
  - `q1_e2e`：`window_input(ep, w, st, measured)`、`make_tcn(ch_in)`（`.body` 为主干）、`K_OUT = 102`、`CH_A = 13`、`L_SEQ = 122`
  - `q1_data.HIST = 20`；`q2_windows.da_span`
- Produces:
  - 常量：`FORMS = ("nonlinear", "condlinear")`、`N_BASIS = 4`、`N_LAG = 20`、`BASIS_SD = 0.05`、`SCALE_FLOOR = 1e-6`
  - `da_scale(eps) -> (6,)`：逐关节均方根，不减均值，下限 1e-6
  - `window_inputs(ep, windows, st, scale) -> (X (B,122,13) float32, Dd (B,122,6) float32)`：序列帧 t0−19 … t0+102
  - `make_nonlinear()`、`make_condlinear()`、`make(form)`
  - `increment(model, form, X, Dd) -> torch (B,122,9)`：归一化单位的增量（关节 6 + 误差 3）
  - `decode_increment(out, windows, st_nom) -> ([Δê (n,3)], [Δq̂ (n,6)])`：位置同 `q1_e2e.decode`（帧 t+3），尺度乘以冻结名义模型的 `s_gap` / `s_eps`

- [ ] **Step 1: 写失败的测试** —— `sim/q2/test_q2_models.py`

```python
import copy

import numpy as np
import pytest
import torch

import q1_data as D1
import q1_e2e as E
import q1_windows as W
import q2_models as Mo
import q2_testing as T


def _st(eps):
    return E.norm_stats(eps)


def _jitter(m, s=0.05):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n != "bases":
                p.add_(torch.randn_like(p) * s)
    return m


def test_da_scale_preserves_zero_and_uses_rms_not_std():
    e = T.synthetic(0)
    e.da = np.zeros_like(e.da); e.da[:, 0] = 2e-3                     # 常数补正：std = 0，RMS = 2e-3
    s = Mo.da_scale([e])
    assert np.isclose(s[0], 2e-3) and s[3] == Mo.SCALE_FLOOR           # j4 恒为 0 ⇒ 下限，0 仍是 0
    assert (np.asarray(e.da) / s)[:, 3].max() == 0


def test_window_inputs_shapes_and_visibility():
    eps = [T.synthetic(s) for s in range(3)]
    st, sc = _st(eps), Mo.da_scale(eps)
    e = eps[0]
    ws = W.chunk_windows(e)
    X, Dd = Mo.window_inputs(e, ws, st, sc)
    assert X.shape == (len(ws), E.L_SEQ, E.CH_A) and Dd.shape == (len(ws), E.L_SEQ, 6)
    w = ws[1]                                                          # 序列帧 = t0−19 … t0+102
    seq = np.arange(w.t0 - D1.HIST + 1, w.t0 + E.K_OUT + 1)
    inside = (seq >= 0) & (seq <= w.t1)
    assert np.allclose(Dd[1][inside], (e.da[seq[inside]] / sc).astype(np.float32))
    assert (Dd[1][~inside] == 0).all()


@pytest.mark.parametrize("form", Mo.FORMS)
def test_zero_init_is_zero_response(form):
    eps = [T.synthetic(s) for s in range(2)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = Mo.make(form)
    X, Dd = Mo.window_inputs(eps[0], W.chunk_windows(eps[0]), st, sc)
    with torch.no_grad():
        assert (Mo.increment(m, form, X, Dd) == 0).all()


@pytest.mark.parametrize("form", Mo.FORMS)
def test_zero_correction_gives_exactly_zero_increment(form):
    eps = [T.synthetic(s) for s in range(2)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = _jitter(Mo.make(form))
    X, Dd = Mo.window_inputs(eps[0], W.chunk_windows(eps[0]), st, sc)
    with torch.no_grad():
        assert (Mo.increment(m, form, X, np.zeros_like(Dd)) == 0).all()
        assert Mo.increment(m, form, X, Dd).abs().max() > 0


def test_condlinear_is_exactly_linear_in_the_correction():
    eps = [T.synthetic(s) for s in range(2)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = _jitter(Mo.make("condlinear")).double()
    X, D1_ = Mo.window_inputs(eps[0], W.chunk_windows(eps[0]), st, sc)
    D2_ = np.random.default_rng(0).normal(size=D1_.shape)
    X = X.astype(np.float64)
    with torch.no_grad():
        f = lambda d: Mo.increment(m, "condlinear", X, d.astype(np.float64))
        assert torch.allclose(f(2.0 * D1_ - 3.0 * D2_), 2.0 * f(D1_) - 3.0 * f(D2_), atol=1e-10)


def test_condlinear_matches_the_explicit_kernel_formula_and_can_go_negative():
    torch.manual_seed(0)
    m = Mo.make("condlinear").double()
    with torch.no_grad():
        m.mix.bias.copy_(torch.tensor([-1.0, 0.5, 0.0, 2.0], dtype=torch.float64))   # 可正可负（不用 softmax）
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1, 40, E.CH_A)); Dd = rng.normal(size=(1, 40, 6))
    with torch.no_grad():
        got = Mo.increment(m, "condlinear", X, Dd)[0].numpy()
        a = m.mix(m.body(torch.as_tensor(X).transpose(1, 2)))[0].numpy()             # (R, L)
    B = m.bases.detach().numpy()
    i = 30
    want = sum(a[r, i] * sum(B[r, tau] @ Dd[0, i - tau] for tau in range(Mo.N_LAG)) for r in range(Mo.N_BASIS))
    assert np.allclose(got[i], want)


@pytest.mark.parametrize("form", Mo.FORMS)
def test_increment_is_causal_over_the_sequence(form):
    torch.manual_seed(0)
    m = _jitter(Mo.make(form))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1, E.L_SEQ, E.CH_A)).astype(np.float32); Dd = rng.normal(size=(1, E.L_SEQ, 6)).astype(np.float32)
    X2, D2_ = X.copy(), Dd.copy(); X2[:, 60:] += 1.0; D2_[:, 60:] += 1.0
    with torch.no_grad():
        assert torch.equal(Mo.increment(m, form, X, Dd)[:, :60], Mo.increment(m, form, X2, D2_)[:, :60])


@pytest.mark.parametrize("form", Mo.FORMS)
def test_increment_ignores_next_chunk_commands_and_corrections(form):
    eps = [T.synthetic(s) for s in range(3)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = _jitter(Mo.make(form))
    e = eps[2]
    ws = W.chunk_windows(e)
    w = ws[2]

    def pred(ep):
        X, Dd = Mo.window_inputs(ep, ws, st, sc)
        with torch.no_grad():
            dE, dQ = Mo.decode_increment(Mo.increment(m, form, X, Dd), ws, st)
        return dE[2].numpy(), dQ[2].numpy()

    bE, bQ = pred(e)
    e1 = copy.copy(e)
    c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    d = e.da.copy(); d[w.t1 + 1:] += 0.01; e1.da = d
    i_ = e.ideal.copy(); i_[w.t1 + 1:] += 0.5; e1.ideal = i_
    E1, Q1 = pred(e1)
    assert np.array_equal(E1, bE) and np.array_equal(Q1, bQ)


def test_decode_increment_positions_and_nominal_scales():
    eps = [T.synthetic(s) for s in range(2)]
    st = _st(eps)
    ws = W.chunk_windows(eps[0])
    out = torch.zeros(len(ws), E.L_SEQ, 9)
    out[0, 2 + D1.HIST, :] = 1.0                                       # 第一个计分帧 t0（序列里是 t0+3）
    dE, dQ = Mo.decode_increment(out, ws, st)
    assert np.allclose(dE[0][0].numpy(), st["s_gap"]) and np.allclose(dQ[0][0].numpy(), st["s_eps"])
    assert (dE[0][1:] == 0).all() and dE[0].shape == (ws[0].n_scored, 3)


def test_parameter_counts():
    torch.manual_seed(0)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(Mo.make("nonlinear")) == 68_361 and n(Mo.make("condlinear")) == 70_436
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_q2_models.py -q`
Expected: `ModuleNotFoundError: No module named 'q2_models'`

- [ ] **Step 3: 实现** —— `sim/q2/q2_models.py`

```python
"""阶段二的两种响应形式 —— 预注册 §3.2–3.4。

两者都输出关节 6 维 + 误差 3 维的**增量**（归一化单位；乘以冻结名义模型的 s_ε、s_gap 才是物理量）。
- 非线性差分：`N(x, δa) − N(x, 0)`。N 与 2a 同构，输入 = 2a 的 13 个通道 + 6 个 δa 通道；输出头零初始化 ⇒ 起点 = 零响应，
  δa = 0 时增量严格为 0。
- 条件线性：`Δy_k = Σ_{r=1}^{4} a_r(k) · Σ_{τ=0}^{19} B_r[τ] · δa_{k−τ}`（控制周期索引）。a_r 由与 2a 同构的指令网络给出（不看 δa），
  **可正可负的普通输出、不用 softmax**，输出头零初始化 ⇒ 起点 = 零响应；基核 B_r 每个元素取 N(0, 0.05²)，由训练种子决定。
  对 δa 严格线性；20 帧限制的是 δa 的记忆，指令网络仍有 61 帧感受野。
δa 归一化保留零点：每个关节只除以训练数据上 δa 的均方根（不减均值），下限 1e-6 rad。
"""
from __future__ import annotations
import numpy as np

import q1_data as D1
import q1_e2e as E
import q2_windows as W2

FORMS = ("nonlinear", "condlinear")
N_BASIS = 4
N_LAG = 20
BASIS_SD = 0.05
SCALE_FLOOR = 1e-6


def da_scale(eps) -> np.ndarray:
    """训练数据上逐关节 δa 的均方根（不减均值），下限 SCALE_FLOOR → (6,)。"""
    X = np.concatenate([np.asarray(e.da, np.float64)[:e.T] for e in eps])
    return np.maximum(np.sqrt((X ** 2).mean(0)), SCALE_FLOOR)


def window_inputs(ep, windows, st: dict, scale) -> tuple:
    """→ (X 指令侧 (B, L_SEQ, 13) float32, Dd 归一化 δa (B, L_SEQ, 6) float32)。序列帧 = t0−19 … t0+102。"""
    X = np.stack([E.window_input(ep, w, st, False) for w in windows])
    Dd = np.stack([W2.da_span(ep, w, w.t0 - D1.HIST + 1, w.t0 + E.K_OUT) / np.asarray(scale) for w in windows])
    return X, Dd.astype(np.float32)


def make_nonlinear():
    return E.make_tcn(E.CH_A + 6)


def make_condlinear():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CondLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = E.make_tcn(E.CH_A).body                       # 与 2a 同构的指令网络主干（61 帧感受野）
            self.mix = nn.Conv1d(64, N_BASIS, 1)                      # 混合权重：普通输出，不用 softmax
            nn.init.zeros_(self.mix.weight)
            nn.init.zeros_(self.mix.bias)
            self.bases = nn.Parameter(torch.randn(N_BASIS, N_LAG, 9, 6) * BASIS_SD)

        def forward(self, x, d):                                      # x (B,L,13)、d (B,L,6) → (B,L,9)
            a = self.mix(self.body(x.transpose(1, 2)))                # (B,R,L)
            w = self.bases.flip(1).permute(0, 2, 3, 1)                # (R,9,6,LAG)：w[r,o,j,m] = B[r, LAG−1−m, o, j]
            dt = F.pad(d.transpose(1, 2), (N_LAG - 1, 0))             # 因果：只看本帧及之前的 δa
            resp = torch.stack([F.conv1d(dt, w[r]) for r in range(N_BASIS)], 1)    # (B,R,9,L)
            return (a.unsqueeze(2) * resp).sum(1).transpose(1, 2)

    return CondLinear()


def make(form: str):
    if form not in FORMS:
        raise ValueError(f"未知的响应形式 {form}")
    return make_nonlinear() if form == "nonlinear" else make_condlinear()


def increment(model, form: str, X, Dd):
    """归一化单位的增量 (B, L, 9)。非线性差分减去 δa = 0 的输出 ⇒ δa = 0 时严格为 0。"""
    import torch
    p = next(model.parameters())
    x = torch.as_tensor(X, dtype=p.dtype)
    d = torch.as_tensor(Dd, dtype=p.dtype)
    if form == "nonlinear":
        return model(torch.cat([x, d], -1)) - model(torch.cat([x, torch.zeros_like(d)], -1))
    return model(x, d)


def decode_increment(out, windows, st_nom: dict):
    """(B, L_SEQ, 9) → 每窗 (Δê (n_scored, 3), Δq̂ (n_scored, 6))；位置同 `q1_e2e.decode`（帧 t+3），
    尺度沿用冻结名义模型的 s_gap、s_ε。"""
    import torch
    s_e = torch.as_tensor(st_nom["s_gap"], dtype=out.dtype)
    s_q = torch.as_tensor(st_nom["s_eps"], dtype=out.dtype)
    dE, dQ = [], []
    for b, w in enumerate(windows):
        i = np.arange(w.n_scored) + 2 + D1.HIST
        dE.append(out[b, i, 6:9] * s_e)
        dQ.append(out[b, i, :6] * s_q)
    return dE, dQ
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q2/test_q2_models.py -q`
Expected: `14 passed`（含参数量 68,361 / 70,436）

- [ ] **Step 5: 提交**

```bash
git add sim/q2/q2_models.py sim/q2/test_q2_models.py
git commit -m "阶段二 Q2：两种响应形式（非线性差分；条件线性 4 基核 × 20 滞后、普通输出、零初始化）与保零点的 δa 尺度"
```

---

### Task 4: 训练、预测与计分

**Files:**
- Create: `sim/q2/q2_train.py`
- Test: `sim/q2/test_q2_train.py`

**Interfaces:**
- Consumes：
  - `q1_train`：`train_one(model, eps, seed, ckpts, log)`、`forward_windows`、`_determinism(seed)`、`with_weights`、`state_dict_sha`
  - `q1_e2e.norm_stats`；`q1_losses`：`loss_norms`、`joint_loss`、`omega_loss`
  - `q1_windows`：`chunk_windows`、`stitch`、`score`、`lead_score`、`P`
  - Task 2、Task 3 的全部接口
- Produces:
  - `CONFIG`：停止规则、选步规则、门槛（`gate_d = 0.05`、`nonlin_margin = 0.02`）、`budget_hours = 24`；`CODE_FILES_Q2`、`CODE_FILES_Q1`、`Q2_DIR`
  - 指纹与选步：`code_sha() -> str`（64 位）、`config_hash() -> str`（16 位）、`pick_step(V) -> int`
  - 名义：
    - `train_nominal(bare_eps, seed, ckpts=None, log=None) -> dict`（= `train_one("2a", …)`）
    - `nominal_windows(model, st, ep, windows)`、`nominal_episode(model, st, ep) -> (n,3)`
  - 固定 H：`h_episode(H, ep) -> (n,3)`
  - 增量：
    - `train_increment(form, nom, corr_eps, seed, ckpts=None, log=None) -> {"model","st_inc","scale","norms","loss","sec","ckpt"}`
    - `with_increment(form, state_dict)`、`increment_episode(form, model, st_inc, scale, nom_st, ep) -> (n,3)`
  - 计分：
    - `ep_metrics(ep, ehat, K) -> {"name","fold","arm","num","den","f3","lead"}`
    - `pooled_rho(rows) -> float`
    - `beta_parts(ep, e_nom, d_e, K) -> (num, den)`
- 本任务的 `CODE_FILES_Q2` 只列 5 个已有文件；Task 5、Task 6 再分别加入 `q2_run.py`、`q2_report.py`。

- [ ] **Step 1: 写失败的测试** —— `sim/q2/test_q2_train.py`

```python
import copy

import numpy as np
import pytest
import torch

import q1_windows as W
import q2_models as Mo
import q2_testing as T
import q2_train as TR


@pytest.fixture(scope="module")
def syn():
    bare = [T.synthetic(s, "ab6", "A", "bare") for s in range(3)]
    corr = [T.synthetic(10 + s, "ab6", "D", "corr") for s in range(3)]
    nom = TR.train_nominal(bare, 0, ckpts=(2,))
    return bare, corr, (nom["model"], nom["stats"])


def test_code_fingerprint_covers_the_roster_and_reused_stage1_code():
    assert "q2_roster.json" in TR.CODE_FILES_Q2 and "q1_train.py" in TR.CODE_FILES_Q1
    assert len(TR.code_sha()) == 64 and TR.code_sha() == TR.code_sha()


def test_config_hash_tracks_the_frozen_rules(monkeypatch):
    h = TR.config_hash()
    for key, val in (("gate_d", 0.04), ("nonlin_margin", 0.03), ("ckpts", (0, 10000)), ("basis_sd", 0.1)):
        monkeypatch.setitem(TR.CONFIG, key, val)
        assert TR.config_hash() != h
        monkeypatch.undo()
    assert TR.config_hash() == h


def test_nominal_is_stage1_train_one():
    import q1_train as TR1
    bare = [T.synthetic(s, "ab6", "A", "bare") for s in range(2)]
    a = TR.train_nominal(bare, 1, ckpts=(3,))
    b = TR1.train_one("2a", bare, 1, ckpts=(3,))
    assert TR1.state_dict_sha(a["ckpt"][3]) == TR1.state_dict_sha(b["ckpt"][3])


@pytest.mark.parametrize("form", Mo.FORMS)
def test_increment_training_is_reproducible_and_seed_dependent(form, syn):
    import q1_train as TR1
    _, corr, nom = syn
    a = TR.train_increment(form, nom, corr, 0, ckpts=(3,))
    b = TR.train_increment(form, nom, corr, 0, ckpts=(3,))
    c = TR.train_increment(form, nom, corr, 1, ckpts=(3,))
    sha = lambda r: TR1.state_dict_sha(r["ckpt"][3])
    assert sha(a) == sha(b) and a["loss"] == b["loss"] and sha(a) != sha(c)


@pytest.mark.parametrize("form", Mo.FORMS)
def test_checkpoints_are_prefixes_and_step_zero_is_zero_response(form, syn):
    import q1_train as TR1
    _, corr, nom = syn
    long = TR.train_increment(form, nom, corr, 0, ckpts=(0, 2, 4))
    short = TR.train_increment(form, nom, corr, 0, ckpts=(2,))
    assert TR1.state_dict_sha(long["ckpt"][2]) == TR1.state_dict_sha(short["ckpt"][2])
    e = corr[0]
    m0 = TR.with_increment(form, long["ckpt"][0])
    d0 = TR.increment_episode(form, m0, long["st_inc"], long["scale"], nom[1], e)
    assert (d0 == 0).all()                                            # 第 0 步 = 零响应
    m4 = TR.with_increment(form, long["ckpt"][4])
    assert np.abs(TR.increment_episode(form, m4, long["st_inc"], long["scale"], nom[1], e)).max() > 0


def test_increment_statistics_come_only_from_the_given_training_runs(syn):
    _, corr, nom = syn
    a = TR.train_increment("condlinear", nom, corr[:2], 0, ckpts=(1,))
    b = TR.train_increment("condlinear", nom, corr, 0, ckpts=(1,))
    assert not np.allclose(a["scale"], b["scale"]) and a["norms"]["c_w"] != b["norms"]["c_w"]
    assert np.allclose(a["scale"], Mo.da_scale(corr[:2]))


def test_predictor_shapes_and_h_equals_the_whole_episode_convolution(syn):
    import q2_windows as W2
    _, corr, nom = syn
    e = corr[1]
    H = np.random.default_rng(0).normal(0, 1, (16, 3, 6))
    assert TR.nominal_episode(*nom, e).shape == (e.n, 3)
    assert np.allclose(TR.h_episode(H, e), W2.h_full(H, e)[:e.n])


def test_metrics_and_beta_parts():
    e = T.synthetic(0)
    K = W.kernel()
    zero = np.zeros_like(e.gap)
    r = TR.ep_metrics(e, zero, K)
    assert r["num"] == r["den"] and set(r["lead"]) == {"3-6", "7-12", "13-27", "28-52", "53-96", "97-102"}
    assert (r["name"], r["fold"], r["arm"]) == (e.name, e.fold, e.arm)
    num, den = TR.beta_parts(e, zero, e.gap, K)                       # 增量 = 残差 ⇒ β = 1
    assert np.isclose(num / den, 1.0)
    num, den = TR.beta_parts(e, zero, zero, K)                        # 增量恒为 0 ⇒ 分母 0（β 不定义）
    assert den == 0.0


def test_pick_step_earliest_minimum_within_full_precision_ties():
    assert TR.pick_step({0: 0.5, 250: 0.4, 500: 0.4 + 1e-17}) == 250
    assert TR.pick_step({0: 0.4, 250: 0.4}) == 0
    assert TR.pick_step({10000: 0.1, 0: 0.2}) == 10000


def test_condlinear_learns_a_planted_linear_response():
    """合成：gap 里加入已知的线性响应 Hs ⊛ δa；条件线性训练后应在留出集上与之高度相关。"""
    import fit_step1_candidate as S1
    rng = np.random.default_rng(0)
    Hs = rng.normal(0, 30.0, (16, 3, 6))
    def plant(e):
        resp = S1.causal_conv(Hs, e.da)[:e.n]
        e.gap = e.gap + resp
        return e, resp
    bare = [T.synthetic(s, "ab6", "A", "bare") for s in range(2)]
    corr = [plant(T.synthetic(20 + s, "ab6", "D", "corr"))[0] for s in range(4)]
    held, resp = plant(T.synthetic(99, "ab6", "D", "corr"))
    nom = TR.train_nominal(bare, 0, ckpts=(1,))
    r = TR.train_increment("condlinear", (nom["model"], nom["stats"]), corr, 0, ckpts=(400,))
    d = TR.increment_episode("condlinear", r["model"], r["st_inc"], r["scale"], nom["stats"], held)
    P = lambda x: W.P(x, W.kernel())
    c = float((P(d) * P(resp)).sum() / np.sqrt((P(d) ** 2).sum() * (P(resp) ** 2).sum()))
    assert c > 0.5, c
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_q2_train.py -q`
Expected: `ModuleNotFoundError: No module named 'q2_train'`

- [ ] **Step 3: 实现** —— `sim/q2/q2_train.py`

```python
"""阶段二的训练、预测与计分 —— 预注册 §5、§6。

- 名义 2a：直接复用阶段一的 `train_one("2a", …)`（同结构、同损失、同确定性），只换训练数据（近期两个会话的裸跑）。
- 增量（两种形式）：名义冻结；只用给定的补正运行训练；损失作用在「名义 + 增量」的总预测上（ω 项 + 关节项，同阶段一）；
  δa 尺度、指令侧归一化统计与损失尺度只用给定的训练数据（按内层划分时就是那一个内层训练会话）。
- 选步规则同阶段一：检查点处快照，验证 ρ 的平均最小者，完整精度平局取最早；第 0 步 = 零响应，是有效选择。
CPU 单线程 + 确定性算法 ⇒ 同种子逐位复现；同种子下训到第 s 步的权重 = 长跑第 s 步的快照。
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import q1_data as D1
import q1_e2e as E
import q1_losses as LS
import q1_train as TR1
import q1_windows as W
import q2_models as Mo
import q2_windows as W2

CONFIG = {"max_steps": 10000,
          "ckpts": (0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000),
          "select": "inner two-way leave-one-session-out; min mean rho; earliest tie",
          "tie_rtol": 1e-12, "tie_atol": 1e-15,
          "lr": 1e-3, "clip": 1.0, "threads": 1, "device": "cpu",
          "seeds": (0, 1, 2), "forms": Mo.FORMS,
          "n_basis": Mo.N_BASIS, "n_lag": Mo.N_LAG, "basis_sd": Mo.BASIS_SD, "scale_floor": Mo.SCALE_FLOOR,
          "gate_d": 0.05, "nonlin_margin": 0.02,
          "budget_hours": 24}
CODE_FILES_Q2 = ("q2_data.py", "q2_roster.json", "q2_windows.py", "q2_models.py", "q2_train.py")
CODE_FILES_Q1 = ("q1_data.py", "q1_windows.py", "q1_losses.py", "q1_e2e.py", "q1_train.py", "q1_g2ref.py", "q1_run.py")
Q2_DIR = Path(__file__).resolve().parent


def code_sha() -> str:
    """阶段二代码 + 只读复用的阶段一代码的指纹。"""
    h = hashlib.sha256()
    for d, files in ((Q2_DIR, CODE_FILES_Q2), (D1.HERE, CODE_FILES_Q1)):
        for f in files:
            h.update(f.encode())
            h.update((d / f).read_bytes())
    return h.hexdigest()


def config_hash() -> str:
    """选模规则、门槛与训练配置 + 代码指纹。锁定的是规则，不是各作业选中的步数。"""
    blob = json.dumps({"config": CONFIG, "code": code_sha()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pick_step(V: dict) -> int:
    steps = sorted(V)
    vmin = min(V[s] for s in steps)
    tol = CONFIG["tie_atol"] + CONFIG["tie_rtol"] * abs(vmin)
    return int(min(s for s in steps if abs(V[s] - vmin) <= tol))


# ── 名义 ──────────────────────────────────────────────────────────────────
def train_nominal(bare_eps, seed: int, ckpts=None, log=None) -> dict:
    return TR1.train_one("2a", bare_eps, seed, ckpts=tuple(CONFIG["ckpts"] if ckpts is None else ckpts), log=log)


def nominal_windows(model, st, ep, windows) -> tuple:
    """冻结名义 2a 的逐窗 (ê, q̂)（numpy；无梯度）。"""
    import torch
    with torch.no_grad():
        Ee, Qq, _ = TR1.forward_windows("2a", model, st, ep, windows)
    return [x.double().numpy() for x in Ee], [x.double().numpy() for x in Qq]


def nominal_episode(model, st, ep) -> np.ndarray:
    ws = W.chunk_windows(ep)
    return W.stitch(ep.n, ws, nominal_windows(model, st, ep, ws)[0])


def h_episode(H, ep) -> np.ndarray:
    ws = W.chunk_windows(ep)
    return W.stitch(ep.n, ws, [W2.h_window(H, ep, w) for w in ws])


# ── 增量 ──────────────────────────────────────────────────────────────────
def train_increment(form: str, nom, corr_eps, seed: int, ckpts=None, log=None) -> dict:
    """nom = (冻结名义模型, 其统计量)。只用 corr_eps 训练增量；在每个检查点（含第 0 步）快照。"""
    import torch
    ckpts = tuple(CONFIG["ckpts"] if ckpts is None else ckpts)
    if list(ckpts) != sorted(set(ckpts)) or ckpts[0] < 0:
        raise ValueError(f"检查点须严格递增且非负：{ckpts}")
    nom_model, nom_st = nom
    TR1._determinism(seed)                                            # 先定种子再建模
    model = Mo.make(form)
    st_inc = E.norm_stats(corr_eps)
    scale = Mo.da_scale(corr_eps)
    norms = LS.loss_norms(corr_eps)
    data = []
    for e in corr_eps:                                                # 名义冻结 ⇒ 预测与输入只算一次
        ws = W.chunk_windows(e)
        En, Qn = nominal_windows(nom_model, nom_st, e, ws)
        X, Dd = Mo.window_inputs(e, ws, st_inc, scale)
        data.append((e, ws, torch.tensor(np.concatenate(En)), torch.tensor(np.concatenate(Qn)),
                     torch.tensor(X), torch.tensor(Dd)))
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    rng = np.random.default_rng(seed)
    snaps, hist, t_start = {}, [], time.time()

    def snap():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    if 0 in ckpts:
        snaps[0] = snap()
    for it in range(ckpts[-1]):
        e, ws, En, Qn, X, Dd = data[int(rng.integers(len(data)))]
        dE, dQ = Mo.decode_increment(Mo.increment(model, form, X, Dd), ws, nom_st)
        loss = (LS.joint_loss(e, Qn + torch.cat(dQ).double(), norms)
                + LS.omega_loss(e, En + torch.cat(dE).double(), norms))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip"])
        opt.step()
        hist.append(float(loss.item()))
        if it + 1 in ckpts:
            snaps[it + 1] = snap()
        if log and (it + 1) % 500 == 0:
            log(f"    {form} seed{seed} step {it + 1}/{ckpts[-1]} loss {np.mean(hist[-500:]):.5f} "
                f"{time.time() - t_start:.0f}s")
    return {"model": model, "st_inc": st_inc, "scale": scale, "norms": norms, "loss": hist,
            "sec": time.time() - t_start, "ckpt": snaps}


def with_increment(form: str, state_dict):
    model = Mo.make(form)
    model.load_state_dict(state_dict)
    return model


def increment_episode(form: str, model, st_inc, scale, nom_st, ep) -> np.ndarray:
    """拼接后的 Δê (n, 3)。"""
    import torch
    ws = W.chunk_windows(ep)
    X, Dd = Mo.window_inputs(ep, ws, st_inc, scale)
    with torch.no_grad():
        dE, _ = Mo.decode_increment(Mo.increment(model, form, X, Dd), ws, nom_st)
    return W.stitch(ep.n, ws, [x.double().numpy() for x in dE])


# ── 计分 ──────────────────────────────────────────────────────────────────
def ep_metrics(ep, ehat, K) -> dict:
    num, den, f3 = W.score(ep.gap, ehat, K)
    return {"name": ep.name, "fold": ep.fold, "arm": ep.arm, "num": num, "den": den, "f3": f3,
            "lead": W.lead_score(ep.gap, ehat, W.chunk_windows(ep), K)}


def pooled_rho(rows) -> float:
    return sum(r["num"] for r in rows) / sum(r["den"] for r in rows)


def beta_parts(ep, e_nom, d_e, K) -> tuple:
    """投影尺度 β 的一集分子与分母：⟨P(gap − ê_nom), P(Δê)⟩ 与 ‖P(Δê)‖²（汇总后再相除）。"""
    r, h = W.P(ep.gap - e_nom, K), W.P(d_e, K)
    return float((r * h).sum()), float((h * h).sum())
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q2/test_q2_train.py -q`
Expected: `12 passed`，约 8 s（含「条件线性能学到植入的线性响应」）

- [ ] **Step 5: 提交**

```bash
git add sim/q2/q2_train.py sim/q2/test_q2_train.py
git commit -m "阶段二 Q2：训练与计分（名义复用 train_one；增量在冻结名义上训练、第 0 步 = 零响应；选步与指纹）"
```

---

### Task 5: 命令行与产物

**Files:**
- Create: `sim/q2/q2_run.py`
- Modify: `sim/q2/q2_train.py:34`（`CODE_FILES_Q2` 加入 `q2_run.py`）
- Test: `sim/q2/test_q2_run.py`

**Interfaces:**
- Consumes：
  - Task 1–4；`q1_g2ref`：`load_g2()`、`episode_ehat(g2, ep, windows, R, b)`；`q1_data.calibration()`
  - `q1_run`：`atomic_json / atomic_npz / atomic_torch`、`file_sha`、`files_fp`、`fp_of`、`env_versions`、`DEP_FILES`、`G2_FILES`
- Produces（Task 6–8 依赖这些名字与字段）：
  - 常量：`PREREG`、`CALIB_FILES`（`outputs/step1_H/step1_H_v1.json`、`outputs/p2_labels/pp2a-deploy/manifest.json`）、`H_FILES`、`RUNNER_JOBS = 15`、`TIMING_MAX_STEPS = 30`、`PHASES = ("refs","nom-inner","nom-final","inc-inner","inc-final")`
  - 异常：`NotFrozenError`、`DeadlineExceeded`、`TimingRefused`（测速步数越界）；`require_frozen(prereg=None)`
  - 折与路径：
    - `train_sessions(fold)`、`inner_train(fold, val)`
    - `nom_inner_json(out, fold, val, seed)`、`nom_final_base(out, fold, seed)`
    - `inc_inner_json(out, form, fold, val, seed)`、`inc_final_base(out, form, fold, seed)`、`ref_json(out, fold)`
  - 输入与截止时间：
    - `roster_files()`、`inputs_file(out)`、`read_inputs(out)`、`current_inputs()`、`check_inputs(I)`
    - `deadline_file(out)`、`incomplete_file(out)`、`mark_incomplete(out, reason)`、`enforce_deadline(out, create=False)`
    - `_load(path)`、`is_done(path, inputs)`
    - `final_done(base, inputs)`：终训作业的完成判定 = JSON 的配置与输入指纹一致 **且** `.pt`、`.npz` 存在、哈希与 JSON 记录一致
  - 作业输入：`ref_inputs(I)`、`nom_inner_inputs(I, fold, val)`、`nom_final_inputs(out, I, fold, seed)`、`nominal_pt_sha(out, fold, seed)`、`inc_inner_inputs(out, I, form, fold, val, seed)`、`inc_final_inputs(out, I, form, fold, seed)`
  - 工具：`load_eps(folds, kinds)`、`load_nominal(out, fold, seed)`、`selection(paths, ckpts) -> (V, step*)`、`stamp(rec, inputs)`、`score_nominal(...)`、`score_increment(...)`、`jobs(out, phase)`
  - 命令：`cmd_prepare / cmd_refs / cmd_nom_inner / cmd_nom_final / cmd_inc_inner / cmd_inc_final / cmd_list_jobs / cmd_timing`；`main(argv)`（未冻结或测速步数越界 → 2，超时 → 3）
  - 产物字段（`stamp` 统一加 `config_hash, code_sha, inputs, input_fp, finished_at`）：
    - `refs/g2/{fold}.json`：`kind="g2", fold, episodes`
    - `nom/inner/seed{k}/{fold}__val_{val}.json`：`kind, fold, val, seed, train_sessions=[内层训练会话], scores={step: {rho, state_sha256}}, sec`
    - `nom/final/seed{k}/{fold}.json`：`kind, fold, seed, train_sessions, V, step_star, boundary, state_sha256, pt_sha256, npz_sha256, sec, q1, zero, H, aux_zero, aux_H, aux_H_beta`
      - `.pt`：`{state_dict, stats, step}`；`.npz`：键 `nom__<集名>`、`H__<集名>`（集名里的 `/` 换成 `__`）
    - `inc/{form}/inner/seed{k}/{fold}__val_{val}.json`：同名义内层，另有 `form`
    - `inc/{form}/final/seed{k}/{fold}.json`：`kind, form, fold, seed, train_sessions, V, step_star, boundary, state_sha256, pt_sha256, npz_sha256, sec, cand, aux_cand, aux_beta`
      - `.pt`：`{form, state_dict, st_inc, scale, step}`；`.npz`：键 `cand__<集名>`

- [ ] **Step 1: 写失败的测试** —— `sim/q2/test_q2_run.py`

```python
import json

import numpy as np
import pytest

import q1_data as D1
import q2_data as D2
import q2_run as RUN
import q2_testing as T
import q2_train as TR

TINY = (0, 1, 2)
ALARMS = []


@pytest.fixture()
def syn(monkeypatch, tmp_path):
    """合成数据：3 个近期会话 × {裸跑, 补正} × 2 集 + ab4 P2/P3 各 2 集；检查点缩成 (0,1,2)；闹钟只记录。"""
    import signal
    eps = T.synthetic_set(2)

    def fake_load(folds=D2.FOLDS, kinds=("bare", "corr", "aux"), check=True):
        out = [e for e in eps if e.kind in kinds and (e.kind == "aux" or e.fold in folds)]
        if check:
            D2.check_roster(out, keys=sorted({f"{e.fold}/{e.arm}" for e in out}))
        return out

    data = tmp_path / "data.json"
    data.write_text("{}")
    monkeypatch.setattr(D2, "load", fake_load)
    monkeypatch.setattr(D2, "ROSTER", T.roster_of(eps))
    monkeypatch.setattr(RUN, "roster_files", lambda: [data])
    monkeypatch.setitem(TR.CONFIG, "ckpts", TINY)
    monkeypatch.setitem(TR.CONFIG, "max_steps", TINY[-1])
    ALARMS.clear()
    monkeypatch.setattr(signal, "alarm", lambda sec: ALARMS.append(sec))
    return eps, data


def _prereg(tmp_path, status="**状态：已冻结（测试）**", sha=None, cfg=None):
    p = tmp_path / "PREREG.md"
    p.write_text(f"# 预注册\n\n{status}\ncode_sha={sha or TR.code_sha()[:16]}，config_hash={cfg or TR.config_hash()}\n")
    return p


def _run_all(out, pre, forms=("nonlinear", "condlinear"), folds=("ab6",), seeds=(0,)):
    for f in folds:
        for s in seeds:
            for v in RUN.train_sessions(f):
                RUN.cmd_nom_inner(out, f, v, s, prereg=pre)
            RUN.cmd_nom_final(out, f, s, prereg=pre)
            for form in forms:
                for v in RUN.train_sessions(f):
                    RUN.cmd_inc_inner(out, form, f, v, s, prereg=pre)
                RUN.cmd_inc_final(out, form, f, s, prereg=pre)


# ── 闸门 ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["missing", "draft", "stale_sha", "stale_config"])
def test_formal_commands_refuse_unless_frozen(tmp_path, monkeypatch, case):
    def boom(*a, **k):
        raise AssertionError("闸门必须在读数据之前")
    monkeypatch.setattr(D2, "load", boom)
    pre = {"missing": tmp_path / "none.md",
           "draft": _prereg(tmp_path, status="**状态：草稿（测试）**"),
           "stale_sha": _prereg(tmp_path, sha="0" * 16),
           "stale_config": _prereg(tmp_path, cfg="0" * 16)}[case]
    for call in (lambda: RUN.cmd_prepare(tmp_path, prereg=pre),
                 lambda: RUN.cmd_refs(tmp_path, "ab5", prereg=pre),
                 lambda: RUN.cmd_nom_inner(tmp_path, "ab5", "ab6", 0, prereg=pre),
                 lambda: RUN.cmd_nom_final(tmp_path, "ab5", 0, prereg=pre),
                 lambda: RUN.cmd_inc_inner(tmp_path, "condlinear", "ab5", "ab6", 0, prereg=pre),
                 lambda: RUN.cmd_inc_final(tmp_path, "condlinear", "ab5", 0, prereg=pre)):
        with pytest.raises(RUN.NotFrozenError):
            call()
    assert RUN.main(["prepare", "--out", str(tmp_path), "--prereg", str(pre)]) == 2


def test_formal_commands_take_no_training_overrides(tmp_path):
    for argv in (["nom-inner", "--out", str(tmp_path), "--steps", "5", "ab5", "ab6", "0"],
                 ["inc-final", "--out", str(tmp_path), "--steps", "5", "condlinear", "ab5", "0"]):
        with pytest.raises(SystemExit):
            RUN.main(argv)


def test_code_fingerprint_covers_the_run_entry():
    assert "q2_run.py" in TR.CODE_FILES_Q2


def test_inner_split_mapping():
    assert RUN.train_sessions("ab5") == ("ab6", "ab7")
    assert RUN.inner_train("ab5", "ab6") == "ab7" and RUN.inner_train("ab5", "ab7") == "ab6"
    with pytest.raises(ValueError):
        RUN.inner_train("ab5", "ab5")


# ── prepare：名单、截止时间、输入指纹 ─────────────────────────────────────
def test_prepare_records_inputs_including_calibration_and_h(tmp_path, syn):
    RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    I = RUN.read_inputs(tmp_path)
    assert set(I["calib_files"]) == set(RUN.CALIB_FILES) and set(I["h_files"]) == set(RUN.H_FILES)
    assert {"data", "calib", "h", "deps", "g2"} <= set(I)
    d = json.loads(RUN.deadline_file(tmp_path).read_text())
    assert d["deadline"] - d["started_at"] == 24 * 3600 and 0 < ALARMS[-1] <= 24 * 3600 + 1


def test_a_roster_mismatch_stops_prepare_before_the_clock_starts(tmp_path, syn, monkeypatch):
    eps, _ = syn
    bad = {k: v for k, v in T.roster_of(eps).items()}
    bad["ab6/D"] = bad["ab6/D"][1:]                                    # 名单少一集
    monkeypatch.setattr(D2, "ROSTER", bad)
    with pytest.raises(D1.DataContractError):
        RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    assert not RUN.deadline_file(tmp_path).exists()


def test_jobs_refuse_when_inputs_changed_after_prepare(tmp_path, syn):
    _, data = syn
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    data.write_text('{"changed": true}')
    with pytest.raises(RuntimeError):
        RUN.cmd_nom_inner(tmp_path, "ab6", "ab5", 0, prereg=pre)


def test_passed_deadline_stops_every_formal_command(tmp_path, syn):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    d = json.loads(RUN.deadline_file(tmp_path).read_text())
    d["deadline"] = d["started_at"] - 1
    RUN.deadline_file(tmp_path).write_text(json.dumps(d))
    for call in (lambda: RUN.cmd_nom_inner(tmp_path, "ab6", "ab5", 0, prereg=pre),
                 lambda: RUN.cmd_inc_final(tmp_path, "condlinear", "ab6", 0, prereg=pre),
                 lambda: RUN.cmd_refs(tmp_path, "ab6", prereg=pre)):
        with pytest.raises(RUN.DeadlineExceeded):
            call()
    assert json.loads(RUN.incomplete_file(tmp_path).read_text())["status"] == "未完成"
    assert RUN.main(["nom-inner", "--out", str(tmp_path), "--prereg", str(pre), "ab6", "ab5", "0"]) == 3


# ── 作业链 ────────────────────────────────────────────────────────────────
def test_nominal_inner_and_final(tmp_path, syn):
    import torch
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    with pytest.raises(RuntimeError):
        RUN.cmd_nom_final(tmp_path, "ab6", 0, prereg=pre)            # 内层表不全
    for v in RUN.train_sessions("ab6"):
        RUN.cmd_nom_inner(tmp_path, "ab6", v, 0, prereg=pre)
    j = json.loads(RUN.nom_inner_json(tmp_path, "ab6", "ab5", 0).read_text())
    assert j["train_sessions"] == ["ab7"] and sorted(int(s) for s in j["scores"]) == list(TINY)
    RUN.cmd_nom_final(tmp_path, "ab6", 0, prereg=pre)
    base = RUN.nom_final_base(tmp_path, "ab6", 0)
    f = json.loads(base.with_suffix(".json").read_text())
    V, step = RUN.selection([RUN.nom_inner_json(tmp_path, "ab6", v, 0) for v in ("ab5", "ab7")], TINY)
    assert f["step_star"] == step and f["V"] == {str(k): V[k] for k in sorted(V)}
    assert torch.load(base.with_suffix(".pt"), weights_only=False)["step"] == step
    assert {r["fold"] for r in f["q1"]} == {"ab6"} and {r["arm"] for r in f["q1"]} == {"A"}
    assert {r["arm"] for r in f["zero"]} == {"D"} and len(f["H"]) == len(f["zero"])
    assert {r["arm"] for r in f["aux_zero"]} == {"P2", "P3"} and all(b["den"] > 0 for b in f["aux_H_beta"])
    assert f["pt_sha256"] == RUN.file_sha(base.with_suffix(".pt"))


def test_increment_chain_and_zero_step_keeps_zero_response(tmp_path, syn):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    with pytest.raises(RuntimeError):
        RUN.cmd_inc_inner(tmp_path, "condlinear", "ab6", "ab5", 0, prereg=pre)   # 名义还没终训
    _run_all(tmp_path, pre)
    for form in ("nonlinear", "condlinear"):
        f = json.loads(RUN.inc_final_base(tmp_path, form, "ab6", 0).with_suffix(".json").read_text())
        assert {r["arm"] for r in f["cand"]} == {"D"} and f["train_sessions"] == ["ab5", "ab7"]
    # 把内层表改成第 0 步最好 ⇒ 终训保留零响应：候选与零响应逐集相同
    for v in ("ab5", "ab7"):
        p = RUN.inc_inner_json(tmp_path, "condlinear", "ab6", v, 0)
        j = json.loads(p.read_text())
        j["scores"]["0"]["rho"] = -1.0
        p.write_text(json.dumps(j))
    RUN.cmd_inc_final(tmp_path, "condlinear", "ab6", 0, prereg=pre)
    f = json.loads(RUN.inc_final_base(tmp_path, "condlinear", "ab6", 0).with_suffix(".json").read_text())
    z = json.loads(RUN.nom_final_base(tmp_path, "ab6", 0).with_suffix(".json").read_text())["zero"]
    assert f["step_star"] == 0 and [r["num"] for r in f["cand"]] == [r["num"] for r in z]
    assert all(b["den"] == 0.0 for b in f["aux_beta"])                 # 增量恒为 0 ⇒ β 不定义


def test_jobs_train_and_select_only_on_the_allowed_runs(tmp_path, syn, monkeypatch):
    """真正传给训练与内层评分的集（不只是记录的字段）：名义只用训练会话的裸跑、增量只用训练会话的补正运行；
    内层只在内层训练会话上训练、在内层验证会话上评分；留出会话与 ab4 从不进入训练或选步。"""
    trained, scored = [], []
    real = {"nom": TR.train_nominal, "inc": TR.train_increment, "sn": RUN.score_nominal, "si": RUN.score_increment}

    def nom(eps, seed, ckpts=None, log=None):
        trained.append(("nom", {(e.fold, e.kind) for e in eps}))
        return real["nom"](eps, seed, ckpts=ckpts, log=log)

    def inc(form, n, eps, seed, ckpts=None, log=None):
        trained.append((form, {(e.fold, e.kind) for e in eps}))
        return real["inc"](form, n, eps, seed, ckpts=ckpts, log=log)

    def sn(model, st, eps, K):
        scored.append(("nom", {(e.fold, e.kind) for e in eps}))
        return real["sn"](model, st, eps, K)

    def si(form, model, r, nom_st, eps, e_nom, K):
        scored.append((form, {(e.fold, e.kind) for e in eps}))
        return real["si"](form, model, r, nom_st, eps, e_nom, K)

    monkeypatch.setattr(TR, "train_nominal", nom)
    monkeypatch.setattr(TR, "train_increment", inc)
    monkeypatch.setattr(RUN, "score_nominal", sn)
    monkeypatch.setattr(RUN, "score_increment", si)
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    _run_all(tmp_path, pre, forms=("condlinear",))                     # 外层折 ab6、种子 0
    assert trained == [("nom", {("ab7", "bare")}), ("nom", {("ab5", "bare")}),
                       ("nom", {("ab5", "bare"), ("ab7", "bare")}),
                       ("condlinear", {("ab7", "corr")}), ("condlinear", {("ab5", "corr")}),
                       ("condlinear", {("ab5", "corr"), ("ab7", "corr")})]
    n_ck = len(TINY)
    assert scored == ([("nom", {("ab5", "bare")})] * n_ck + [("nom", {("ab7", "bare")})] * n_ck   # 内层：验证会话
                      + [("nom", {("ab6", "bare")})]                                         # 终训：留出裸跑 Q1 复查
                      + [("condlinear", {("ab5", "corr")})] * n_ck + [("condlinear", {("ab7", "corr")})] * n_ck)


def test_list_jobs_counts_and_requeue_on_input_change(tmp_path, syn, capsys):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    n = {ph: len(RUN.jobs(tmp_path, ph)) for ph in RUN.PHASES}
    assert n == {"refs": 3, "nom-inner": 18, "nom-final": 9, "inc-inner": 36, "inc-final": 18}
    _run_all(tmp_path, pre)
    assert ["ab6", "ab5", "0"] not in RUN.jobs(tmp_path, "nom-inner")
    assert ["condlinear", "ab6", "0"] not in RUN.jobs(tmp_path, "inc-final")
    p = RUN.nom_final_base(tmp_path, "ab6", 0).with_suffix(".pt")       # 名义产物变了 ⇒ 下游增量重排
    p.write_bytes(p.read_bytes() + b"x")
    assert ["condlinear", "ab6", "ab5", "0"] in RUN.jobs(tmp_path, "inc-inner")


def test_final_jobs_requeue_when_weights_or_predictions_are_missing_or_corrupt(tmp_path, syn):
    """终训的 JSON 指纹一致还不够：.pt、.npz 缺失或哈希与记录不符 ⇒ 重新排队。"""
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    _run_all(tmp_path, pre, forms=("condlinear",))
    assert ["ab6", "0"] not in RUN.jobs(tmp_path, "nom-final")
    assert ["condlinear", "ab6", "0"] not in RUN.jobs(tmp_path, "inc-final")
    nom, inc = RUN.nom_final_base(tmp_path, "ab6", 0), RUN.inc_final_base(tmp_path, "condlinear", "ab6", 0)
    keep = {p: p.read_bytes() for p in (nom.with_suffix(".npz"), inc.with_suffix(".pt"), inc.with_suffix(".npz"))}
    nom.with_suffix(".npz").write_bytes(b"corrupt")                    # 名义的预测文件损坏
    assert ["ab6", "0"] in RUN.jobs(tmp_path, "nom-final")
    nom.with_suffix(".npz").write_bytes(keep[nom.with_suffix(".npz")])
    assert ["ab6", "0"] not in RUN.jobs(tmp_path, "nom-final")
    inc.with_suffix(".pt").unlink()                                     # 增量权重缺失
    assert ["condlinear", "ab6", "0"] in RUN.jobs(tmp_path, "inc-final")
    inc.with_suffix(".pt").write_bytes(keep[inc.with_suffix(".pt")])
    inc.with_suffix(".npz").write_bytes(keep[inc.with_suffix(".npz")] + b"x")    # 增量预测文件被改
    assert ["condlinear", "ab6", "0"] in RUN.jobs(tmp_path, "inc-final")


def test_changed_code_cannot_continue_an_old_run(tmp_path, syn, monkeypatch):
    """冻结代码改动后须重新冻结，且不能在旧目录里续跑：截止时间记录绑定旧指纹 ⇒ prepare 与作业都拒绝。"""
    RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    monkeypatch.setattr(TR, "code_sha", lambda: "f" * 64)              # 模拟改了冻结代码
    pre2 = _prereg(tmp_path)                                            # 已按新指纹重新冻结
    for call in (lambda: RUN.cmd_prepare(tmp_path, prereg=pre2),
                 lambda: RUN.cmd_nom_inner(tmp_path, "ab6", "ab5", 0, prereg=pre2)):
        with pytest.raises(RuntimeError, match="另一版冻结的实验"):
            call()


@pytest.mark.parametrize("steps", [0, -1, 31, 10000])
def test_timing_refuses_out_of_range_steps_before_loading_or_training(steps, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("越界必须在加载数据、训练之前拒绝")
    monkeypatch.setattr(D2, "load", boom)
    monkeypatch.setattr(TR, "train_nominal", boom)
    with pytest.raises(RUN.TimingRefused):
        RUN.cmd_timing(steps)
    assert RUN.main(["timing", "--steps", str(steps)]) == 2


def test_timing_needs_no_prereg(syn, monkeypatch, capsys):
    monkeypatch.setattr(RUN, "PREREG", D1.ROOT / "reports" / "does_not_exist.md")
    RUN.cmd_timing(1)
    out = capsys.readouterr().out
    assert "s/步" in out and "核·小时" in out and "condlinear" in out


@pytest.mark.slow
def test_refs_on_synthetic(tmp_path, syn):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    RUN.cmd_refs(tmp_path, "ab7", prereg=pre)
    j = json.loads(RUN.ref_json(tmp_path, "ab7").read_text())
    assert {r["fold"] for r in j["episodes"]} == {"ab7"} and j["config_hash"] == TR.config_hash()
    mtime = RUN.ref_json(tmp_path, "ab7").stat().st_mtime_ns
    RUN.cmd_refs(tmp_path, "ab7", prereg=pre)                          # 已是最新 ⇒ 跳过
    assert RUN.ref_json(tmp_path, "ab7").stat().st_mtime_ns == mtime
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_q2_run.py -q`
Expected: `ModuleNotFoundError: No module named 'q2_run'`

- [ ] **Step 3: 实现** —— `sim/q2/q2_run.py`

```python
"""阶段二命令行入口 —— `reports/PREREG_Q2_STAGE2.md` §5、§11。

⚠ `prepare / refs / nom-inner / nom-final / inc-inner / inc-final` 是**正式运行**：入口先查预注册 ——
状态行须以「**状态：已冻结」开头，`code_sha=` 与 `config_hash=` 须与当前一致，否则拒绝；不接受训练参数覆盖。
`timing` 只用于另行授权的短程耗时测量（1–30 步，越界在读数据之前拒绝）：不评分外层留出会话、不选步、
不保存正式训练产物（它不是运行的旁路）。

作业（外层折 = 留出的会话；内层验证会话 ∈ 该折另外两个会话）：
- `nom-inner FOLD VAL SEED`：名义 2a 用「另一个训练会话」的裸跑训到上限，在 VAL 的裸跑上给每个检查点评分；
- `nom-final FOLD SEED`：由两张内层表选步 → 用两个训练会话的裸跑训到选中步数并冻结；评分：留出裸跑（Q1 复查）、
  留出补正运行上的零响应与固定 H、ab4 P2/P3 上的零响应与固定 H（含 H 的 β 分子分母）；
- `inc-inner FORM FOLD VAL SEED`：「固定名义模型下的增量选步验证」——增量用「另一个训练会话」的补正运行训到上限，
  在 VAL 的补正运行上给每个检查点评分；名义 = 该折已冻结的 2a；
- `inc-final FORM FOLD SEED`：由两张内层表选步 → 用两个训练会话的补正运行训到选中步数（第 0 步 = 保留零响应，不训练）；
  评分：留出补正运行上的候选、ab4 P2/P3 上的候选（含 β 分子分母）；
- `refs FOLD`：G-2 同口径窗口重预测在留出会话裸跑上的分数（Q1 复查的背景）。
续跑只跳过配置指纹与输入指纹都一致的作业；终训作业还须 `.pt`、`.npz` 存在且哈希与记录一致，损坏就重新排队。
输入指纹含数据文件、标定 R/b 文件、固定 H 文件、依赖代码与库版本；截止时间与数据名单的落实同阶段一。
冻结代码一改，`code_sha` / `config_hash` 就变：须重新冻结，且旧目录的截止时间记录绑定旧指纹 ⇒ 不能在旧目录里续跑。

    python -u sim/q2/q2_run.py timing --steps 30
    python -u sim/q2/q2_run.py prepare   --out outputs/q2_stage2
    python -u sim/q2/q2_run.py list-jobs --out outputs/q2_stage2 --phase nom-inner
    python -u sim/q2/q2_run.py nom-inner --out outputs/q2_stage2 ab5 ab6 0
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import q1_data as D1           # noqa: E402
import q1_run as R1            # noqa: E402
import q1_train as TR1         # noqa: E402
import q1_windows as W         # noqa: E402
import q2_data as D2           # noqa: E402
import q2_models as Mo         # noqa: E402
import q2_train as TR          # noqa: E402
import q2_windows as W2        # noqa: E402

ROOT = D1.ROOT
FOLDS = D2.FOLDS
SEEDS = TR.CONFIG["seeds"]
FORMS = Mo.FORMS
PREREG = ROOT / "reports/PREREG_Q2_STAGE2.md"
CALIB_FILES = ("outputs/step1_H/step1_H_v1.json", "outputs/p2_labels/pp2a-deploy/manifest.json")
H_FILES = ("outputs/step1_H/step1_H_v1.npz",)
TIMING_FOLD, TIMING_VAL = "ab5", "ab7"
TIMING_MAX_STEPS = 30
RUNNER_JOBS = 15
atomic_json, atomic_npz, atomic_torch = R1.atomic_json, R1.atomic_npz, R1.atomic_torch
file_sha, files_fp, fp_of = R1.file_sha, R1.files_fp, R1.fp_of


class NotFrozenError(RuntimeError):
    """预注册不存在、未冻结，或冻结时的代码/配置指纹与当前不符。"""


class DeadlineExceeded(RuntimeError):
    """本轮已超出预注册的墙钟上限：停止，并标记「未完成」。"""


class TimingRefused(ValueError):
    """测速步数越界：`timing` 只是另行授权的短程测速（1–30 步），不是运行的旁路。"""


def require_frozen(prereg=None) -> None:
    p = Path(prereg) if prereg is not None else PREREG
    if not p.exists():
        raise NotFrozenError(f"拒绝运行：{p} 不存在 —— 正式运行须先冻结预注册")
    text = p.read_text()
    if not re.search(r"^\*\*状态：已冻结", text, re.M):
        raise NotFrozenError(f"拒绝运行：{p.name} 尚未冻结（状态行须以「**状态：已冻结」开头）")
    m = re.search(r"code_sha=([0-9a-f]{16})", text)
    if not m or m.group(1) != TR.code_sha()[:16]:
        raise NotFrozenError("拒绝运行：预注册记录的 code_sha 与当前代码不符 —— 代码改动后须重新冻结")
    m = re.search(r"config_hash=([0-9a-f]{16})", text)
    if not m or m.group(1) != TR.config_hash():
        raise NotFrozenError("拒绝运行：预注册记录的 config_hash 与当前规则不符 —— 须重新冻结")


# ── 折与路径 ──────────────────────────────────────────────────────────────
def train_sessions(fold: str) -> tuple:
    if fold not in FOLDS:
        raise ValueError(f"未知的折 {fold}")
    return tuple(s for s in FOLDS if s != fold)


def inner_train(fold: str, val: str) -> str:
    """该折的内层：VAL 是内层验证会话，另一个训练会话是内层训练会话。"""
    tr = train_sessions(fold)
    if val not in tr:
        raise ValueError(f"{val} 不是折 {fold} 的训练会话")
    return next(s for s in tr if s != val)


def nom_inner_json(out, fold, val, seed) -> Path:
    return Path(out) / "nom" / "inner" / f"seed{seed}" / f"{fold}__val_{val}.json"


def nom_final_base(out, fold, seed) -> Path:
    return Path(out) / "nom" / "final" / f"seed{seed}" / fold


def inc_inner_json(out, form, fold, val, seed) -> Path:
    return Path(out) / "inc" / form / "inner" / f"seed{seed}" / f"{fold}__val_{val}.json"


def inc_final_base(out, form, fold, seed) -> Path:
    return Path(out) / "inc" / form / "final" / f"seed{seed}" / fold


def ref_json(out, fold) -> Path:
    return Path(out) / "refs" / "g2" / f"{fold}.json"


# ── 输入指纹、名单与截止时间 ───────────────────────────────────────────────
def roster_files() -> list:
    """冻结名单里全部 183 集的数据文件。"""
    return [ROOT / "outputs" / n.split("/")[0] / f"{n.split('/')[1]}.json" for v in D2.ROSTER.values() for n in v]


def inputs_file(out) -> Path:
    return Path(out) / "inputs.json"


def read_inputs(out) -> dict:
    p = inputs_file(out)
    if not p.exists():
        raise RuntimeError(f"缺 {p} —— 先跑 prepare")
    return json.loads(p.read_text())


def current_inputs() -> dict:
    dfiles, dfp = files_fp(roster_files())
    cfiles, cfp = files_fp([ROOT / f for f in CALIB_FILES])
    hfiles, hfp = files_fp([ROOT / f for f in H_FILES])
    pfiles, _ = files_fp([ROOT / f for f in R1.DEP_FILES])
    gfiles, gfp = files_fp([ROOT / f for f in R1.G2_FILES])
    env = R1.env_versions()
    return {"code_sha": TR.code_sha(), "data": dfp, "calib": cfp, "h": hfp,
            "deps": fp_of({"files": pfiles, "env": env}), "g2": gfp,
            "data_files": dfiles, "calib_files": cfiles, "h_files": hfiles, "deps_files": pfiles,
            "g2_files": gfiles, "env": env, "roster": D2.ROSTER}


def check_inputs(I: dict) -> None:
    now = current_inputs()
    bad = [k for k in ("code_sha", "data", "calib", "h", "deps", "g2") if now[k] != I.get(k)]
    if bad:
        raise RuntimeError(f"输入自 prepare 以来已变：{bad} —— 重跑 prepare（或换 --out）")


def deadline_file(out) -> Path:
    return Path(out) / "deadline.json"


def incomplete_file(out) -> Path:
    return Path(out) / "INCOMPLETE.json"


def mark_incomplete(out, reason: str) -> None:
    if not incomplete_file(out).exists():
        atomic_json(incomplete_file(out), {"status": "未完成", "reason": reason, "at": time.time()})


def enforce_deadline(out, create: bool = False) -> dict:
    """第一次 prepare 记录截止时间（续跑不重置）；超时 ⇒ 标记未完成并拒绝；否则设闹钟，到点中止本作业。"""
    p = deadline_file(out)
    fp = {"code_sha": TR.code_sha(), "config_hash": TR.config_hash()}
    if p.exists():
        d = json.loads(p.read_text())
        if {k: d.get(k) for k in fp} != fp:
            raise RuntimeError(f"{p}: 本目录属于另一版冻结的实验 —— 换一个 --out")
    elif create:
        now = time.time()
        d = {**fp, "budget_hours": TR.CONFIG["budget_hours"], "started_at": now,
             "deadline": now + TR.CONFIG["budget_hours"] * 3600}
        atomic_json(p, d)
    else:
        raise RuntimeError(f"缺 {p} —— 先跑 prepare")
    left = d["deadline"] - time.time()
    if left <= 0:
        mark_incomplete(out, f"超出预注册的墙钟上限（{TR.CONFIG['budget_hours']} 小时）")
        raise DeadlineExceeded("已超出本轮截止时间 —— 拒绝运行，本轮标记为未完成")

    def on_alarm(signum, frame):
        mark_incomplete(out, f"作业运行中到达墙钟上限（{TR.CONFIG['budget_hours']} 小时）")
        raise DeadlineExceeded("到达本轮截止时间 —— 作业中止，本轮标记为未完成")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(int(left) + 1)
    return d


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_done(path: Path, inputs) -> bool:
    j = _load(path) if inputs is not None else None
    return bool(j) and j.get("config_hash") == TR.config_hash() and j.get("input_fp") == fp_of(inputs)


def final_done(base: Path, inputs) -> bool:
    """终训作业是否完成：JSON 的配置与输入指纹一致，且 `.pt`、`.npz` 存在、哈希与 JSON 记录一致（损坏就重新排队）。"""
    if not is_done(base.with_suffix(".json"), inputs):
        return False
    j = _load(base.with_suffix(".json"))
    return all(base.with_suffix(ext).exists() and file_sha(base.with_suffix(ext)) == j.get(key)
               for ext, key in ((".pt", "pt_sha256"), (".npz", "npz_sha256")))


def base_inputs(I: dict) -> dict:
    return {"data": I["data"], "calib": I["calib"], "deps": I["deps"]}


def nom_inner_inputs(I, fold, val) -> dict:
    return base_inputs(I)


def nom_final_inputs(out, I, fold, seed):
    inner = {}
    for v in train_sessions(fold):
        p = nom_inner_json(out, fold, v, seed)
        if not p.exists():
            return None
        inner[v] = file_sha(p)
    return {**base_inputs(I), "h": I["h"], "inner": inner}


def nominal_pt_sha(out, fold, seed):
    p = nom_final_base(out, fold, seed).with_suffix(".pt")
    j = _load(nom_final_base(out, fold, seed).with_suffix(".json"))
    if not p.exists() or not j or j.get("pt_sha256") != file_sha(p):
        return None
    return j["pt_sha256"]


def inc_inner_inputs(out, I, form, fold, val, seed):
    nom = nominal_pt_sha(out, fold, seed)
    return None if nom is None else {**base_inputs(I), "nominal": nom}


def inc_final_inputs(out, I, form, fold, seed):
    nom = nominal_pt_sha(out, fold, seed)
    if nom is None:
        return None
    inner = {}
    for v in train_sessions(fold):
        p = inc_inner_json(out, form, fold, v, seed)
        if not p.exists():
            return None
        inner[v] = file_sha(p)
    return {**base_inputs(I), "nominal": nom, "inner": inner}


def ref_inputs(I) -> dict:
    return {**base_inputs(I), "g2": I["g2"]}


# ── 载入与评分小工具 ──────────────────────────────────────────────────────
def load_eps(folds, kinds) -> list:
    """按冻结名单核对后载入；`folds` 可以为空（只要 ab4 辅助时）。"""
    return D2.load(folds=tuple(folds), kinds=tuple(kinds), check=True)


def load_nominal(out, fold, seed):
    import torch
    base = nom_final_base(out, fold, seed)
    j = _load(base.with_suffix(".json"))
    if not j or j.get("pt_sha256") != file_sha(base.with_suffix(".pt")):
        raise RuntimeError(f"{base}: 名义模型缺失或与记录不符 —— 先跑 nom-final")
    ck = torch.load(base.with_suffix(".pt"), weights_only=False)
    model = TR1.with_weights("2a", ck["stats"], ck["state_dict"])
    model.eval()
    return model, ck["stats"]


def selection(paths, ckpts) -> tuple:
    """两张内层表 → (V {step: 平均 ρ}, step*)。表缺失或检查点不符 ⇒ RuntimeError。"""
    rho = []
    for p in paths:
        j = _load(p)
        if j is None:
            raise RuntimeError(f"缺内层表 {p}")
        if sorted(int(s) for s in j["scores"]) != list(ckpts):
            raise RuntimeError(f"{p}: 检查点与选步规则不符")
        rho.append({int(s): float(r["rho"]) for s, r in j["scores"].items()})
    V = {s: float(np.mean([r[s] for r in rho])) for s in ckpts}
    return V, TR.pick_step(V)


def stamp(rec: dict, inputs: dict) -> dict:
    return {**rec, "config_hash": TR.config_hash(), "code_sha": TR.code_sha(), "inputs": inputs,
            "input_fp": fp_of(inputs), "finished_at": time.time()}


def _guard(out, prereg, create=False) -> dict:
    require_frozen(prereg)
    if create:
        return {}
    enforce_deadline(out)
    I = read_inputs(out)
    check_inputs(I)
    return I


# ── prepare / refs ────────────────────────────────────────────────────────
def cmd_prepare(out, prereg=None) -> None:
    require_frozen(prereg)
    eps = D2.load()                                                   # 全部 183 集，核对名单与日志契约
    enforce_deadline(out, create=True)                                # 数据通过核验后才开始计时；续跑不重置
    I = current_inputs()
    atomic_json(inputs_file(out), I)
    print(f"  载入 {len(eps)} 集 · 数据 {I['data']} · 标定 {I['calib']} · H {I['h']} · 依赖 {I['deps']}", flush=True)


def cmd_refs(out, fold: str, prereg=None) -> None:
    I = _guard(out, prereg)
    inputs = ref_inputs(I)
    if is_done(ref_json(out, fold), inputs):
        print(f"  参照 {fold} 已是最新，跳过", flush=True)
        return
    import q1_g2ref as G
    eps = load_eps((fold,), ("bare",))
    R, b = D1.calibration()
    K = W.kernel()
    g2 = G.load_g2()
    rows = [TR.ep_metrics(e, G.episode_ehat(g2, e, W.chunk_windows(e), R, b), K) for e in eps]
    atomic_json(ref_json(out, fold), stamp({"kind": "g2", "fold": fold, "episodes": rows}, inputs))
    print(f"  参照 {fold} 完成", flush=True)


# ── 名义 ──────────────────────────────────────────────────────────────────
def score_nominal(model, st, eps, K) -> list:
    return [TR.ep_metrics(e, TR.nominal_episode(model, st, e), K) for e in eps]


def cmd_nom_inner(out, fold, val, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    tr_s = inner_train(fold, val)
    eps = load_eps((tr_s, val), ("bare",))
    tr, va = D2.select(eps, tr_s), D2.select(eps, val)
    K = W.kernel()
    res = TR.train_nominal(tr, seed, log=lambda m: print(m, flush=True))
    scores = {}
    for step in sorted(res["ckpt"]):
        m = TR1.with_weights("2a", res["stats"], res["ckpt"][step])
        rows = score_nominal(m, res["stats"], va, K)
        scores[str(step)] = {"rho": TR.pooled_rho(rows), "state_sha256": TR1.state_dict_sha(res["ckpt"][step])}
    atomic_json(nom_inner_json(out, fold, val, seed), stamp(
        {"kind": "nom-inner", "fold": fold, "val": val, "seed": seed, "train_sessions": [tr_s],
         "scores": scores, "sec": res["sec"]}, nom_inner_inputs(I, fold, val)))
    print(f"  名义内层 {fold} 验证 {val} seed{seed} 完成", flush=True)


def cmd_nom_final(out, fold, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    H = W2.load_h()
    inputs = nom_final_inputs(out, I, fold, seed)
    if inputs is None:
        raise RuntimeError(f"折 {fold} 种子 {seed} 的名义内层表不全 —— 先跑 nom-inner")
    V, step = selection([nom_inner_json(out, fold, v, seed) for v in train_sessions(fold)], TR.CONFIG["ckpts"])
    eps = load_eps(FOLDS, ("bare", "corr", "aux"))
    tr = [e for e in eps if e.kind == "bare" and e.fold in train_sessions(fold)]
    K = W.kernel()
    res = TR.train_nominal(tr, seed, ckpts=(step,), log=lambda m: print(m, flush=True))
    sd = res["ckpt"][step]
    model = TR1.with_weights("2a", res["stats"], sd)
    base = nom_final_base(out, fold, seed)
    atomic_torch(base.with_suffix(".pt"), {"state_dict": sd, "stats": res["stats"], "step": step})
    rec = {"kind": "nom-final", "fold": fold, "seed": seed, "train_sessions": list(train_sessions(fold)),
           "V": {str(s): V[s] for s in sorted(V)}, "step_star": step, "boundary": step == TR.CONFIG["max_steps"],
           "state_sha256": TR1.state_dict_sha(sd), "pt_sha256": file_sha(base.with_suffix(".pt")), "sec": res["sec"]}
    held_bare = D2.select(eps, fold, "bare")
    held_corr = D2.select(eps, fold, "corr")
    aux = D2.select(eps, "ab4", "aux")
    rec["q1"] = score_nominal(model, res["stats"], held_bare, K)
    preds, zero, hres, aux_zero, aux_h, aux_beta = {}, [], [], [], [], []
    for e in held_corr:
        en, h = TR.nominal_episode(model, res["stats"], e), TR.h_episode(H, e)
        zero.append(TR.ep_metrics(e, en, K))
        hres.append(TR.ep_metrics(e, en + h, K))
        preds[f"nom__{e.name.replace('/', '__')}"] = en
        preds[f"H__{e.name.replace('/', '__')}"] = en + h
    for e in aux:
        en, h = TR.nominal_episode(model, res["stats"], e), TR.h_episode(H, e)
        aux_zero.append(TR.ep_metrics(e, en, K))
        aux_h.append(TR.ep_metrics(e, en + h, K))
        num, den = TR.beta_parts(e, en, h, K)
        aux_beta.append({"name": e.name, "arm": e.arm, "num": num, "den": den})
    atomic_npz(base.with_suffix(".npz"), **preds)
    rec.update({"zero": zero, "H": hres, "aux_zero": aux_zero, "aux_H": aux_h, "aux_H_beta": aux_beta,
                "npz_sha256": file_sha(base.with_suffix(".npz"))})
    atomic_json(base.with_suffix(".json"), stamp(rec, inputs))
    note = "（最佳点落在搜索边界）" if rec["boundary"] else ""
    print(f"  名义终训 {fold} seed{seed}：step* = {step}{note}", flush=True)


# ── 增量 ──────────────────────────────────────────────────────────────────
def score_increment(form, model, r, nom_st, eps, e_nom, K) -> list:
    return [TR.ep_metrics(e, e_nom[e.name] + TR.increment_episode(form, model, r["st_inc"], r["scale"], nom_st, e), K)
            for e in eps]


def cmd_inc_inner(out, form, fold, val, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    inputs = inc_inner_inputs(out, I, form, fold, val, seed)
    if inputs is None:
        raise RuntimeError(f"折 {fold} 种子 {seed} 的名义模型缺失 —— 先跑 nom-final")
    nom_model, nom_st = load_nominal(out, fold, seed)
    tr_s = inner_train(fold, val)
    eps = load_eps((tr_s, val), ("corr",))
    tr, va = D2.select(eps, tr_s), D2.select(eps, val)
    K = W.kernel()
    res = TR.train_increment(form, (nom_model, nom_st), tr, seed, log=lambda m: print(m, flush=True))
    e_nom = {e.name: TR.nominal_episode(nom_model, nom_st, e) for e in va}
    scores = {}
    for step in sorted(res["ckpt"]):
        m = TR.with_increment(form, res["ckpt"][step])
        rows = score_increment(form, m, res, nom_st, va, e_nom, K)
        scores[str(step)] = {"rho": TR.pooled_rho(rows), "state_sha256": TR1.state_dict_sha(res["ckpt"][step])}
    atomic_json(inc_inner_json(out, form, fold, val, seed), stamp(
        {"kind": "inc-inner", "form": form, "fold": fold, "val": val, "seed": seed, "train_sessions": [tr_s],
         "scores": scores, "sec": res["sec"]}, inputs))
    print(f"  增量内层 {form} {fold} 验证 {val} seed{seed} 完成", flush=True)


def cmd_inc_final(out, form, fold, seed, prereg=None) -> None:
    I = _guard(out, prereg)
    inputs = inc_final_inputs(out, I, form, fold, seed)
    if inputs is None:
        raise RuntimeError(f"{form} 折 {fold} 种子 {seed} 的增量内层表或名义模型不全")
    V, step = selection([inc_inner_json(out, form, fold, v, seed) for v in train_sessions(fold)], TR.CONFIG["ckpts"])
    nom_model, nom_st = load_nominal(out, fold, seed)
    eps = load_eps(FOLDS, ("corr", "aux"))
    tr = [e for e in eps if e.kind == "corr" and e.fold in train_sessions(fold)]
    held = D2.select(eps, fold, "corr")
    aux = D2.select(eps, "ab4", "aux")
    K = W.kernel()
    res = TR.train_increment(form, (nom_model, nom_st), tr, seed, ckpts=(step,), log=lambda m: print(m, flush=True))
    sd = res["ckpt"][step]                                            # step* = 0 ⇒ 初始权重 = 零响应，未训练
    model = TR.with_increment(form, sd)
    base = inc_final_base(out, form, fold, seed)
    atomic_torch(base.with_suffix(".pt"), {"form": form, "state_dict": sd, "st_inc": res["st_inc"],
                                           "scale": res["scale"], "step": step})
    e_nom = {e.name: TR.nominal_episode(nom_model, nom_st, e) for e in held + aux}
    preds, cand, aux_c, aux_beta = {}, [], [], []
    for e in held:
        d = TR.increment_episode(form, model, res["st_inc"], res["scale"], nom_st, e)
        cand.append(TR.ep_metrics(e, e_nom[e.name] + d, K))
        preds[f"cand__{e.name.replace('/', '__')}"] = e_nom[e.name] + d
    for e in aux:
        d = TR.increment_episode(form, model, res["st_inc"], res["scale"], nom_st, e)
        aux_c.append(TR.ep_metrics(e, e_nom[e.name] + d, K))
        num, den = TR.beta_parts(e, e_nom[e.name], d, K)
        aux_beta.append({"name": e.name, "arm": e.arm, "num": num, "den": den})
    atomic_npz(base.with_suffix(".npz"), **preds)
    rec = {"kind": "inc-final", "form": form, "fold": fold, "seed": seed, "train_sessions": list(train_sessions(fold)),
           "V": {str(s): V[s] for s in sorted(V)}, "step_star": step, "boundary": step == TR.CONFIG["max_steps"],
           "state_sha256": TR1.state_dict_sha(sd), "pt_sha256": file_sha(base.with_suffix(".pt")),
           "npz_sha256": file_sha(base.with_suffix(".npz")), "cand": cand, "aux_cand": aux_c,
           "aux_beta": aux_beta, "sec": res["sec"]}
    atomic_json(base.with_suffix(".json"), stamp(rec, inputs))
    note = "（选中第 0 步：保留零响应）" if step == 0 else ("（最佳点落在搜索边界）" if rec["boundary"] else "")
    print(f"  增量终训 {form} {fold} seed{seed}：step* = {step}{note}", flush=True)


# ── 待办清单 ──────────────────────────────────────────────────────────────
PHASES = ("refs", "nom-inner", "nom-final", "inc-inner", "inc-final")


def jobs(out, phase: str) -> list:
    """→ 该阶段尚未完成的作业参数（字符串列表的列表）。"""
    I = read_inputs(out)
    todo = []
    for fold in FOLDS:
        if phase == "refs":
            if not is_done(ref_json(out, fold), ref_inputs(I)):
                todo.append([fold])
            continue
        for seed in SEEDS:
            if phase == "nom-inner":
                todo += [[fold, v, str(seed)] for v in train_sessions(fold)
                         if not is_done(nom_inner_json(out, fold, v, seed), nom_inner_inputs(I, fold, v))]
            elif phase == "nom-final":
                if not final_done(nom_final_base(out, fold, seed), nom_final_inputs(out, I, fold, seed)):
                    todo.append([fold, str(seed)])
            else:
                for form in FORMS:
                    if phase == "inc-inner":
                        todo += [[form, fold, v, str(seed)] for v in train_sessions(fold)
                                 if not is_done(inc_inner_json(out, form, fold, v, seed),
                                                inc_inner_inputs(out, I, form, fold, v, seed))]
                    elif not final_done(inc_final_base(out, form, fold, seed), inc_final_inputs(out, I, form, fold, seed)):
                        todo.append([form, fold, str(seed)])
    return todo


def cmd_list_jobs(out, phase: str) -> None:
    for args in jobs(out, phase):
        print(" ".join(args))


# ── 耗时测量（另行授权的短程测速；不评分外层留出会话、不选步、不保存产物）─────────
def cmd_timing(steps: int) -> None:
    if not 1 <= steps <= TIMING_MAX_STEPS:                            # 在读数据、训练之前拒绝
        raise TimingRefused(f"拒绝测速：--steps 须在 1–{TIMING_MAX_STEPS} 之间（得到 {steps}）—— "
                            "timing 只是另行授权的短程测速，不是运行的旁路")
    tr_s = inner_train(TIMING_FOLD, TIMING_VAL)
    eps = load_eps((tr_s,), ("bare", "corr"))
    bare, corr = D2.select(eps, tr_s, "bare"), D2.select(eps, tr_s, "corr")
    K = W.kernel()
    res = TR.train_nominal(bare, 0, ckpts=(steps,))
    per = {"nominal": res["sec"] / steps}
    m = TR1.with_weights("2a", res["stats"], res["ckpt"][steps])
    t = time.time()
    score_nominal(m, res["stats"], bare[:5], K)                       # 只在训练会话上计时，不打印 ρ
    per_ep = {"nominal": (time.time() - t) / 5}
    e_nom = {e.name: TR.nominal_episode(m, res["stats"], e) for e in corr[:5]}
    for form in FORMS:
        r = TR.train_increment(form, (m, res["stats"]), corr, 0, ckpts=(steps,))
        per[form] = r["sec"] / steps
        mi = TR.with_increment(form, r["ckpt"][steps])
        t = time.time()
        score_increment(form, mi, r, res["stats"], corr[:5], e_nom, K)
        per_ep[form] = (time.time() - t) / 5
    for k in per:
        print(f"  {k}: {per[k]:.3f} s/步 · 评分 {per_ep[k]:.3f} s/集", flush=True)
    S, n_ck, load = TR.CONFIG["max_steps"], len(TR.CONFIG["ckpts"]), 90.0
    n_val = {"ab5": (40, 43), "ab6": (30, 30), "ab7": (15, 15)}                    # 各会话的（裸跑, 补正）集数
    nom = sum(len(SEEDS) * (S * per["nominal"] + n_ck * n_val[v][0] * per_ep["nominal"] + load)
              for f in FOLDS for v in train_sessions(f))
    nomf = len(FOLDS) * len(SEEDS) * (S * per["nominal"] + 120 * per_ep["nominal"] + load)
    inc = sum(len(SEEDS) * (S * per[fm] + n_ck * n_val[v][1] * per_ep[fm] + load)
              for fm in FORMS for f in FOLDS for v in train_sessions(f))
    incf = sum(len(FOLDS) * len(SEEDS) * (S * per[fm] + 60 * per_ep[fm] + load) for fm in FORMS)
    total = nom + nomf + inc + incf
    print(f"  合计 ≤ {total / 3600:.1f} 核·小时（按步数取上限计；含评分与载入）；{RUNNER_JOBS} 路并行、四个阶段依次进行，"
          f"墙钟约 {sum(x / RUNNER_JOBS for x in (nom, nomf, inc, incf)) / 3600 + 0.5:.1f} 小时（含 prepare 与参照约 0.5 小时）")


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--prereg", default=None)
    p = sub.add_parser("refs")
    p.add_argument("--out", required=True)
    p.add_argument("--prereg", default=None)
    p.add_argument("fold", choices=FOLDS)
    p = sub.add_parser("list-jobs")
    p.add_argument("--out", required=True)
    p.add_argument("--phase", required=True, choices=PHASES)
    for name, args in (("nom-inner", ("fold", "val", "seed")), ("nom-final", ("fold", "seed")),
                       ("inc-inner", ("form", "fold", "val", "seed")), ("inc-final", ("form", "fold", "seed"))):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--prereg", default=None)
        for a in args:
            if a == "seed":
                p.add_argument(a, type=int, choices=SEEDS)
            elif a == "form":
                p.add_argument(a, choices=FORMS)
            else:
                p.add_argument(a, choices=FOLDS)
    p = sub.add_parser("timing")
    p.add_argument("--steps", type=int, default=30)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            cmd_prepare(a.out, a.prereg)
        elif a.cmd == "refs":
            cmd_refs(a.out, a.fold, a.prereg)
        elif a.cmd == "list-jobs":
            cmd_list_jobs(a.out, a.phase)
        elif a.cmd == "nom-inner":
            cmd_nom_inner(a.out, a.fold, a.val, a.seed, a.prereg)
        elif a.cmd == "nom-final":
            cmd_nom_final(a.out, a.fold, a.seed, a.prereg)
        elif a.cmd == "inc-inner":
            cmd_inc_inner(a.out, a.form, a.fold, a.val, a.seed, a.prereg)
        elif a.cmd == "inc-final":
            cmd_inc_final(a.out, a.form, a.fold, a.seed, a.prereg)
        else:
            cmd_timing(a.steps)
    except (NotFrozenError, TimingRefused) as err:
        print(err, flush=True)
        return 2
    except DeadlineExceeded as err:
        print(err, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
```

- [ ] **Step 4: 把运行入口计入代码指纹** —— `sim/q2/q2_train.py:34`

把

```python
CODE_FILES_Q2 = ("q2_data.py", "q2_roster.json", "q2_windows.py", "q2_models.py", "q2_train.py")
```

改为

```python
CODE_FILES_Q2 = ("q2_data.py", "q2_roster.json", "q2_windows.py", "q2_models.py", "q2_train.py",
                 "q2_run.py")
```

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest sim/q2/test_q2_run.py sim/q2/test_q2_train.py -q -m "not slow"` → Expected: `34 passed, 1 deselected`
Run: `$PY -m pytest sim/q2/test_q2_run.py -q -m slow` → Expected: `1 passed`，约 10 s（G-2 在合成集上的同口径参照，以及「已是最新则跳过」）

- [ ] **Step 6: 提交**

```bash
git add sim/q2/q2_run.py sim/q2/test_q2_run.py sim/q2/q2_train.py
git commit -m "阶段二 Q2：命令行与产物（五个阶段、冻结闸门、截止时间、名单与输入指纹含 R/b 与 H；timing 限 1–30 步且不评分留出会话；终训续跑核对 .pt/.npz 哈希）"
```

---

### Task 6: 汇总与判据

**Files:**
- Create: `sim/q2/q2_report.py`
- Modify: `sim/q2/q2_train.py:34-35`（`CODE_FILES_Q2` 加入 `q2_report.py`，判据代码进冻结指纹）
- Test: `sim/q2/test_q2_report.py`

**Interfaces:**
- Consumes：Task 5 的路径、作业输入函数、`selection`、`_load`、产物字段；`q2_data.ROSTER / BARE / CORR / AUX`
- Produces：
  - 核验：
    - `artifacts(out) -> {键: 路径}`：应有的 84 份产物 = 3 参照 + 18 名义内层 + 9 名义终训 + 36 增量内层 + 18 增量终训
    - `expected(fold, kind) -> [集名]`、`verify(out) -> [问题]`、`run_status(out) -> (ok, reason)`
  - 判据（纯函数）：
    - `seed_mean_rho({fold: {seed: rows}}) -> {fold: ρ̄}`、`per_seed_rho(...)`
    - `gate(rb) -> {form: {d, d_mean, per_fold_vs_H, per_fold_vs_zero, pass}}`
    - `nonlin_vs_cond(rb) -> {rel, rel_mean, better_all}`、`choose(g, rb) -> "nonlinear" | "condlinear" | None`
  - 只报告：`beta(rows) -> float | None`、`beta_seed_mean({seed: rows}) -> {per_seed, mean, n_defined}`、`lead_rho({seed: rows})`、`aux_table(aux, braw)`
  - 汇总：`collect(out)`、`summarize(out) -> dict`、`markdown(s) -> str`、`main(argv)`
    - 退出码：0 = 完成；3 = 本轮未完成（不作判决）；4 = 核验未通过（拒绝汇总）
    - 三种情况都写 `summary.json`；完成时另写 `summary.md`

- [ ] **Step 1: 写失败的测试** —— `sim/q2/test_q2_report.py`

```python
import json

import numpy as np
import pytest

import q2_data as D2
import q2_report as REP
import q2_run as RUN
import q2_testing as T
import q2_train as TR

F = D2.FOLDS


def _rb(zero, H, nl, cl):
    return {"zero": dict(zip(F, zero)), "H": dict(zip(F, H)), "nonlinear": dict(zip(F, nl)), "condlinear": dict(zip(F, cl))}


# ── 判据与选择规则（纯函数）─────────────────────────────────────────────────
def test_gate_requires_mean_five_percent_and_every_fold_beating_both_baselines():
    rb = _rb([0.60] * 3, [0.50] * 3, [0.47, 0.47, 0.47], [0.49, 0.49, 0.49])
    g = REP.gate(rb)
    assert g["nonlinear"]["pass"] and abs(g["nonlinear"]["d_mean"] - 0.06) < 1e-12
    assert not g["condlinear"]["pass"] and g["condlinear"]["per_fold_vs_H"] == {"ab5": True, "ab6": True, "ab7": True}
    assert abs(g["condlinear"]["d_mean"] - 0.02) < 1e-12                    # 逐折都优于 H，但平均不到 5%
    rb2 = _rb([0.60] * 3, [0.50] * 3, [0.40, 0.40, 0.51], [0.47] * 3)        # 平均好很多也救不了一折
    assert not REP.gate(rb2)["nonlinear"]["pass"]
    rb3 = _rb([0.45] * 3, [0.50] * 3, [0.46] * 3, [0.40] * 3)                # 优于 H 但不优于零响应
    assert not REP.gate(rb3)["nonlinear"]["pass"] and REP.gate(rb3)["condlinear"]["pass"]


def test_gate_aggregates_relative_reduction_per_fold_then_averages():
    rb = _rb([1.0] * 3, [0.5, 0.4, 0.8], [0.45, 0.36, 0.72], [0.5] * 3)
    assert abs(REP.gate(rb)["nonlinear"]["d_mean"] - 0.10) < 1e-12           # 每折都降 10%（相对），不是百分点


def test_choose_prefers_condlinear_unless_nonlinear_wins_every_fold_by_two_percent():
    both = _rb([0.60] * 3, [0.50] * 3, [0.44, 0.44, 0.44], [0.46, 0.46, 0.46])
    g = REP.gate(both)
    assert REP.choose(g, both) == "nonlinear"                               # (0.46 − 0.44)/0.46 = 4.3% ≥ 2%
    close = _rb([0.60] * 3, [0.50] * 3, [0.455, 0.455, 0.455], [0.46] * 3)
    assert REP.choose(REP.gate(close), close) == "condlinear"               # 只低 1.1%
    split = _rb([0.60] * 3, [0.50] * 3, [0.40, 0.40, 0.465], [0.46] * 3)
    assert REP.choose(REP.gate(split), split) == "condlinear"               # 有一折更差
    only_nl = _rb([0.60] * 3, [0.50] * 3, [0.44] * 3, [0.49] * 3)
    assert REP.choose(REP.gate(only_nl), only_nl) == "nonlinear"
    none = _rb([0.60] * 3, [0.50] * 3, [0.49] * 3, [0.49] * 3)
    assert REP.choose(REP.gate(none), none) is None


def test_seed_mean_then_fold_order():
    rows = lambda num: [{"num": num, "den": 1.0}]
    per_seed = {"ab5": {0: rows(0.2), 1: rows(0.4)}}
    assert abs(REP.seed_mean_rho(per_seed)["ab5"] - 0.3) < 1e-12


def test_beta_is_na_when_the_increment_is_identically_zero():
    assert REP.beta([{"num": 0.0, "den": 0.0}]) is None
    assert abs(REP.beta([{"num": 1.0, "den": 2.0}, {"num": 1.0, "den": 2.0}]) - 0.5) < 1e-12


def test_beta_across_seeds_averages_only_defined_seeds():
    b = REP.beta_seed_mean({0: [{"num": 1.0, "den": 2.0}], 1: [{"num": 0.0, "den": 0.0}], 2: [{"num": 3.0, "den": 4.0}]})
    assert b["per_seed"] == {0: 0.5, 1: None, 2: 0.75} and b["n_defined"] == 2 and abs(b["mean"] - 0.625) < 1e-12
    assert REP.beta_seed_mean({0: [{"num": 0.0, "den": 0.0}]})["mean"] is None


def test_code_fingerprint_covers_the_report():
    assert "q2_report.py" in TR.CODE_FILES_Q2


# ── 整条流水线在合成数据上跑一遍，再核验与汇总 ──────────────────────────────
@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    import signal
    out = tmp_path_factory.mktemp("q2out")
    mp = pytest.MonkeyPatch()
    eps = T.synthetic_set(2)

    def fake_load(folds=D2.FOLDS, kinds=("bare", "corr", "aux"), check=True):
        o = [e for e in eps if e.kind in kinds and (e.kind == "aux" or e.fold in folds)]
        if check:
            D2.check_roster(o, keys=sorted({f"{e.fold}/{e.arm}" for e in o}))
        return o

    data = out / "data.json"
    data.write_text("{}")
    mp.setattr(D2, "load", fake_load)
    mp.setattr(D2, "ROSTER", T.roster_of(eps))
    mp.setattr(RUN, "roster_files", lambda: [data])
    mp.setitem(TR.CONFIG, "ckpts", (0, 1, 2))
    mp.setitem(TR.CONFIG, "max_steps", 2)
    mp.setattr(signal, "alarm", lambda sec: None)
    import q1_g2ref as G
    mp.setattr(G, "load_g2", lambda: None)
    mp.setattr(G, "episode_ehat", lambda g2, e, ws, R, b: np.zeros_like(e.gap))
    pre = out / "PREREG.md"
    pre.write_text(f"**状态：已冻结（测试）**\ncode_sha={TR.code_sha()[:16]}，config_hash={TR.config_hash()}\n")
    RUN.cmd_prepare(out, prereg=pre)
    for f in F:
        RUN.cmd_refs(out, f, prereg=pre)
        for s in TR.CONFIG["seeds"]:
            for v in RUN.train_sessions(f):
                RUN.cmd_nom_inner(out, f, v, s, prereg=pre)
            RUN.cmd_nom_final(out, f, s, prereg=pre)
            for form in TR.CONFIG["forms"]:
                for v in RUN.train_sessions(f):
                    RUN.cmd_inc_inner(out, form, f, v, s, prereg=pre)
                RUN.cmd_inc_final(out, form, f, s, prereg=pre)
    yield out, mp
    mp.undo()


def test_complete_tree_verifies_and_aggregates(tree):
    out, _ = tree
    assert REP.verify(out) == []
    s = REP.summarize(out)
    rb = s["rho_bar"]
    assert set(rb) == {"zero", "H", "nonlinear", "condlinear"} and set(rb["zero"]) == set(F)
    j = [json.loads(RUN.nom_final_base(out, "ab6", k).with_suffix(".json").read_text()) for k in (0, 1, 2)]
    want = np.mean([sum(r["num"] for r in x["zero"]) / sum(r["den"] for r in x["zero"]) for x in j])
    assert abs(rb["zero"]["ab6"] - want) < 1e-12                           # 先种子平均再按折
    assert s["choice"] in (None, "nonlinear", "condlinear") and set(s["steps"]) == {"nominal", "nonlinear", "condlinear"}
    assert set(s["aux"]) == {"P2", "P3"} and isinstance(s["aux"]["P2"]["beta"]["H"]["ab5"]["mean"], float)
    assert set(s["q1_recheck"]) == {"nominal", "g2"} and set(s["lead"]) == set(rb)
    md = REP.markdown(s)
    assert "开发检验" in md and "不能把胜负单独解释成非线性是否有用" in md and "step*" in md and "不等于统计显著" in md


@pytest.mark.parametrize("damage", ["missing_final", "tampered_pt", "tampered_npz", "changed_inner", "stale_config",
                                    "wrong_rows", "stale_inputs", "late"])
def test_damaged_trees_are_rejected(tree, damage, tmp_path):
    import shutil
    src, _ = tree
    out = tmp_path / "copy"
    shutil.copytree(src, out)
    jf = RUN.inc_final_base(out, "condlinear", "ab7", 1).with_suffix(".json")
    if damage == "missing_final":
        jf.unlink()
    elif damage == "tampered_pt":
        jf.with_suffix(".pt").write_bytes(b"changed")
    elif damage == "changed_inner":
        p = RUN.nom_inner_json(out, "ab5", "ab6", 2)
        j = json.loads(p.read_text()); j["scores"]["0"]["rho"] = -1.0; p.write_text(json.dumps(j))
    elif damage == "stale_config":
        j = json.loads(jf.read_text()); j["config_hash"] = "0" * 16; jf.write_text(json.dumps(j))
    elif damage == "wrong_rows":
        j = json.loads(jf.read_text()); j["cand"] = j["cand"][1:]; jf.write_text(json.dumps(j))
    elif damage == "tampered_npz":
        jf.with_suffix(".npz").write_bytes(b"changed")
    elif damage == "stale_inputs":                                      # 标定 R/b 文件的记录哈希与现在不符
        p = RUN.inputs_file(out)
        I = json.loads(p.read_text()); k = sorted(I["calib_files"])[0]; I["calib_files"][k] = "0" * 64
        p.write_text(json.dumps(I))
    elif damage == "late":
        d = json.loads(RUN.deadline_file(out).read_text())
        j = json.loads(jf.read_text()); j["finished_at"] = d["deadline"] + 1; jf.write_text(json.dumps(j))
    if damage == "late":
        assert REP.run_status(out)[0] is False and REP.main(["--out", str(out)]) == 3
    else:
        assert REP.verify(out) != [] and REP.main(["--out", str(out)]) == 4
        s = json.loads((out / "summary.json").read_text())
        assert s["status"] == "核验未通过" and "rho_bar" not in s


def test_main_marks_incomplete_runs_and_gives_no_verdict(tree, tmp_path, capsys):
    import shutil
    src, _ = tree
    out = tmp_path / "copy"
    shutil.copytree(src, out)
    RUN.mark_incomplete(out, "测试")
    assert REP.main(["--out", str(out)]) == 3
    s = json.loads((out / "summary.json").read_text())
    assert s["status"] == "未完成" and "rho_bar" not in s
    assert REP.main(["--out", str(src)]) == 0
    assert json.loads((src / "summary.json").read_text())["status"] == "完成"
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_q2_report.py -q`
Expected: `ModuleNotFoundError: No module named 'q2_report'`

- [ ] **Step 3: 实现** —— `sim/q2/q2_report.py`

```python
"""阶段二汇总 —— `reports/PREREG_Q2_STAGE2.md` §6、§7、§11。

会话误差比 `ρ = Σ_e ‖P(gap − ê)‖² / Σ_e ‖P(gap)‖²`（留出会话的补正运行内按集池化）。
**汇总顺序固定**：先对种子平均误差，得每折 ρ̄；`d_f = (ρ̄_H,f − ρ̄_候选,f) / ρ̄_H,f`，三折等权平均。
**判据**（每个候选单独判断，三条都满足才通过）：三折平均 d ≥ 5%；每折 ρ̄_候选 < ρ̄_H；每折 ρ̄_候选 < ρ̄_零响应。
**两种形式之间**（事前定死，不逐折挑赢家）：只有一个通过 ⇒ 用它；都通过 ⇒ 默认条件线性，只有非线性差分三折都更好、
且三折平均相对误差再降 ≥ 2%（分母 = 条件线性的 ρ̄）才保留非线性差分；都不通过 ⇒ 阶段二不通过。
⚠ 开发检验，不是密封确认；5% 与 2% 是事前的工程阈值，不等于统计显著，也不能换算成真机改善。
汇总前强制核验（任一不满足即拒绝汇总，并列出全部问题）：
- 3 折 × 3 种子 × {名义, 两种增量} 的内层表与终训产物、3 个 G-2 参照齐全；每份产物逐集覆盖冻结名单上该折该类的集。
- 全部 `code_sha` = 当前代码、`config_hash` = 冻结的选模规则；输入自 prepare 以来未变（数据、标定 R/b、固定 H、依赖、G-2
  逐一重算哈希）；每个作业的输入指纹与当前一致（终训含内层表哈希；增量含该折名义权重的哈希）。
- 由内层评分表重算 V 与 step*，与记录一致；`.pt` 的步数 = step*、权重哈希与记录一致；预测文件哈希一致。
本轮未完成（有「未完成」标记、缺截止时间记录、或任一产物晚于截止时间）⇒ 只写「未完成」，**不作通过判决**。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import q1_run as R1            # noqa: E402
import q1_train as TR1         # noqa: E402
import q2_data as D2           # noqa: E402
import q2_run as RUN           # noqa: E402
import q2_train as TR          # noqa: E402

FOLDS = D2.FOLDS
FORMS = RUN.FORMS
SEEDS = RUN.SEEDS
PREDICTORS = ("zero", "H") + FORMS
AUX_FOLD = "ab4"
AUX_ARMS = D2.AUX[AUX_FOLD]
LABEL = {"zero": "零响应（只用名义 2a）", "H": "固定 H", "nonlinear": "非线性差分", "condlinear": "条件线性",
         "nominal": "名义 2a"}


# ── 应有的产物与冻结名单 ────────────────────────────────────────────────────
def artifacts(out) -> dict:
    """→ {键: json 路径}：应有的全部 84 份产物（3 参照 + 18 名义内层 + 9 名义终训 + 36 增量内层 + 18 增量终训）。"""
    A = {}
    for f in FOLDS:
        A[("refs", f)] = RUN.ref_json(out, f)
        for s in SEEDS:
            for v in RUN.train_sessions(f):
                A[("nom-inner", f, v, s)] = RUN.nom_inner_json(out, f, v, s)
            A[("nom-final", f, s)] = RUN.nom_final_base(out, f, s).with_suffix(".json")
            for form in FORMS:
                for v in RUN.train_sessions(f):
                    A[("inc-inner", form, f, v, s)] = RUN.inc_inner_json(out, form, f, v, s)
                A[("inc-final", form, f, s)] = RUN.inc_final_base(out, form, f, s).with_suffix(".json")
    return A


def expected(fold: str, kind: str) -> list:
    """冻结名单上该会话该类（bare / corr / aux）的全部集名。"""
    arms = {"bare": D2.BARE, "corr": D2.CORR, "aux": D2.AUX}[kind][fold]
    return sorted(n for a in arms for n in D2.ROSTER.get(f"{fold}/{a}", []))


def _ident(key) -> dict:
    ph = key[0]
    if ph == "refs":
        return {"kind": "g2", "fold": key[1]}
    if ph == "nom-inner":
        return {"kind": ph, "fold": key[1], "val": key[2], "seed": key[3]}
    if ph == "nom-final":
        return {"kind": ph, "fold": key[1], "seed": key[2]}
    if ph == "inc-inner":
        return {"kind": ph, "form": key[1], "fold": key[2], "val": key[3], "seed": key[4]}
    return {"kind": ph, "form": key[1], "fold": key[2], "seed": key[3]}


# 每类产物里的逐集行 → 应覆盖的（会话, 类）
ROWS = {"refs": {"episodes": "bare"},
        "nom-final": {"q1": "bare", "zero": "corr", "H": "corr", "aux_zero": "aux", "aux_H": "aux", "aux_H_beta": "aux"},
        "inc-final": {"cand": "corr", "aux_cand": "aux", "aux_beta": "aux"}}


# ── 核验 ──────────────────────────────────────────────────────────────────
def _check_rows(P: list, path, j: dict, ph: str, fold: str) -> None:
    for key, kind in ROWS.get(ph, {}).items():
        want = expected(AUX_FOLD if kind == "aux" else fold, kind)
        rows = j.get(key)
        got = sorted(r["name"] for r in rows) if isinstance(rows, list) else []
        if got != want:
            P.append(f"{path}: {key} 的集与冻结名单不符（{len(got)} 集，应为 {len(want)}）")


def _check_inputs(P: list, out, I: dict, key, path, j: dict) -> None:
    ph = key[0]
    if ph == "refs":
        exp = RUN.ref_inputs(I)
    elif ph == "nom-inner":
        exp = RUN.nom_inner_inputs(I, key[1], key[2])
    elif ph == "nom-final":
        exp = RUN.nom_final_inputs(out, I, key[1], key[2])
    elif ph == "inc-inner":
        exp = RUN.inc_inner_inputs(out, I, *key[1:])
    else:
        exp = RUN.inc_final_inputs(out, I, *key[1:])
    if exp is None or j.get("inputs") != exp or j.get("input_fp") != RUN.fp_of(exp):
        P.append(f"{path}: 输入指纹与当前的数据 / 标定 / H / 内层表 / 名义权重不符")
    if ph in ("nom-inner", "inc-inner"):
        fold, val = (key[1], key[2]) if ph == "nom-inner" else (key[2], key[3])
        if j.get("train_sessions") != [RUN.inner_train(fold, val)]:
            P.append(f"{path}: 内层训练会话 {j.get('train_sessions')} 不对")


def _check_final(P: list, out, key, path, j: dict) -> None:
    """终训：由内层表重算的 V 与 step*、边界标记、权重与预测文件。"""
    import torch
    if key[0] == "nom-final":
        fold, seed, form = key[1], key[2], None
        inner = [RUN.nom_inner_json(out, fold, v, seed) for v in RUN.train_sessions(fold)]
    else:
        form, fold, seed = key[1], key[2], key[3]
        inner = [RUN.inc_inner_json(out, form, fold, v, seed) for v in RUN.train_sessions(fold)]
    if j.get("train_sessions") != list(RUN.train_sessions(fold)):
        P.append(f"{path}: 训练会话 {j.get('train_sessions')} 不对")
    try:
        V, step = RUN.selection(inner, TR.CONFIG["ckpts"])
    except (RuntimeError, KeyError, TypeError, ValueError) as err:
        P.append(f"{path}: 无法由内层表重算 step*（{err}）")
        return
    if j.get("V") != {str(s): V[s] for s in sorted(V)} or j.get("step_star") != step:
        P.append(f"{path}: 记录的 V / step* 与内层评分表重算不符")
    if j.get("boundary") != (step == TR.CONFIG["max_steps"]):
        P.append(f"{path}: 边界标记与 step* 不符")
    pt, npz = path.with_suffix(".pt"), path.with_suffix(".npz")
    if not pt.exists() or RUN.file_sha(pt) != j.get("pt_sha256"):
        P.append(f"{pt}: 缺失或与记录的 pt_sha256 不符")
    else:
        ck = torch.load(pt, weights_only=False)
        if ck.get("step") != step or ck.get("form", form) != form or \
                TR1.state_dict_sha(ck["state_dict"]) != j.get("state_sha256"):
            P.append(f"{pt}: 权重的步数、形式或哈希与 step* 记录不符")
    if not npz.exists() or RUN.file_sha(npz) != j.get("npz_sha256"):
        P.append(f"{npz}: 缺失或与记录的 npz_sha256 不符")


def verify(out) -> list:
    """完整性核验 → 问题列表（空 = 通过）。"""
    out = Path(out)
    try:
        I = RUN.read_inputs(out)
    except RuntimeError as err:
        return [str(err)]
    P = []
    for group in ("data_files", "calib_files", "h_files", "deps_files", "g2_files"):
        for rel, h in I.get(group, {}).items():
            f = RUN.ROOT / rel
            if not f.exists() or RUN.file_sha(f) != h:
                P.append(f"输入文件自 prepare 以来已变或缺失：{rel}")
    code_now, cfg_now = TR.code_sha(), TR.config_hash()
    if I.get("code_sha") != code_now:
        P.append("代码自 prepare 以来已变")
    if I.get("env") != R1.env_versions():
        P.append("库版本自 prepare 以来已变")
    if I.get("roster") != D2.ROSTER:
        P.append("冻结名单自 prepare 以来已变")
    d = RUN._load(RUN.deadline_file(out))
    if d and (d.get("code_sha"), d.get("config_hash")) != (code_now, cfg_now):
        P.append("deadline.json 属于另一版冻结的实验")
    for key, path in artifacts(out).items():
        j = RUN._load(path)
        if j is None:
            P.append(f"缺产物或无法解析：{path}")
            continue
        ident = _ident(key)
        if {k: j.get(k) for k in ident} != ident:
            P.append(f"{path}: 身份字段与路径不符")
        if j.get("code_sha") != code_now:
            P.append(f"{path}: code_sha 与当前代码不一致")
        if j.get("config_hash") != cfg_now:
            P.append(f"{path}: config_hash 与冻结的选模规则不符")
        _check_inputs(P, out, I, key, path, j)
        _check_rows(P, path, j, key[0], ident["fold"])
        if key[0] in ("nom-final", "inc-final"):
            _check_final(P, out, key, path, j)
    return P


def run_status(out) -> tuple:
    """本轮是否在截止时间内完成：有「未完成」标记、缺截止时间记录、或任一产物晚于截止时间 ⇒ 未完成。"""
    out = Path(out)
    mark = RUN.incomplete_file(out)
    if mark.exists():
        return False, json.loads(mark.read_text()).get("reason", "已标记未完成")
    d = RUN._load(RUN.deadline_file(out))
    if not d:
        return False, "缺本轮截止时间记录（deadline.json）"
    late = []
    for p in artifacts(out).values():
        j = RUN._load(p)
        if j is not None and j.get("finished_at", float("inf")) > d["deadline"]:
            late.append(p)
    if late:
        return False, f"{len(late)} 份产物晚于本轮截止时间（如 {late[0].relative_to(out)}）"
    return True, ""


# ── 判据（纯函数）─────────────────────────────────────────────────────────
def seed_mean_rho(per_seed: dict) -> dict:
    """{fold: {seed: 逐集行}} → {fold: 各种子池化 ρ 的平均}（先对种子平均误差，不平均预测）。"""
    return {f: float(np.mean([TR.pooled_rho(rows) for rows in by_seed.values()])) for f, by_seed in per_seed.items()}


def per_seed_rho(per_seed: dict) -> dict:
    return {f: {s: TR.pooled_rho(rows) for s, rows in by_seed.items()} for f, by_seed in per_seed.items()}


def gate(rb: dict) -> dict:
    """rb = {预测: {fold: ρ̄}} → 每个候选的 d_f、三折平均 d、逐折比较与是否通过。"""
    out = {}
    for form in FORMS:
        c, h, z = rb[form], rb["H"], rb["zero"]
        d = {f: (h[f] - c[f]) / h[f] for f in FOLDS}
        vs_h = {f: bool(c[f] < h[f]) for f in FOLDS}
        vs_z = {f: bool(c[f] < z[f]) for f in FOLDS}
        d_mean = float(np.mean([d[f] for f in FOLDS]))
        out[form] = {"d": d, "d_mean": d_mean, "per_fold_vs_H": vs_h, "per_fold_vs_zero": vs_z,
                     "pass": bool(d_mean >= TR.CONFIG["gate_d"] and all(vs_h.values()) and all(vs_z.values()))}
    return out


def nonlin_vs_cond(rb: dict) -> dict:
    """非线性差分相对条件线性的逐折相对误差降低（分母 = 条件线性的 ρ̄）与三折平均；三折是否都更好。"""
    n, c = rb["nonlinear"], rb["condlinear"]
    rel = {f: (c[f] - n[f]) / c[f] for f in FOLDS}
    return {"rel": rel, "rel_mean": float(np.mean([rel[f] for f in FOLDS])),
            "better_all": all(n[f] < c[f] for f in FOLDS)}


def choose(g: dict, rb: dict):
    """→ "nonlinear" / "condlinear" / None（两个都不通过 ⇒ 阶段二不通过）。"""
    ok = [f for f in FORMS if g[f]["pass"]]
    if not ok:
        return None
    if len(ok) == 1:
        return ok[0]
    v = nonlin_vs_cond(rb)
    return "nonlinear" if v["better_all"] and v["rel_mean"] >= TR.CONFIG["nonlin_margin"] else "condlinear"


def beta(rows):
    """投影尺度 β = Σ_e 分子 / Σ_e 分母；增量恒为 0（分母 = 0）⇒ None（N/A）。"""
    num, den = sum(r["num"] for r in rows), sum(r["den"] for r in rows)
    return None if den == 0 else num / den


def beta_seed_mean(by_seed: dict) -> dict:
    """{seed: 逐集行} → 逐种子 β、有定义种子的平均与个数（全部 N/A ⇒ 平均 None）。"""
    per = {s: beta(rows) for s, rows in by_seed.items()}
    ok = [b for b in per.values() if b is not None]
    return {"per_seed": per, "mean": float(np.mean(ok)) if ok else None, "n_defined": len(ok)}


def lead_rho(by_seed: dict) -> dict:
    """{seed: 逐集行} → 按前瞻分组的 ρ（各组分子分母先对种子平均再池化）。"""
    acc, k = {}, len(by_seed)
    for rows in by_seed.values():
        for r in rows:
            for b, (n, d) in r["lead"].items():
                a = acc.setdefault(b, [0.0, 0.0])
                a[0] += n / k
                a[1] += d / k
    return {b: n / d for b, (n, d) in acc.items()}


# ── 汇总 ──────────────────────────────────────────────────────────────────
def collect(out) -> dict:
    """读终训与参照产物 → 逐种子的行（调用前须已通过 `verify`）。"""
    A = {k: RUN._load(p) for k, p in artifacts(out).items()}
    held = {p: {f: {} for f in FOLDS} for p in PREDICTORS}
    aux = {p: {f: {} for f in FOLDS} for p in PREDICTORS}
    braw = {p: {f: {} for f in FOLDS} for p in ("H",) + FORMS}
    q1 = {f: {} for f in FOLDS}
    steps = {b: {f: {} for f in FOLDS} for b in ("nominal",) + FORMS}
    for f in FOLDS:
        for s in SEEDS:
            n = A[("nom-final", f, s)]
            held["zero"][f][s], held["H"][f][s] = n["zero"], n["H"]
            aux["zero"][f][s], aux["H"][f][s], braw["H"][f][s] = n["aux_zero"], n["aux_H"], n["aux_H_beta"]
            q1[f][s] = n["q1"]
            steps["nominal"][f][s] = {"step_star": n["step_star"], "boundary": n["boundary"]}
            for form in FORMS:
                c = A[("inc-final", form, f, s)]
                held[form][f][s], aux[form][f][s], braw[form][f][s] = c["cand"], c["aux_cand"], c["aux_beta"]
                steps[form][f][s] = {"step_star": c["step_star"], "boundary": c["boundary"]}
    return {"held": held, "aux": aux, "beta": braw, "q1": q1, "steps": steps,
            "g2": {f: A[("refs", f)]["episodes"] for f in FOLDS}}


def aux_table(aux: dict, braw: dict) -> dict:
    """ab4 P2 / P3 分开：每折终训模型上的 ρ̄（种子平均）与 β（逐种子 + 有定义种子的平均）。"""
    arm = lambda rows, a: [r for r in rows if r["arm"] == a]
    return {a: {"rho": {p: seed_mean_rho({f: {s: arm(rows, a) for s, rows in aux[p][f].items()} for f in FOLDS})
                        for p in PREDICTORS},
                "beta": {p: {f: beta_seed_mean({s: arm(rows, a) for s, rows in braw[p][f].items()}) for f in FOLDS}
                         for p in braw}}
            for a in AUX_ARMS}


def summarize(out) -> dict:
    C = collect(out)
    rb = {p: seed_mean_rho(C["held"][p]) for p in PREDICTORS}
    g = gate(rb)
    choice = choose(g, rb)
    return {"rho_bar": rb, "rho_per_seed": {p: per_seed_rho(C["held"][p]) for p in PREDICTORS},
            "gate": g, "nonlin_vs_cond": nonlin_vs_cond(rb), "choice": choice, "stage2_pass": choice is not None,
            "q1_recheck": {"nominal": seed_mean_rho(C["q1"]), "g2": {f: TR.pooled_rho(C["g2"][f]) for f in FOLDS}},
            "aux": aux_table(C["aux"], C["beta"]),
            "lead": {p: lead_rho({s: [r for f in FOLDS for r in C["held"][p][f][s]] for s in SEEDS})
                     for p in PREDICTORS},
            "steps": C["steps"]}


def markdown(s: dict) -> str:
    rho = lambda x: f"{x:.4f}"
    pct = lambda x: f"{x * 100:+.2f}%"
    fold_head = "| " + " | ".join(FOLDS) + " |"
    L = ["## 阶段二（Q2）开发检验汇总", "",
         "⚠ 这是开发检验，不是密封确认，也不是补正因果效应的直接验证。5% 与 2% 是事前的工程推进 / 选择阈值，"
         "不等于统计显著，也不能换算成真机改善。结论只在「维持 ab5/ab6 装置状态」的前提下有意义。", "",
         "### 留出会话补正运行上的 ρ̄（先对种子平均；越小越好）", "", "| 预测 " + fold_head, "|---" * 4 + "|"]
    for p in PREDICTORS:
        L.append(f"| {LABEL[p]} | " + " | ".join(rho(s["rho_bar"][p][f]) for f in FOLDS) + " |")
    L += ["", "### 判据（每个候选单独判断，三条都满足才通过）", "",
          "| 候选 | " + " | ".join(f"d_{f}" for f in FOLDS) + " | 三折平均 d | 逐折 < 固定 H | 逐折 < 零响应 | 判定 |",
          "|---" * 8 + "|"]
    for form in FORMS:
        g = s["gate"][form]
        why = [] if g["pass"] else (
            ([f"平均 d < {TR.CONFIG['gate_d'] * 100:.0f}%"] if g["d_mean"] < TR.CONFIG["gate_d"] else [])
            + [f"{f} 不优于 H" for f in FOLDS if not g["per_fold_vs_H"][f]]
            + [f"{f} 不优于零响应" for f in FOLDS if not g["per_fold_vs_zero"][f]])
        yn = lambda d: "是" if all(d.values()) else "否（" + "、".join(f for f in FOLDS if not d[f]) + "）"
        L.append(f"| {LABEL[form]} | " + " | ".join(pct(g["d"][f]) for f in FOLDS) + f" | {pct(g['d_mean'])} | "
                 f"{yn(g['per_fold_vs_H'])} | {yn(g['per_fold_vs_zero'])} | "
                 + ("**过**" if g["pass"] else "不过（" + "；".join(why) + "）") + " |")
    v = s["nonlin_vs_cond"]
    verdict = {None: "两个候选都不通过 ⇒ **阶段二不通过**，不进入阶段三。",
               "condlinear": "保留**条件线性**。", "nonlinear": "保留**非线性差分**（三折都更好，且平均相对误差再降 ≥ 2%）。"}
    L += ["", "### 两种形式之间（事前定死；不逐折挑赢家）", "",
          "- 非线性差分相对条件线性的相对误差降低（分母 = 条件线性的 ρ̄）：" + "、".join(
              f"{f} {pct(v['rel'][f])}" for f in FOLDS) + f"；三折平均 {pct(v['rel_mean'])}；三折都更好："
          + ("是" if v["better_all"] else "否"),
          f"- 结论：{verdict[s['choice']]}",
          "- 对照结论只能是「两套具体响应模型谁的折外预测更好」。两者还差着扰动记忆长度（61 对 20 帧）、四基核约束与参数化，"
          "不能把胜负单独解释成非线性是否有用。",
          "- 通过只意味着可以进入阶段三的离线 π 研究，不代表 π 会在真机上有效。",
          "", "## 只报告（不作判据）", "",
          "### 留出裸跑上的 Q1 复查（名义 2a 种子平均 ρ；G-2 同口径窗口重预测作背景）", "",
          "| 预测 " + fold_head, "|---" * 4 + "|",
          f"| {LABEL['nominal']} | " + " | ".join(rho(s["q1_recheck"]["nominal"][f]) for f in FOLDS) + " |",
          "| G-2 | " + " | ".join(rho(s["q1_recheck"]["g2"][f]) for f in FOLDS) + " |",
          "", "### 逐种子 ρ（留出补正运行；每格为种子 " + " / ".join(str(k) for k in SEEDS) + "）", "",
          "| 预测 " + fold_head, "|---" * 4 + "|"]
    for p in PREDICTORS:
        ps = s["rho_per_seed"][p]
        L.append(f"| {LABEL[p]} | " + " | ".join(" / ".join(rho(ps[f][k]) for k in SEEDS) for f in FOLDS) + " |")
    L += ["", "### ab4 辅助参考（旧装置；复用每折终训模型；P1 不参与；不用于选步、选模或判决）", ""]
    na = lambda b: "N/A" if b["mean"] is None else (
        f"{b['mean']:.3f}" + ("" if b["n_defined"] == len(SEEDS) else f"（{b['n_defined']}/{len(SEEDS)} 个种子有定义）"))
    for a in AUX_ARMS:
        t = s["aux"][a]
        L += [f"**{a}**：ρ̄（种子平均）与投影尺度 β（有定义种子的平均）；列 = 各折的终训模型", "",
              "| 预测 | " + " | ".join(f"ρ̄ {f}" for f in FOLDS) + " | " + " | ".join(f"β {f}" for f in FOLDS) + " |",
              "|---" * 7 + "|"]
        for p in PREDICTORS:
            bc = [na(t["beta"][p][f]) if p in t["beta"] else "—" for f in FOLDS]
            L.append(f"| {LABEL[p]} | " + " | ".join(rho(t["rho"][p][f]) for f in FOLDS) + " | " + " | ".join(bc) + " |")
        L.append("")
    L += ["- β 只是投影尺度，不是「方向和大小都正确」：β ≈ 1 时仍可能漏掉很大的正交响应；增量恒为 0（选中第 0 步）时 β 不定义，记 N/A。",
          "- 闭环运行下，随机探针不能自动保证与名义预测误差无关；扣掉名义预测后的残差不能称为「已分离的真实增量」。",
          "- P3/P2 的 β 比值不是严格线性检验（指令、种子不同，P3 还有裁剪）；P3 是跨幅值的压力参考。"
          "历史的 0.955、1.013 只作历史记录，不与本表直接比较。",
          "", "### 按前瞻分组的 ρ（留出补正运行按集池化、分子分母先对种子平均；带通混合相邻帧 ⇒ 近似）", ""]
    bins = list(s["lead"][PREDICTORS[0]])
    L += ["| 预测 | " + " | ".join(bins) + " |", "|---" * (len(bins) + 1) + "|"]
    for p in PREDICTORS:
        L.append(f"| {LABEL[p]} | " + " | ".join(rho(s["lead"][p][b]) for b in bins) + " |")
    top = TR.CONFIG["max_steps"]
    L += ["", "### step*（每格为种子 " + " / ".join(str(k) for k in SEEDS) + f"；† = 落在搜索边界 {top} 步）", "",
          "| 分支 " + fold_head, "|---" * 4 + "|"]
    hit = zero = False
    for b in ("nominal",) + FORMS:
        cells = []
        for f in FOLDS:
            ks = [s["steps"][b][f][k] for k in SEEDS]
            hit |= any(k["boundary"] for k in ks)
            zero |= b != "nominal" and any(k["step_star"] == 0 for k in ks)
            cells.append(" / ".join(f"{k['step_star']}{'†' if k['boundary'] else ''}" for k in ks))
        L.append(f"| {LABEL[b]} | " + " | ".join(cells) + " |")
    if hit:
        L += ["", f"† 最佳点落在搜索边界（{top} 步），尚不能确认训练充分；按预注册不追加步数。"]
    if zero:
        L += ["", "增量选中第 0 步 = 保留零响应（未训练），是有效选择；该处候选与零响应逐集相同。"]
    return "\n".join(L)


def _key_str(obj):
    """JSON 只接受字符串键：把整数种子键转成字符串。"""
    if isinstance(obj, dict):
        return {str(k): _key_str(v) for k, v in obj.items()}
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    ok, why = run_status(out)
    if not ok:
        R1.atomic_json(out / "summary.json", {"status": "未完成", "reason": why})
        print(f"**本轮未完成**：{why} —— 按预注册记为未完成，不作通过判决。", flush=True)
        return 3
    problems = verify(out)
    if problems:
        R1.atomic_json(out / "summary.json", {"status": "核验未通过", "problems": problems})
        print("产物不完整或不一致，拒绝汇总：\n  " + "\n  ".join(problems), flush=True)
        return 4
    s = {"status": "完成", **summarize(out)}
    R1.atomic_json(out / "summary.json", _key_str(s))
    md = markdown(s)
    (out / "summary.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 把汇总计入代码指纹** —— `sim/q2/q2_train.py:34-35`

把

```python
CODE_FILES_Q2 = ("q2_data.py", "q2_roster.json", "q2_windows.py", "q2_models.py", "q2_train.py",
                 "q2_run.py")
```

改为

```python
CODE_FILES_Q2 = ("q2_data.py", "q2_roster.json", "q2_windows.py", "q2_models.py", "q2_train.py",
                 "q2_run.py", "q2_report.py")
```

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest sim/q2 -q -m "not slow"`
Expected: `79 passed, 2 deselected`。`test_q2_report.py` 的 17 个约 11 s：先在合成集上跑通全部 3 折 × 3 种子 × 两种形式，再逐项损坏后核验。

- [ ] **Step 6: 提交**

```bash
git add sim/q2/q2_report.py sim/q2/test_q2_report.py sim/q2/q2_train.py
git commit -m "阶段二 Q2：汇总与判据（先种子后折、5% 与逐折、2% 选择规则；强制核验；未完成不判决）"
```

---

### Task 7: 只报告的波形图

**Files:**
- Create: `sim/q2/fig_q2.py`
- Test: `sim/q2/test_fig_q2.py`

**Interfaces:**
- Consumes：
  - Task 5：`nom_final_base`、`inc_final_base`、`load_eps`，以及终训产物 `.json` 的逐集 `num/den`、`.npz` 的预测
  - Task 6：`verify`
  - `sim/q1/fig_pp2_q1.bp`（去均值 + bandpass41）；`q1_data.ALIGN_L / FS`
- Produces：
  - `fold_rows(out, fold, seed=0) -> ([{name, ep, pred, rho}], {form: step*})`、`pick(rows) -> (中位, 最难)`、`plot(rows, fold, steps, path)`
  - `main(argv)`：核验不过 → 4，不作图；否则写 `outputs/q2_stage2/fig/q2_{fold}_seed0.png`（3 张）

- [ ] **Step 1: 写失败的测试** —— `sim/q2/test_fig_q2.py`

```python
import json

import numpy as np

import fig_q2 as FIG
import q2_run as RUN
import q2_testing as T


def _fake_fold(out, eps, steps):
    """一折种子 0 的终训产物（只有作图要读的字段）：零响应 / H / 两种候选的逐集预测与 num、den。"""
    rng = np.random.default_rng(0)
    rows = lambda s: [{"name": e.name, "num": s * (i + 1), "den": 10.0} for i, e in enumerate(eps)]
    k = lambda e: e.name.replace("/", "__")
    base = RUN.nom_final_base(out, "ab7", 0)
    base.parent.mkdir(parents=True)
    np.savez(base.with_suffix(".npz"), **{f"{p}__{k(e)}": rng.normal(size=(e.n, 3)) for e in eps for p in ("nom", "H")})
    base.with_suffix(".json").write_text(json.dumps({"zero": rows(1.0), "H": rows(2.0)}))
    for form, s in steps.items():
        b = RUN.inc_final_base(out, form, "ab7", 0)
        b.parent.mkdir(parents=True)
        np.savez(b.with_suffix(".npz"), **{f"cand__{k(e)}": rng.normal(size=(e.n, 3)) for e in eps})
        b.with_suffix(".json").write_text(json.dumps({"cand": rows(0.5), "step_star": s}))


def test_fold_rows_reads_saved_predictions_and_per_episode_rho(tmp_path, monkeypatch):
    eps = [T.synthetic(s, "ab7", "D", "corr") for s in range(3)]
    monkeypatch.setattr(RUN, "load_eps", lambda folds, kinds: eps)
    _fake_fold(tmp_path, eps, {"nonlinear": 0, "condlinear": 250})
    rows, steps = FIG.fold_rows(tmp_path, "ab7")
    z = np.load(RUN.nom_final_base(tmp_path, "ab7", 0).with_suffix(".npz"))
    assert [r["name"] for r in rows] == [e.name for e in eps] and steps == {"nonlinear": 0, "condlinear": 250}
    assert np.array_equal(rows[1]["pred"]["zero"], z["nom__ab7__D_ep1"])
    assert rows[1]["rho"] == {"zero": 0.2, "H": 0.4, "nonlinear": 0.1, "condlinear": 0.1}


def test_plot_picks_median_and_hardest_by_zero_response_and_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(5):
        e = T.synthetic(i, "ab7", "D", "corr")
        rows.append({"name": e.name, "ep": e, "pred": {p: rng.normal(0, 0.01, (e.n, 3)) for p in FIG.PREDS},
                     "rho": {p: 1.0 + i for p in FIG.PREDS}})
    med, hard = FIG.pick(rows)
    assert (med["name"], hard["name"]) == (rows[2]["name"], rows[4]["name"])
    p = tmp_path / "f.png"
    FIG.plot(rows, "ab7", {"nonlinear": 0, "condlinear": 250}, p)
    assert p.exists() and p.stat().st_size > 10_000


def test_refuses_to_draw_from_an_unverified_run(tmp_path, capsys):
    assert FIG.main(["--out", str(tmp_path)]) == 4
    assert "不作图" in capsys.readouterr().out and not (tmp_path / "fig").exists()


def test_not_part_of_the_code_fingerprint():
    import q2_train as TR
    assert "fig_q2.py" not in TR.CODE_FILES_Q2
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_fig_q2.py -q`
Expected: `ModuleNotFoundError: No module named 'fig_q2'`

- [ ] **Step 3: 实现** —— `sim/q2/fig_q2.py`

```python
"""阶段二只报告的波形图：留出补正运行上「预测 IMU vs 实测 IMU」（每折一张，种子 0；中位集 + 最难集）。

沿用「改完 G 必出波形图」的版式（`sim/q1/fig_pp2_q1.py`）：0.5–9 Hz 带通、中位集与最难集、口径注记。
- 左列：实测 IMU 与四种预测的 IMU（`ideal_t + ê_t`，与实测 `ω_{t+3}` 对齐；无大域位移，同评分口径的 L = 3）。
- 右列：实测 gap 减名义预测（= 名义没解释的部分）与三种预测增量（固定 H、两种候选）。这只是直观对照，
  **不是**「已分离的真实增量」（闭环运行里残差还含名义模型自身的误差）。
- 中位 / 最难按零响应的逐集 ρ_e 排序（与候选无关）。预测取自终训产物的 `.npz`，作图前先通过 `q2_report.verify`。
**只作图，不进任何判据**；不选步、不选模。不在代码指纹里。

    python -u sim/q2/fig_q2.py --out outputs/q2_stage2
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import fig_pp2_q1 as F1        # noqa: E402
import q1_data as D1           # noqa: E402
import q2_run as RUN           # noqa: E402

SEED = 0
FOLDS, FORMS = RUN.FOLDS, RUN.FORMS
PREDS = ("zero", "H") + FORMS
STYLE = {"zero": ("零応答（名義 2a のみ）", "#9aa0a6", ":", 1.8), "H": ("固定 H", "#e07a1f", "-.", 1.8),
         "nonlinear": ("非線形差分", "#8b2fb0", "--", 2.0), "condlinear": ("条件付き線形", "#2a9d4b", "--", 2.0)}
C_REAL = "#1f6fb2"


def fold_rows(out, fold: str, seed: int = SEED) -> tuple:
    """→ ([{"name", "ep", "pred": {预测: ê (n, 3)}, "rho": {预测: ρ_e}}], {候选: step*})。"""
    nb = RUN.nom_final_base(out, fold, seed)
    nj, nz = json.loads(nb.with_suffix(".json").read_text()), np.load(nb.with_suffix(".npz"))
    cj = {f: json.loads(RUN.inc_final_base(out, f, fold, seed).with_suffix(".json").read_text()) for f in FORMS}
    cz = {f: np.load(RUN.inc_final_base(out, f, fold, seed).with_suffix(".npz")) for f in FORMS}
    rho = lambda rows: {r["name"]: r["num"] / r["den"] for r in rows}
    R = {"zero": rho(nj["zero"]), "H": rho(nj["H"]), **{f: rho(cj[f]["cand"]) for f in FORMS}}
    rows = []
    for e in RUN.load_eps((fold,), ("corr",)):
        k = e.name.replace("/", "__")
        pred = {"zero": nz[f"nom__{k}"], "H": nz[f"H__{k}"], **{f: cz[f][f"cand__{k}"] for f in FORMS}}
        rows.append({"name": e.name, "ep": e, "pred": pred, "rho": {p: R[p][e.name] for p in PREDS}})
    return rows, {f: cj[f]["step_star"] for f in FORMS}


def pick(rows) -> tuple:
    """按零响应的逐集 ρ_e 取中位集与最难集。"""
    order = np.argsort([r["rho"]["zero"] for r in rows], kind="stable")
    return rows[order[len(order) // 2]], rows[order[-1]]


def plot(rows, fold: str, steps: dict, path: Path, span: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.16, left=0.06, right=0.99, top=0.83, bottom=0.13)
    for i, (r, title) in enumerate(zip(pick(rows), ("① 中央値のエピソード", "② 最も難しいエピソード"))):
        e = r["ep"]
        real = F1.bp(e.w[D1.ALIGN_L:D1.ALIGN_L + e.n])
        ax_i = int(np.argmax(np.abs(real).max(0)))
        c = int(np.argmax(np.abs(real[:, ax_i])))
        s = max(0, min(c - span // 2, len(real) - span))
        t = np.arange(span) / D1.FS
        a = fig.add_subplot(gs[i, 0])
        a.plot(t, real[s:s + span, ax_i], color=C_REAL, lw=2.4, label="実機 IMU")
        for p in PREDS:
            lab, col, ls, lw = STYLE[p]
            a.plot(t, F1.bp(e.ideal[:e.n] + r["pred"][p])[s:s + span, ax_i], color=col, ls=ls, lw=lw, label=lab)
        a.set_title(f"{title}  {r['name']}  軸 ω{'xyz'[ax_i]}\nρ_e：" + " · ".join(
            f"{STYLE[p][0].split('（')[0]} {r['rho'][p]:.3f}" for p in PREDS), fontsize=10.5, loc="left", color="#22223b")
        a.set_ylabel("角速度 (rad/s)\n0.5–9 Hz", fontsize=10)
        b = fig.add_subplot(gs[i, 1], sharex=a)
        b.plot(t, F1.bp(e.gap - r["pred"]["zero"])[s:s + span, ax_i], color=C_REAL, lw=2.2,
               label="実測 gap − 名義予測")
        for p in PREDS[1:]:
            lab, col, ls, lw = STYLE[p]
            b.plot(t, F1.bp(r["pred"][p] - r["pred"]["zero"])[s:s + span, ax_i], color=col, ls=ls, lw=lw,
                   label=f"{lab} の増分")
        b.set_title("名義が説明しない部分 と 予測増分（直観的対照のみ）", fontsize=10.5, loc="left", color="#22223b")
        for x in (a, b):
            x.set_xlabel("時間 (s)", fontsize=10)
            x.grid(alpha=0.25)
        if i == 0:
            a.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
            b.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper right")
    note = "・".join(f"{STYLE[f][0]} step*={steps[f]}" + ("（零応答と同一）" if steps[f] == 0 else "") for f in FORMS)
    fig.suptitle(f"阶段二 開発検証：{fold} 留出・補正運転 — 予測 IMU と 実機 IMU（seed {SEED}）",
                 fontsize=16, fontweight="bold", color="#22223b", y=0.985)
    fig.text(0.5, 0.915, f"帯域 0.5–9 Hz・大域シフトなし（評価と同じ L=3）・chunk 先頭起点・末端クランプ・{note}",
             ha="center", fontsize=10.5, color="#5a5a72")
    fig.text(0.5, 0.012, "※ 開発検証（密封確認ではない）・図は報告のみで判定に使わない。ρ_e = ‖P(gap−ê)‖²/‖P(gap)‖²"
             "（1.0 = ê ≡ 0 相当）。中央値／最難は零応答の ρ_e で選択。\n"
             "δa はログに記録された要求補正量（実行変位ではない）；当該 chunk の補正は離線で与えた条件。"
             "右列の残差は名義モデル自身の誤差も含む（分離した真の増分ではない）。",
             ha="center", fontsize=9, color="#c1121f", linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    gc.disable()
    import q2_report as REP
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    problems = REP.verify(a.out)
    if problems:
        print("产物核验未通过，不作图：\n  " + "\n  ".join(problems), flush=True)
        return 4
    for fold in FOLDS:
        rows, steps = fold_rows(a.out, fold)
        p = Path(a.out) / "fig" / f"q2_{fold}_seed{SEED}.png"
        plot(rows, fold, steps, p)
        print(f"  → {p}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q2/test_fig_q2.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q2/fig_q2.py sim/q2/test_fig_q2.py
git commit -m "阶段二 Q2：只报告的波形图（留出补正运行的中位集与最难集；不进指纹与判据）"
```

---

### Task 8: 正式运行脚本

**Files:**
- Create: `sim/q2/run_q2_stage2.sh`（可执行）
- Test: `sim/q2/test_run_q2_stage2.py`

**Interfaces:**
- Consumes：`q2_run.py` 的子命令与退出码（`prepare` 未冻结 → 2；作业超时 → 3）、`list-jobs --phase` 的逐行位置参数；`q2_report.py`；`fig_q2.py`
- Produces：`bash sim/q2/run_q2_stage2.sh`。环境变量：`OUT`（默认 `outputs/q2_stage2`）、`JOBS`（默认 15）、`PY`
  - 每次运行新建 `$OUT/logs/attempt_NNN/`（`mkdir` 原子取号），里面有：
    - 逐作业日志 `<阶段>_<参数>.log`，以及 `prepare.log`、`report.log`、`fig.log`、各阶段待办清单；
    - `status.tsv`：每行「时间、阶段、参数、退出码」。
  - 续跑不覆盖以前尝试的日志与退出状态。

- [ ] **Step 1: 写失败的测试**（假解释器只记录参数，不训练、不读数据）—— `sim/q2/test_run_q2_stage2.py`

```python
"""正式运行脚本的空跑测试：假解释器只记参数，**不训练、不读数据**。"""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_q2_stage2.sh"
PHASES = ("refs", "nom-inner", "nom-final", "inc-inner", "inc-final")
TODO = {"refs": ["ab5", "ab7"], "nom-inner": ["ab5 ab6 0", "ab7 ab5 2"], "nom-final": ["ab5 0"],
        "inc-inner": ["condlinear ab5 ab6 0", "nonlinear ab7 ab6 1"], "inc-final": ["nonlinear ab6 2"]}


def _fake(tmp_path, todo, extra="", silent=None):
    """假解释器：记录每次调用。list-jobs 列出该阶段尚未完成的作业；作业成功退出即记为完成
    （silent 阶段除外：成功退出却不记完成，模拟作业没写出产物）。extra：插在最前面的 shell 片段，用来注入失败。"""
    calls = tmp_path / "calls.log"
    for ph, jobs in todo.items():
        (tmp_path / f"todo_{ph}.txt").write_text("".join(j + "\n" for j in jobs))
    keep = "" if silent is None else f'[ "$sub" = "{silent}" ] && exit 0\n  '
    fake = tmp_path / "fakepy.sh"
    fake.write_text(f"""#!/usr/bin/env bash
echo "$* | BLAS=$OPENBLAS_NUM_THREADS" >> "{calls}"
{extra}
if [ "$2" = "sim/q2/q2_run.py" ]; then
  sub=$3
  case "$sub" in
    prepare) echo "prepared"; exit 0 ;;
    list-jobs)
      f="{tmp_path}/todo_$7.txt"; [ -f "$f" ] || exit 0
      while IFS= read -r l; do [ -f "{tmp_path}/done_$7_$(echo "$l" | tr ' ' _)" ] || echo "$l"; done < "$f"
      exit 0 ;;
  esac
  shift 5
  echo "ran $sub $*"
  {keep}touch "{tmp_path}/done_${{sub}}_$(echo "$*" | tr ' ' _)"
  exit 0
fi
echo "ran $*"
""")
    fake.chmod(0o755)
    return fake, calls


def _run(tmp_path, fake):
    env = {**os.environ, "PY": str(fake), "OUT": str(tmp_path / "out"), "JOBS": "2"}
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def _is(line, sub):
    return f"q2_run.py {sub} " in line


def _cmds(calls, sub):
    return [line.split("|")[0].split() for line in calls.read_text().splitlines() if _is(line, sub)]


def _status(attempt):
    """status.tsv → [[阶段, 参数, 退出码]]（第一列是时间）。"""
    return [line.split("\t")[1:] for line in (attempt / "status.tsv").read_text().splitlines()]


def test_phases_run_in_order_with_their_arguments(tmp_path):
    fake, calls = _fake(tmp_path, TODO)
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    for ph in PHASES:
        n = len(TODO[ph][0].split())
        assert sorted(c[-n:] for c in _cmds(calls, ph)) == sorted(t.split() for t in TODO[ph])
        assert all(c[c.index(ph) + 1] == "--out" for c in _cmds(calls, ph))
    log = calls.read_text().splitlines()
    pos = {ph: [i for i, x in enumerate(log) if _is(x, ph)] for ph in PHASES}
    for a, b in zip(PHASES, PHASES[1:]):
        assert max(pos[a]) < min(pos[b]), f"{a} 必须全部先于 {b}"
    report = next(i for i, x in enumerate(log) if "q2_report.py" in x)
    assert max(pos["inc-final"]) < report < next(i for i, x in enumerate(log) if "fig_q2.py" in x)


def test_resume_with_nothing_left_still_reports_and_draws(tmp_path):
    fake, calls = _fake(tmp_path, {})
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(_cmds(calls, ph) for ph in PHASES)
    assert "q2_report.py" in calls.read_text() and "fig_q2.py" in calls.read_text()


def test_prepare_refusal_stops_before_any_job(tmp_path):
    """预注册未冻结 ⇒ prepare 退出码 2 ⇒ 不跑任何作业；退出码记进本次尝试的 status.tsv。"""
    fake, calls = _fake(tmp_path, TODO, extra='case "$*" in *"q2_run.py prepare"*) exit 2 ;; esac')
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and not any(_cmds(calls, ph) for ph in PHASES)
    assert _status(tmp_path / "out" / "logs" / "attempt_001") == [["prepare", "", "2"]]


def test_failed_job_stops_the_run_before_later_phases(tmp_path):
    fake, calls = _fake(tmp_path, TODO, extra='case "$*" in *"q2_run.py nom-final "*) exit 1 ;; esac')
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and "nom-final" in r.stdout
    assert not _cmds(calls, "inc-inner") and "q2_report.py" not in calls.read_text()


def test_a_job_stopped_by_the_deadline_stops_the_run_with_a_clear_message(tmp_path):
    fake, calls = _fake(tmp_path, TODO, extra='case "$*" in *"q2_run.py inc-inner "*) exit 3 ;; esac')
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and "INCOMPLETE.json" in r.stdout
    assert not _cmds(calls, "inc-final") and "q2_report.py" not in calls.read_text()


def test_jobs_still_listed_after_a_phase_stop_the_run(tmp_path):
    """作业成功退出、list-jobs 却仍列出它（如作业没写出产物）⇒ 停止。"""
    fake, calls = _fake(tmp_path, TODO, silent="nom-final")
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and "todo_nom-final_after" in r.stdout and not _cmds(calls, "inc-inner")


def test_each_attempt_keeps_its_own_logs_and_exit_status(tmp_path):
    """续跑不覆盖失败证据：每次运行新建 attempt 目录，逐作业日志 + status.tsv（时间、阶段、参数、退出码）。"""
    fail = 'case "$*" in *"q2_run.py nom-final "*) echo "boom $*"; exit 1 ;; esac'
    fake, calls = _fake(tmp_path, TODO, extra=fail)
    assert _run(tmp_path, fake).returncode != 0
    fake.write_text(fake.read_text().replace(fail, ""))                 # 修好后续跑
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    logs = tmp_path / "out" / "logs"
    a1, a2 = logs / "attempt_001", logs / "attempt_002"
    assert sorted(p.name for p in logs.iterdir()) == ["attempt_001", "attempt_002"]
    assert "boom" in (a1 / "nom-final_ab5_0.log").read_text()          # 第一次的失败日志还在
    assert "ran nom-final ab5 0" in (a2 / "nom-final_ab5_0.log").read_text()
    s1, s2 = _status(a1), _status(a2)
    assert ["prepare", "", "0"] in s1 and ["nom-final", "ab5 0", "1"] in s1
    assert not any(row[0] == "inc-inner" for row in s1)                # 第一次在名义终训失败处就停了
    assert ["nom-final", "ab5 0", "0"] in s2 and ["report", "", "0"] in s2 and ["fig", "", "0"] in s2
    assert not any(row[0] == "nom-inner" for row in s2)                 # 已完成的作业续跑时跳过
    assert str(a2) in r.stdout


def test_blas_threads_are_pinned(tmp_path):
    fake, calls = _fake(tmp_path, TODO)
    assert _run(tmp_path, fake).returncode == 0
    assert all(line.endswith("BLAS=1") for line in calls.read_text().splitlines())
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q2/test_run_q2_stage2.py -q`
Expected: `8 failed`（脚本不存在，`bash` 返回 127）

- [ ] **Step 3: 实现** —— `sim/q2/run_q2_stage2.sh`

```bash
#!/usr/bin/env bash
# 阶段二（Q2 开发检验）正式运行 —— **只在 reports/PREREG_Q2_STAGE2.md 冻结后**，在用户自己的终端里运行。
# 五个阶段依次进行，每个阶段全部完成才进入下一个：
#   refs（G-2 在留出裸跑上，Q1 复查的背景）→ nom-inner → nom-final（选步、冻结名义 2a、零响应与固定 H 的分数）
#   → inc-inner（固定名义模型下的增量选步验证）→ inc-final（候选的分数）；然后汇总（q2_report.py）与作图（fig_q2.py）。
# 可重入：配置指纹与输入指纹都一致（终训还须 .pt / .npz 的哈希一致）的作业会被跳过；中断后直接重跑本脚本即可。
# 每次运行是一次「尝试」：日志写进新建的 $OUT/logs/attempt_NNN/（逐作业日志；status.tsv 记录时间、阶段、参数、退出码），
# 续跑不覆盖以前的失败证据。
# 墙钟上限：第一次 prepare 记录截止时间（+24 小时），续跑不重置；到点后作业拒绝/中止并写 INCOMPLETE.json，
# 汇总只写「未完成」、不作通过判决。
# 修复若涉及冻结代码（sim/q2 的 q2_*.py 与名单、只读复用的 sim/q1 代码）：须重新冻结预注册，并换新的 OUT 从头运行 ——
# 旧目录的截止时间记录绑定旧指纹，prepare 与作业都会拒绝。本脚本与作图脚本不在代码指纹里（不改变任何作业的计算）。
#   bash sim/q2/run_q2_stage2.sh            # 默认 OUT=outputs/q2_stage2 JOBS=15
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/home/enolab-guo/miniconda3/envs/lerobot/bin/python}
OUT=${OUT:-outputs/q2_stage2}
JOBS=${JOBS:-15}
mkdir -p "$OUT/logs"
n=1
while ! mkdir "$OUT/logs/attempt_$(printf %03d "$n")" 2>/dev/null; do n=$((n + 1)); done
LOG="$OUT/logs/attempt_$(printf %03d "$n")"
echo "本次尝试的日志：$LOG"

step() {  # step <名字> <命令…>：输出同时写进本次尝试的 <名字>.log，退出码记进 status.tsv
  local name=$1 rc=0
  shift
  "$@" 2>&1 | tee "$LOG/$name.log" || rc=$?
  printf '%s\t%s\t\t%s\n' "$(date -Iseconds)" "$name" "$rc" >> "$LOG/status.tsv"
  return "$rc"
}

run_phase() {
  local ph=$1
  "$PY" -u sim/q2/q2_run.py list-jobs --out "$OUT" --phase "$ph" > "$LOG/todo_$ph.txt"
  echo "[$ph] 待跑作业 $(wc -l < "$LOG/todo_$ph.txt") 个，并行 $JOBS"
  # -r：没有待跑作业时不调用；每行是该阶段作业的位置参数；逐作业日志与退出码留在本次尝试的目录里
  xargs -r -P "$JOBS" -L 1 sh -c \
    'py=$0; ph=$1; out=$2; log=$3; shift 3; name="${ph}_$(echo "$*" | tr " " _)"; rc=0
     "$py" -u sim/q2/q2_run.py "$ph" --out "$out" "$@" > "$log/$name.log" 2>&1 || rc=$?
     printf "%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$ph" "$*" "$rc" >> "$log/status.tsv"
     exit "$rc"' \
    "$PY" "$ph" "$OUT" "$LOG" < "$LOG/todo_$ph.txt" \
    || { echo "[$ph] 有作业失败或已到本轮截止时间（见 $LOG/${ph}_*.log 与 $LOG/status.tsv；超时会写 $OUT/INCOMPLETE.json）"; exit 1; }
  "$PY" -u sim/q2/q2_run.py list-jobs --out "$OUT" --phase "$ph" > "$LOG/todo_${ph}_after.txt"
  if [ -s "$LOG/todo_${ph}_after.txt" ]; then
    echo "[$ph] 仍有未完成的作业（见 $LOG/todo_${ph}_after.txt 与 ${ph}_*.log）—— 修复后重跑本脚本"; exit 1
  fi
}

step prepare "$PY" -u sim/q2/q2_run.py prepare --out "$OUT"
for ph in refs nom-inner nom-final inc-inner inc-final; do
  run_phase "$ph"
done
step report "$PY" -u sim/q2/q2_report.py --out "$OUT"
step fig "$PY" -u sim/q2/fig_q2.py --out "$OUT"
echo "完成：$OUT/summary.json、$OUT/summary.md、$OUT/fig/（本次尝试的日志：$LOG）"
```

然后 `chmod +x sim/q2/run_q2_stage2.sh`。

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q2/test_run_q2_stage2.py -q`
Expected: `8 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q2/run_q2_stage2.sh sim/q2/test_run_q2_stage2.py
git commit -m "阶段二 Q2：正式运行脚本（五个阶段依次、可重入、失败或超时即停；每次尝试独立日志与退出状态；冻结后在用户终端运行）"
```

---

### Task 9: 全量自测 + 耗时测量 + 回填预注册 → **停，等用户确认 24 小时并冻结**

**Files:**
- Modify: `reports/PREREG_Q2_STAGE2.md`（抬头、§1 最后一条、§8；**状态行仍为草稿**）

- [ ] **Step 1: 全量自测**

Run: `$PY -m pytest sim/q2 -q`
Expected: `93 passed`，约 2 分钟。任何失败都先修，不进入 Step 2。

- [ ] **Step 2: 确认阶段一没有被改动**

Run:

```bash
git status --short sim/q1
$PY -c "import gc,sys; gc.disable(); sys.path.insert(0,'sim/q1'); import q1_train as T; print(T.code_sha()[:16], T.config_hash())"
```

Expected：第一条无输出；第二条打印 `d6a823ba1ed4a862 be83d2e77958e85c`。

- [ ] **Step 3: 耗时测量**（另行授权的短程测速，只为填预算）

这一步：
- 只用折 ab5 的一个内层训练会话 ab6：名义在 ab6 的裸跑上训 30 步，两种增量在 ab6 的补正运行上各训 30 步；
- 只在这些训练集上计评分耗时；
- **不评分外层留出会话、不选步、不保存产物、不打印任何 ρ**。

Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -u sim/q2/q2_run.py timing --steps 30`（`--steps` 只接受 1–30）

Expected：
- 3 行 `<nominal | nonlinear | condlinear>: x.xxx s/步 · 评分 x.xxx s/集`；
- 1 行合计：核·小时上限，以及 15 路并行、四个阶段依次进行的墙钟估计。

参考：阶段一的 2a 是 0.010–0.017 s/步。阶段一实测墙钟是估计的约 1.5 倍（14 h 13 min 对 9.6 h，15 路并行在 10 个物理核上争用 + 续跑）。

- [ ] **Step 4: 回填预注册** —— `reports/PREREG_Q2_STAGE2.md`

取指纹：

```bash
git rev-parse --short HEAD
$PY -c "import gc,sys; gc.disable(); sys.path[:0]=['sim/q1','sim/q2']; import q2_train as T; print(T.code_sha()[:16], T.config_hash())"
sha256sum sim/q2/q2_roster.json
```

改四处（其余正文不动）：
1. 状态行整行改为 `**状态：草稿（<日期>），实施完成，待用户确认预算并冻结。冻结前不运行任何正式入口。**`
2. 抬头第 2 条（「代码、`code_sha`…实施后填入；冻结前不运行任何正式入口。」）整条替换为：

   ```markdown
   - 实施：`reports/PLAN_IMPL_Q2_STAGE2.md`；代码 `sim/q2/`（提交 `<git 短哈希>`；code_sha=<16 位>，config_hash=<16 位>）；数据名单 `sim/q2/q2_roster.json`（sha256 `<前 16 位>`，在代码指纹里）。
     冻结时把状态行整行改为 `**状态：已冻结（<日期>，用户确认）**`。正式入口只认这一行，并核对 `code_sha=` 与 `config_hash=`；代码或规则改动后须重新冻结。
   ```

   ⚠ 冻结闸门用正则找**第一个** `code_sha=` / `config_hash=` 后的 16 位十六进制。全文不要再出现别的 `code_sha=…` / `config_hash=…` 写法；例如阶段一的指纹，要写也只用「code_sha `d6a8…`」这种不带 `=` 的形式。
3. §1 最后一条的「冻结为新的名单文件」后补上「`sim/q2/q2_roster.json`」。
4. §8 第 1 条（「实施后实测，填入此处。默认上限：墙钟 ≤ 24 小时。」）替换为 Step 3 的结果：
   - 三个 s/步、三个 s/集；
   - 合计核·小时与墙钟估计；
   - 按阶段一 1.5 倍争用系数折算后的墙钟；
   - 一句「上限暂定 24 小时，待用户确认」。

- [ ] **Step 5: 核对闸门能读出指纹**

用一份临时副本把状态行换成「已冻结」，检查指纹与当前代码一致；再确认真实预注册仍被拒绝：

```bash
$PY - <<'EOF'
import gc, pathlib, re, sys, tempfile; gc.disable(); sys.path[:0] = ["sim/q1", "sim/q2"]
import q2_run as R
t = re.sub(r"^\*\*状态：.*$", "**状态：已冻结（闸门自检，不是真的冻结）**", R.PREREG.read_text(), count=1, flags=re.M)
p = pathlib.Path(tempfile.mkdtemp()) / "prereg_check.md"
p.write_text(t)
R.require_frozen(p)
print("指纹与当前代码一致")
try:
    R.require_frozen()
except R.NotFrozenError as err:
    print("真实预注册仍未冻结：", err)
EOF
```

Expected：先打印「指纹与当前代码一致」，再打印「真实预注册仍未冻结：拒绝运行：PREREG_Q2_STAGE2.md 尚未冻结…」。

- [ ] **Step 6: 提交**

```bash
git add reports/PREREG_Q2_STAGE2.md
git commit -m "阶段二 Q2：预注册回填 code_sha / config_hash / 名单文件与实测耗时（仍为草稿，待用户冻结）"
```

- [ ] **Step 7: 停**

向用户报告：
- 测试结果；
- 阶段一指纹不变；
- 耗时估计，以及按 1.5 倍折算后与 24 小时上限的对比；
- 两处我加的默认（β 跨种子汇总、波形图）。

请用户确认 24 小时上限，并决定是否冻结。**冻结前不运行 `prepare / refs / nom-* / inc-*`。**

---

### Task 10: 正式运行与结果（**仅在预注册冻结后**）

**Files:**
- Modify: `reports/PREREG_Q2_STAGE2.md`（只改状态行）
- Create: `reports/RESULT_Q2_STAGE2.md`

- [ ] **Step 1: 记录冻结**（用户明确同意后）

把状态行整行改为 `**状态：已冻结（<日期>，用户确认）**`，并确认 `code_sha=` 与 `config_hash=` 仍等于当前代码（Task 9 Step 4 的命令）。然后提交：

```bash
git add reports/PREREG_Q2_STAGE2.md
git commit -m "阶段二 Q2：预注册冻结"
```

记下这个提交哈希。冻结后代码一变，`config_hash` 就会变，旧产物自动作废。

- [ ] **Step 2: 正式运行**

在用户自己的终端运行（也可以在本会话输入 `! bash sim/q2/run_q2_stage2.sh`）：

Run: `bash sim/q2/run_q2_stage2.sh`
Expected：结尾打印「完成：outputs/q2_stage2/summary.json、outputs/q2_stage2/summary.md、outputs/q2_stage2/fig/」。

脚本五个阶段依次进行，可重入。每次运行新建 `outputs/q2_stage2/logs/attempt_NNN/`，以前尝试的日志与退出状态保留。
- 某个作业失败：查本次尝试的 `<阶段>_*.log` 与 `status.tsv`，然后分两种情况：
  - 瞬时故障（如原生库段错误）、代码没改：在原目录重跑本脚本。已完成的作业跳过；JSON、`.pt`、`.npz` 任一缺失或损坏的终训作业重新排队；
  - 修复涉及冻结代码（`CODE_FILES_Q2` / `CODE_FILES_Q1` 内的文件）：**停**。修复并提交 → 回到 Task 9 Step 4–6 回填新指纹 → 用户重新冻结 → 换新的 `OUT`（如 `OUT=outputs/q2_stage2_r2`）从头运行；
  - 旧目录只留作记录，不续跑。它的截止时间记录绑定旧指纹，入口会拒绝。
- 到截止时间：作业写 `INCOMPLETE.json`，本轮按预注册记为未完成，**不临时改判据、不补跑挑结果**。

- [ ] **Step 3: 核对产物**

依次核对：
- 参照：`ls outputs/q2_stage2/refs/g2/*.json | wc -l` → `3`
- 名义：`ls outputs/q2_stage2/nom/inner/seed*/*.json | wc -l` → `18`；`ls outputs/q2_stage2/nom/final/seed*/*.pt | wc -l` → `9`
- 增量：`ls outputs/q2_stage2/inc/*/inner/seed*/*.json | wc -l` → `36`；`ls outputs/q2_stage2/inc/*/final/seed*/*.pt | wc -l` → `18`
- 汇总：`$PY -u sim/q2/q2_report.py --out outputs/q2_stage2` 退出码 0，即通过全部核验（冻结的选模规则、输入指纹、由内层表重算的 step*、权重与预测文件、名单、截止时间）
- 图：`ls outputs/q2_stage2/fig/` 有 3 张 `q2_*_seed0.png`

- [ ] **Step 4: 写结果** —— `reports/RESULT_Q2_STAGE2.md`

数字一律取自 `summary.json` 与 `summary.md`。结构如下：

```markdown
# 结果 —— G→π v2 阶段二（Q2 开发检验：预测补正造成的变化）

预注册 `reports/PREREG_Q2_STAGE2.md`（冻结于 `<哈希>`）· 代码 `sim/q2/`（`<哈希>`；code_sha `<16 位>`、config_hash `<16 位>`）· 零机时 · 墙钟 <…>（上限 <…> 小时）

⚠ 开发检验，不是密封确认，也不是补正因果效应的直接验证。5% 与 2% 是事前的工程推进 / 选择阈值，不等于统计显著，也不能换算成真机改善。结论只在「维持 ab5/ab6 装置状态」的前提下有意义。

## 一、判据（逐候选；先对种子平均，再按折）
<ρ̄ 表与判据表（取自 summary.md）；逐条写明每个候选过 / 不过及原因>

## 二、两种形式之间（事前定死）
<逐折相对误差降低与三折平均、是否三折都更好、结论：保留哪一个 / 阶段二不通过>
- 对照结论只能是「两套具体响应模型谁的折外预测更好」。两者还差着扰动记忆长度（61 对 20 帧）、四基核约束与参数化，不能把胜负单独解释成非线性是否有用。
- 每折同时留出了「会话 + 补正策略」：不能把结果单独归因于对未见策略的泛化。

## 三、只报告：留出裸跑上的 Q1 复查
<每折名义 2a 的 ρ 与 G-2 并列；据此判断失败来自名义预测，还是来自加入响应之后的预测>

## 四、只报告：ab4 辅助参考（旧装置）
<P2、P3 分开的 ρ̄ 与 β（有定义种子的平均，注明个数）；β 的三处解读限定照抄预注册 §7；不用于选步、选模或判决>

## 五、只报告：前瞻分层与逐种子
## 六、step*（含第 0 步与搜索边界）
## 七、波形图（只报告，不作判据）
## 八、证据边界（预注册 §9 原文）
## 九、执行偏差
<各次尝试（logs/attempt_NNN/status.tsv）里失败的作业、退出码与原因，用同一冻结配置续跑的情况；墙钟；任何偏离。没有则写「无」>
## 十、产物
<outputs/q2_stage2 下 summary.json 的 sha256、各终训产物的 pt / npz sha256 摘要；冻结提交哈希>
```

- [ ] **Step 5: 提交结果**

```bash
git add reports/RESULT_Q2_STAGE2.md
git commit -m "阶段二 Q2：结果（逐候选判据、两种形式之间、Q1 复查、ab4 辅助参考与波形图）"
```

- [ ] **Step 6: 更新记忆**

- 改 `memory/finding_q1_stage1.md` 或新建 `finding_q2_stage2.md`：结论、措辞约束、产物路径；
- 在 `MEMORY.md` 里加一行索引。

- [ ] **Step 7: 停**

阶段三（离线 π 研究）只有在至少一个候选通过时才可进入，而且要另写预注册；pilot 仍推迟。**不自动开始。**

---

## 修订记录

- 2026-10-03 初版（`0af206e`）。代码已在暂存区镜像跑通（快速 84 + 集成 2），并按任务顺序逐个重放：每个任务实现前测试失败、实现后通过，最终文件与镜像逐字节相同。
- 2026-10-03 按用户审阅修订（执行方式定为 Native；两处默认认可并写进预注册 `405a7a5`）：
  1. `q2_run.cmd_timing` 入口强制 1 ≤ steps ≤ 30，越界在读数据、训练之前抛 `TimingRefused`（入口退出码 2）。
     测试：`test_timing_refuses_out_of_range_steps_before_loading_or_training`（0 / −1 / 31 / 10000）。
  2. 终训作业的完成判定改为 `final_done`：JSON 指纹之外，还须 `.pt`、`.npz` 存在且哈希与记录一致，损坏就重新排队。
     测试：`test_final_jobs_requeue_when_weights_or_predictions_are_missing_or_corrupt`。
  3. 运行脚本每次运行新建 `logs/attempt_NNN/`，逐作业日志 + `status.tsv`（时间、阶段、参数、退出码），续跑不覆盖失败证据。
     测试：`test_each_attempt_keeps_its_own_logs_and_exit_status`。空跑测试的假解释器改为「list-jobs 只列未完成的作业」，更接近真实续跑。
  - 「修复涉及冻结代码须重新冻结、不沿用旧实验」写进 Task 10 与运行脚本说明，并加回归测试 `test_changed_code_cannot_continue_an_old_run`。这一条现有机制已满足，测试一写就通过。
  - 测试由 86 个增至 93 个（快速 91 + 集成 2）。重新按任务顺序重放：每个任务先红后绿，最终与镜像逐字节相同。
- 2026-10-03 Native 实施 Task 1–9（`38ddb20`…`2afef23` 代码与本计划代码块逐字节一致；`2936dac` 预注册回填）。随后按整支独立复审（结论「修好后可冻结」：0 Critical、2 Important、8 Minor）修复，见 `12773a4`。**自此仓库代码在以下几处与本计划代码块不同，以仓库为准**：
  1. `q2_report.aux_table` / `markdown`：ab4 逐折之外加三折等权平均（设计稿第 5 项原文要求，计划漏了）；β 每格显示「有定义种子的平均（有效种子数：逐种子值）」。新增 `fold_mean`。
  2. `q2_run.selection`：非有限的验证 ρ（训练发散）按 +∞ 处理，永不被选中。原先 NaN 不会被选中，但记录的 V 含 NaN，核验时 NaN ≠ NaN 会让整轮无法汇总。汇总列出这些检查点。
  3. `q2_report.main` 入口 `gc.disable()`，`__main__` 改用 `os._exit`（全局约束「入口 gc.disable()」）。
  - 新测试 5 个：`test_fold_mean_is_equal_weight_over_defined_folds`、`test_ab4_reports_the_three_fold_average_and_per_seed_beta`、`test_selection_treats_a_non_finite_validation_rho_as_never_best`、`test_a_diverged_inner_checkpoint_keeps_the_run_verifiable`、`test_report_entry_disables_gc`。测试共 98 个（快速 96 + 集成 2）。
  - 预注册重新回填：`code_sha=` / `config_hash=` 换成新值；§5 写明非有限 ρ 的处理（待用户确认）；§7 写明 ab4 三折等权平均；§8 的耗时注记补上分波次的影响。
  - 8 个 Minor 记为暂缓（台账 `Final: minor (deferred)`），没有修。
- 2026-10-03 冻结前按用户三项决定补齐（`77103cd`），仍以仓库为准：
  1. 保持 24 小时上限，不调整预算或训练配置。
  2. 非有限验证 ρ 按 +∞ 只用于内层选步。异常检查点在汇总里逐个报告（分支、折、验证会话、种子、步）。所有检查点都没有有限分数时 `selection` 抛 `RuntimeError`，该作业失败。判据加 `finite`：候选或基准任一折 ρ̄ 非有限即不通过，并写明原因。
  3. 超时出口：`mark_incomplete` 经 `tempfile.mkstemp` 的独立临时文件原子改名；`run_status` 见到标记即判未完成，标记损坏也不崩溃。
  - 新测试：`test_selection_fails_when_no_checkpoint_has_a_finite_score`、`test_incomplete_marker_is_written_through_its_own_temp_file`、`test_a_corrupt_incomplete_marker_still_records_an_incomplete_run`、`test_outer_scores_never_pass_by_ignoring_a_non_finite_value`；`test_a_diverged_inner_checkpoint_keeps_the_run_verifiable` 改为核对逐个报告的内容。测试共 102 个（快速 100 + 集成 2）。
  - 预注册回填最终指纹并写入上述决定，随后冻结。
