# 研究计划（更新版）

## 研究命题
改良 ACT 架构，使其在 pick-and-place 任务上输出更平滑的动作。
（ACT 模型使用 `lerobot-piper` 目录下已跑通的实现，baseline 训练与推理已完成）

## 方法
引入末端 IMU 输入 + Gazebo 仿真 IMU 作为训练期辅助监督。

**直观解释：**
- ACT 原本只学"看到 o，输出 a"
- 现在强迫它同时学"看到 o，心里想着一个理想机械臂（sim）在这里会有什么震动表现"
- 推理时不需要任何 IMU、不需要 sim，部署零额外开销

## 硬件与参考仓库
- **机械臂**：六轴 Piper
- **现实 IMU**：TSND151，安装在末端，姿态标定已完成
  - 参考库：https://github.com/eno-lab/tsnd.git
  - 只支持软件触发，目标采样率 **30 Hz**
- **仿真**：`piper_ros`（https://github.com/agilexrobotics/piper_ros.git）
  - Gazebo 中已跑通，URDF 末端挂载 IMU 插件
  - **场景中只有机械臂和 IMU，无桌面/物块**
- 现有参考数据：`/home/enolab-guo/test.jpeg` 为一组 real IMU vs sim IMU 对比图

## 频率对齐
| 项目 | 频率 |
|---|---|
| Piper 控制频率 | 30 Hz |
| 相机 | 30 Hz |
| IMU 采样 | 30 Hz |
| action chunk 预测 | 100 步（≈3.33 s） |
| action chunk 执行 | 10 步（≈333 ms） |

三者全部 30 Hz，一帧 ↔ 一个 IMU 样本 ↔ 一个 action 步，无需重采样。

## 架构改动（A + B 一次到位，不做消融开关）

### A. Encoder 新增 real IMU 输入分支
- 输入：3 轴加速度 × 20 步历史窗（≈667 ms）
- 处理：`ImuTemporalEncoder`（1D CNN）→ 单 token 注入 `encoder_in_tokens`
- 注入点：`lerobot/common/policies/act/modeling_act.py` L392–530 的 token 构造处

### B. Encoder 后新增辅助预测头
- 输出：**执行期 10 步** sim IMU 加速度（3 × 10 = 30 维）
- 训练监督：Gazebo 运动学回放 action → sim IMU
- 损失：`L = L_action + λ · L_imu`，λ 初始 0.1（可调）
- 推理时丢弃此头，零额外开销

### Config 扩展（`configuration_act.py`）
- `imu_history_len: int = 20`
- `imu_future_len: int = 10`
- `imu_aux_loss_weight: float = 0.1`

## 仿真 IMU 获取方案（方案 B：运动学回放）
- **不搭** pick-and-place 场景（不引入物块、桌面、接触物理）
- 离线回放脚本：读取已录 demo 的 action 序列 → 喂 `piper_ros` Gazebo → 让机械臂在空中走一遍同样轨迹 → 采集 sim IMU → 写回同一 episode 的 parquet
- **不做重力补偿**（real/sim 都保留重力分量，只要同坐标即可）

## 数据
- 先录 3~5 条试点，验证管线通畅
- 再扩至 30 条 pick-and-place 正式训练数据
- 每条存：
  - real action, state, images（现有）
  - `observation.imu_real`：3 轴加速度（新，每帧 1 样本，历史窗通过 `delta_timestamps` 拼）
  - `observation.imu_sim_future`：执行期 10 步的 sim IMU（3 × 10 = 30 维，训练监督用）

## 训练
监督学习优先（数据量小、调参成本低）；RL 非必须，初期不采用。

## 评价指标
- **jerk RMS**：action 时序三次差分后的 RMS（反映动作平滑度）
- **PSD gap**：real vs sim 加速度在 0–15 Hz 功率谱的 L2 距离（Welch 法）
- **任务成功率**：必须不下降

## 暂不做的事（已明确）
- **不做 4 组消融实验**，聚焦跑通改造后的模型（A + B full 版）
- **不做重力补偿**
- **不搭 Gazebo 场景**（运动学回放即可）
- **不在初期用 RL**

## 执行优先级

### P0｜评测脚本
baseline ACT 已跑通，仅需实现 `jerk RMS` 与 `PSD gap` 计算脚本，用于后续判断改动是否真的更平滑。

### P1｜数据管线（风险最大）
1. TSND151 采集封装（30 Hz 软件触发，参考 `eno-lab/tsnd`）
2. `LeRobotDataset` 扩展：新增 `observation.imu_real` 与 `observation.imu_sim_future` 两个 feature
3. Gazebo 运动学回放脚本：action 序列 → sim IMU，写回 parquet
4. 先录 3~5 条试点数据验证管线，再扩至 30 条

### P2｜模型改动（A + B full 版）
- `ImuTemporalEncoder`（1D CNN）→ token 注入 encoder
- 辅助头 MLP → 30 维 sim IMU 预测
- Config 扩展三个新字段

### P3｜端到端训练跑通
- 先用试点数据 overfit 一轮，确认 loss 下降、shape 无误
- 录满 30 条后跑正式训练
- 用 P0 指标对比原 baseline
