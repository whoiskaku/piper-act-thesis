# #76 真机 单chunk(end-to-end) 动作平滑验证 — 录制协议

**目标**：在真机上、用**单 chunk(无边界/无 blend/无闭环重规划)**口径，验证 #75 sim 残差
(Hattn cut2.0, 命令 2-4.5Hz −40%) 的动作平滑是否迁移到真机。隔离 #69-real 的闭环边界问题
(= 老师建议的 one-chunk/end-to-end 条件)。

**为什么"第一个 chunk"= 单 chunk**：`chunk_size=100=n_action_steps`(ACT 冻结)。每条 episode
开头 ACT 在起点预测一次、执行 100 步、其间不重新查询、无边界、residual 模式无 blend
→ 分析 **frame 0-100** 即一个干净单 chunk。任务整体 ~570 步是多 chunk, 但只取第一 chunk 即可。

## 条件 (不需要改 wrapper)
- **A = 裸 ACT**：`REFINE_MODE=off`（或不设 `REFINE_CKPT`）
- **B = ACT + Hattn cut2.0 残差**：`REFINE_MODE=residual`，
  `REFINE_CKPT=outputs/train/refine_net/phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150/ckpt_best.pt`
  （residual 模式 = 仅残差、**无 blend**，对齐单 chunk 口径）

## 配对要求 (matched)
- **同一 home 起始位姿** + **同一物体位置**（A、B 两组尽量摆一致）；
- 每条件 **N ≥ 6–9** 集；能交错录最好（减少漂移）；
- 录完整 episode 即可（分析只取 frame 0-100）；
- 录制带 IMU：`--robot.inference_time=true --robot.enable_imu=true`（沿用 #67/#69 的 control_robot 录制流程，仅改上面两个 env）。

## 分析 (工具已就绪)
```bash
conda activate lerobot && cd ~/claude-test-2/lerobot-piper
python sim/eval_realchunk_action.py --datasets ACT=<A数据集名> refined=<B数据集名>
```
输出两表：**FIRST-CHUNK(主验证)** 与 FULL-EPISODE(对照)，列 = 命令 / 执行关节 / IMU accel / IMU gyro 的 2-4.5Hz RMS + 相对裸ACT比值。

## 判读 (预期)
- **first-chunk 命令**：refined **<1**（应 ~0.7，像 sim −40%；若 ≈1 = sim→real 输入漂移削弱了残差）；
- **first-chunk 执行关节**：看真机控制器是否把命令平滑透传（<1 = 透传）；
- **first-chunk IMU**：看是否被指令无关地板盖住（#74：真机 IMU 相干<0.15，很可能 ≈1）；
- **full-episode**：refined 命令 **>1**（闭环边界淹没 = #69-real），印证"整流落地需边界处理(blend)"。
- 另记**任务成功率**(A vs B)。

## 可选：命令的严格配对 (无需配对物体位置)
只录 B(residual) 一组, 离线在 B 录制的 frame-0 观测上重跑 frozen ACT → 得残差前裸 ACT chunk[0] = A,
与录到的 refined chunk[0] = B 严格同观测对比。需要时告诉我, 我加进分析脚本(re-run ACT, 同 deploy_eval)。
