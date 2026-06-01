# Stage 4.0 — Object-aware transfer audit 结果

> 配套：`RESEARCH_PLAN_v4.md` §Stage4.0 / `STAGE3B_RESULTS.md`(Stage 3 + 真机 de-risk）
> 脚本：`sim/build_object_model.py`(B+ 胶带圆环模型) / `sim/calib_contact.py` / `sim/calib_weld.py` / `sim/transfer_audit_weld.py`

## 对象与建模

真实 object = ASKUL 养护胶带卷（薄圆环）：外径 87.04 / 内径 76.12 / 高 5.35mm，**实测质量 61g**，平放捏环壁。

**B+ 模型**：N=24 段 box 拼成 annulus（碰撞 proxy，保留环孔）+ 显式解析 annulus 惯量、COM 在环心（距抓取点 ~40mm）→ 重力力矩/动态负载物理正确。摆放：环心 = demo 抓取帧 FK 指垫点 + mean_radius 沿夹爪水平前向（图片确认侧前接近）。

## 关键 bug（已修，曾污染所有早期结果）

- **桌子太小**：half_extent=0.30 居中原点，但抓取点在 x≈0.33 → 胶带 spawn 在桌外坠落到 −88m → weld 把它从 −88m 拽回 → 爆炸/不稳定/carry_lift=0。**修：half_extent→0.50**。
- **weld relpose**：默认锁 reset 相对位姿，attach 时产生假 spike → **修：attach 时设 relpose=当前相对位姿 + 速度同步 gripper**。
- **weld vs 接触打架**：welded tape 仍与指 mesh 碰撞 → 不稳定 → **修：tape 与 table 同 contype/conaffinity=2，不与手臂碰撞**（weld=附着代理，不建模指-环接触）。

## 标定

- **接触 solref**（桌面/自由接触）：timeconst 0.2 → peak≈7.8 ≈ 真机抓取 IMU 靶 7.5（`calib_contact`）。
- **weld solref**（payload 附着，单独配）：0.1 → carry_lift~79mm 稳定、accel_hp 收敛 0.13 无振铃；<0.02 振铃（`calib_weld`）。
- **自由刚体夹持**：真实 61g 下仍失败（甩飞 1.2m）→ 采用 **weld attachment proxy**（oracle grasp attachment，**非真实抓取接触物理**）。

## 结果（weld payload audit, sustained-load 窗，3 val eps）

| | no-object | +weld payload | ratio |
|---|---|---|---|
| sustained IMU hf (clean, ACT) | 0.369 | **0.259** | **0.70** |
| sustained IMU hf (noisy, ACT) | 0.810 | 0.590 | 0.73 |

- **61g payload 使搬运 IMU 高频降 ~30%**：末端加质量阻尼高频响应（物理真实）——物体让动作更平滑，非更抖。
- **action 平滑（refined vs ACT）在带载下不降 IMU**：clean 持平（ratio 1.06）、noisy 反而更差（1.32）。
- attach/release event peak 1.7–6.2（合理，单列，**非真实接触峰值**）。
- 早前 mass-insensitivity（40g=61g IMU 逐字节相同）一致：link6 IMU 由机械臂自身动力学主导。

## 战略结论

1. sim 里 link6 IMU 由**机械臂自身动力学 + 噪声层主导**；61g 物体次要且阻尼高频。
2. **action 平滑在 sim 里不传递到 IMU 平滑**（自由空间微弱、带载无效/变差）。
3. 真机 de-risk 有真实 **IMU hf +48.7%** gap（ACT 抖动）。

→ **Phase-1 sim 无法验证 IMU 平滑收益**（action→IMU 通路弱/阻尼主导，未复现真机机制）。按既定判据：**停止打磨 Stage 4 sim 接触，进 Stage 5 真机验证**。RefineNet 仍有效（动作空间 chunk 内 0.28×、轨迹保真↑），但其 IMU 收益只能在真机上证。

## Document History
| 日期 | 内容 |
|---|---|
| 2026-05-30 | Stage 4.0 B+ 模型 + 接触/weld 标定 + transfer audit；结论：sim IMU 臂主导、action→IMU 不传递 → 转 Stage 5 |
