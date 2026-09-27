# ③c R 阶段 · 第②步（被动读取）结果

> **本文件记录 2026-09-27 的三次被动读取。以最后一次 `112124` 为准。**
> 前两次分别暴露了「派生值未读源」和「派生值源有效却未升级状态」两个报告缺陷，
> 都已修；**三份原始 JSON 全部保留、不回填**。

| 次 | 记录 | 这次暴露/验证了什么 |
|---|---|---|
| 1 | `readonly_20260927T074217_549076Z.json` | 未读源反馈 ⇒ `enable_status` **未核实**；状态流背靠背采样 |
| 2 | `readonly_20260927T111949_692339Z.json` | 补读 `low_spd_info`（valid）+ 采样间隔 0.501 s；但派生值**仍误留 `unknown`** |
| **3（准）** | **`readonly_20260927T112124_159895Z.json`** | 派生值升级逻辑已修 ⇒ `enable_status` **valid** |

执行 2026-09-27。`sim/readonly_check_3c.py --can can0 --i-confirm-arm-is-idle`
（**未加** `--send-query-frames`）。退出码 0。

原始记录：`outputs/probe_3c/readonly/readonly_20260927T074217_549076Z.json`

> ⚠ **原始 JSON 按当时的分类方案写成，保留原样、不回填。**
> 复审后有两条结论订正 + 两处报告口径修正（§3、§4），
> **引用那份 JSON 时须按本文件的订正读。**

---

## 1. 程序健康：干净

| | |
|---|---|
| `fatal` | **不存在** |
| `disconnect_error` | **不存在** |
| `program_errors` | **不存在** |
| `steps_done` | `connect_readonly(piper_init=False)` → `reads + idle_check` → `disconnect` |

⚠ **退出码 0 不等于核验完成。**

## 2. 本轮静止检查通过（限定说法）

`idle_check`：时间戳 `1790494938.154 → 1790494939.154`，**正向推进 +1.0000 s**
（200 Hz 在收流），关节角逐项不变 ⇒ `confirmed_idle = True`。

> 依据是**状态流 + 约 1 秒间隔的两次关节读数**。
> 这支持**本轮**静止检查通过，**不解决运动安全依据**。

## 3. `enable_status`：从「未核实」到「有证据」

### 3.1 ⚠ 订正一（对第 1 次读取）：全 `false` **不能**证明电机未使能

`GetArmEnableStatus`（`piper_interface_v2.py`）是从**六个电机低速反馈缓存**的
`foc_status.driver_enable_status` 提取的派生值 —— 而**该缓存初始化也全是 `False`**。
**本次没有保存这些源反馈（`GetArmLowSpdInfoMsgs`）的有效性证据。**

> ⇒ 只能写「**使能状态未核实**」，**不是**「未使能」。
> ⇒ 这**也不等于**发现了电机已使能。

**已修**：读取表补入 `low_spd_info`（源反馈）；`enable_status` 标为
`source = device_feedback_derived` + `derived_from`，
**源反馈 status ≠ valid 时，派生值一律记 `unverified`**。

### 3.2 第 3 次读取：源反馈拿到了，**是真实设备数据**

`GetArmLowSpdInfoMsgs` → `status = valid`：

| | |
|---|---|
| 流 | `Hz = 40.0`，`time_stamp` 非零 |
| 六电机**全在** | `can_id` 609–614（= 0x261–0x266） |
| **真实物理量** | 电压 **23.0 V** · FOC 温度 34–35 °C · 电机温度 29–30 °C |
| 故障位 | `_foc_status_code = 0`；过热/过流/碰撞/堵转标志**全 `false`** |

**缓存默认值会是全零 —— 这些不是。** 脚本据此判定源「六电机齐全 + 含非默认物理量」，
把派生值升为 `valid`。

> ⇒ 六个 `driver_enable_status = False` **是设备自报的值**，不是缓存初始化残留。
> ⚠ **但它仍只表示「驱动器自报的使能位」，其厂商语义待核实** ——
> 该位是否等同于「电机未上电/不可驱动」，要看厂商说明。

### 3.3 ⚠ 又一个报告缺陷（第 2 次读取暴露，已修）

第 2 次读取里源反馈已 `valid`，但 `enable_status` **仍留在 `unknown`** ——
派生值逻辑只在理由里追加了「源有效」，**没把 status 升上去**。
已修：源 `valid` **且**六电机齐全 **且**含非默认物理量 ⇒ 派生值判 `valid`；
三个条件缺一即 `unverified`。e2e 回归 19 → **22 项**（三个场景各加一条断言）。

## 4. ⚠ 订正二：静止位置超文档范围 **不能**直接否定文档范围

实测姿态（0.001°）：j1 153 · **j2 −1455** · j3 623 · j4 0 · **j5 28601** · j6 0。
其中 j2、j5 落在 `piper.py::write` docstring 范围之外。

> ⚠ **文档可能约束的是「允许下发的目标」，而不是机械臂所有可能处于的位置。**
> ⇒ 只能记录「**反馈位置与该文档范围不一致，适用性待核实**」，
> **不得**据此断言文档范围错误。

（此前一版我写成「文档范围不是设备的实际许可范围」，**已撤回**。）

## 5. 读取结果（按订正后的分类）

**设备反馈** —— **4 项 `valid`**（第 3 次读取）：

| | |
|---|---|
| `arm_status_0x2A1` | 200 Hz · `_ctrl_mode 0` · `_arm_status 0` · `_mode_feed 0` · `_teach_status 0` · `_motion_status 0` · `_err_code 0` · 角度限位错误标志全 `false` |
| `joint_now` | 200 Hz · 见 §4 |
| `low_spd_info` | 40 Hz · 六电机齐全 · 真实电压/温度 · 故障位全清 |
| `enable_status` | **派生值**，源有效 ⇒ 六位 `driver_enable_status` 全 `False`（语义待核实） |

⚠ `_ctrl_mode 0` 是**当前静止态**的读数，**不是部署时的模式**，不能代替 R 阶段的模式确认。

**设备反馈** —— 8 项 `no_reply`（全是**应答式**数据）：
固件（返回 `-0x4AF`）· 两个聚合限值 · `cur_angle_limit_max_vel` · `cur_max_acc_limit` ·
`end_vel_acc_param` · `crash_protection` · `listened_ctrl_0x151`。

> **这是被动读取下的预期结果**，**不据此自动进入查询阶段**。

**SDK 本地值**（`sdk_local` / `sdk_statistic`，**不是设备反馈**）：
SDK 0.6.1 · 协议 V2 · 接口 V2 · `can_fps` 3060 · 夹爪范围 `[0, 0.07]` ·
SDK 软限（rad）j1 ±2.6179 / j2 `[0, 3.14]` / j3 `[-2.967, 0]` / j4 ±1.745 /
j5 ±1.22 / j6 ±2.09439。
⚠ 该软限默认 `start_sdk_joint_limit=False` ⇒ **不生效**，只是存着的默认值。

## 6. 查询证据：正确地全为否

`authorized=False` · `methods_called=[]` · 两个应答式接口 `reply_received=False` ·
`firmware_reply.newly_received=False` · `send_layer_evidence=null`。
**没发帧，也没有任何东西被误记成「收到了」。**

## 7. 两处报告口径修正（已实现）

### 7.1 状态流采样：始终用明确间隔并记录实际间隔

原始 JSON 里 `status_stream.advanced_forward = {False, False}`，`pre`/`post`
时间戳**逐位相同** —— 因为不发查询帧时 `pre`/`post` 是**背靠背**采的（零间隔）。

> ⚠ **背靠背的 `False` 只能表示「这两次没观察到更新」，不能解释成断流。**
> 同一次运行里 `idle_check` 用 1.0 s 间隔测到的是**确实在推进**。

**已修**：无论是否发查询，都保证 `STREAM_MIN_INTERVAL = 0.5 s`，
并记录 `sampling_interval_s`（实际间隔）。
第 3 次读取实测 **`0.5014 s`**，`advanced_forward` 两项**均为 `True`**
⇒ 证实第 1 次那个 `False` 确实只是零间隔采样的产物。

### 7.2 「数据来源」与「有效性／新鲜度」分开记录

旧版把 12 项一律记 `unknown`，混淆了三类不同的东西。现在 `source` 独立成字段：

| `source` | 含义 | 例 |
|---|---|---|
| `device_feedback` | 设备 CAN 反馈 | `arm_status` · `joint_now` · 聚合限值 |
| **`device_feedback_derived`** | **设备反馈的派生值**（须看源） | **`enable_status`** |
| `sdk_local` | SDK 本地常量 | 版本 · SDK 软限 · 夹爪范围 |
| `sdk_statistic` | SDK 本地统计量 | `can_fps` |

`status` 相应新增 `sdk_local` / `sdk_statistic`（**本地值没有新鲜度可言，不是 `unknown`**）
与 `unverified`（**派生值且源未核实**）。

## 8. R 阶段的三个问题：本次一个都没答上

被动读取**结构上答不了**：控制模式的语义 · `MotionCtrl_2` 第三参数(=100) ·
限速/加速度**约束的是下发目标的变化还是控制器内部生成的轨迹**。
前两项要厂商说明；第三项还要先把限值读出来（**需查询帧**）。

`applicability` 各项**全部留空**，`verdict.released = false`。
**③c 复放阻断保持生效；W / M / P 不动。**

## 9. 自测状态

| | |
|---|---|
| `--dry-run`（判定逻辑） | **30/30** |
| `--self-test-e2e`（主流程接线） | **22/22**（含注入 `NameError` → 退出码 5） |
