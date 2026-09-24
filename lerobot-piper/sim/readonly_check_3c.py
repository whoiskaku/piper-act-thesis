#!/usr/bin/env python3
"""③c **R 阶段专用只读核验脚本** —— 读固件/控制模式/限位/限速/加速度。

⛔⛔ 本脚本的硬约束 ⛔⛔
========================
**只允许查询。禁止使能、动作、夹爪、参数修改。**

这不是靠注释保证的 —— 用**允许清单代理**（`_ReadOnlyProxy`）在**属性访问层**拦截：
任何不在 `ALLOW` 里的方法名，**取到的瞬间就抛 `ForbiddenCall`**，根本调不到。
禁用清单（使能/动作/夹爪/配置）**另有显式断言**，被访问时抛出并打印方法名。

**不走** `make_robot` / lerobot 的 robot 包装
--------------------------------------------
现有连接流程会**使能 + 标定归位**，断开流程也可能发送动作 ⇒ **不能借它来「只读」**。
本脚本**直接**用 `piper_sdk`，并且：

- `ConnectPort(can_init=False, piper_init=False, start_thread=True)`
  ⚠ **`piper_init` 必须为 `False`** —— 默认 `True` 会自动调 `PiperInit()`，
    那会**立刻发三条查询帧**。本脚本要把「发帧」这一步**显式化、可拒绝**。
- `DisconnectPort()` —— 已读源码确认：**只置标志位、停读线程、join**，**不发任何动作**。

⚠ 一条必须说清楚的：**查询也是往总线写帧**
-------------------------------------------
`Search*` 系列会发 CAN **查询**帧（如最大加速度查询 `0x472`）。
它们**不是运动指令**，但**确实是总线写入**。
⇒ 本脚本把它们放在 `--send-query-frames` 后面，**默认不发**。
   只读到「SDK 本地缓存 / SDK 侧软限」时不需要它；
   要读**固件侧**的限速/加速度/固件版本，**必须**发查询帧。

用法
----
    # ① 离线自检（**不碰 CAN**，先审查用这个）
    python -u sim/readonly_check_3c.py --dry-run

    # ② 现场确认机械臂静止、无其他程序发指令后，再跑（仍不发查询帧）
    python -u sim/readonly_check_3c.py --i-confirm-arm-is-idle

    # ③ 需要固件侧限值时，额外显式同意发查询帧
    python -u sim/readonly_check_3c.py --i-confirm-arm-is-idle --send-query-frames

⚠ **读到数字不自动放行。** 本脚本只产出事实；
适用对象（约束的是「下发目标的变化」还是「控制器内部生成的轨迹」）
**必须结合厂商说明另行判断**，见 `reports/CHECKLIST_3c_readonly.md`。
"""
from __future__ import annotations
import argparse, json, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_3c/readonly_check.json"

# ══════════════════════════════════════════════════════════════════════
#  允许清单 / 禁用清单
# ══════════════════════════════════════════════════════════════════════
# 只读取（返回 SDK 已缓存的反馈，不发帧）
ALLOW_READ = {
    "GetPiperFirmwareVersion",       # 固件版本
    "GetArmStatus",                  # **控制模式 / 运动状态 / 示教状态**
    "GetArmEnableStatus",            # 使能状态（只读，不改）
    "GetArmJointMsgs",               # 当前关节角（确认静止用）
    "GetArmHighSpdInfoMsgs",         # 高速反馈（转速/电流）
    "GetArmLowSpdInfoMsgs",
    "GetAllMotorAngleLimitMaxSpd",   # **逐关节 位置限位 + 最大速度**
    "GetAllMotorMaxAccLimit",        # **逐关节 最大加速度**
    "GetCurrentMotorAngleLimitMaxVel",
    "GetCurrentMotorMaxAccLimit",
    "GetCurrentEndVelAndAccParam",
    "GetCrashProtectionLevelFeedback",
    "GetArmModeCtrl",                # ⚠ 名字带 Ctrl 但**是 Get**：读模式反馈
    "GetSDKJointLimitParam",         # SDK 侧软限（**不是固件侧**）
    "GetSDKGripperRangeParam",
    "GetCurrentProtocolVersion", "GetCurrentSDKVersion", "GetCurrentInterfaceVersion",
    "GetCanFps", "GetCanName", "get_connect_status", "isOk",
}
# 会往总线发**查询**帧（不是运动指令，但仍是总线写入）⇒ 需 --send-query-frames
ALLOW_QUERY = {
    "SearchPiperFirmwareVersion",
    "SearchAllMotorMaxAngleSpd",
    "SearchAllMotorMaxAccLimit",
}
# 连接/断开（DisconnectPort 已读源码确认不发动作）
ALLOW_CONN = {"ConnectPort", "DisconnectPort"}

# 显式禁用（被访问即抛，并打印方法名）—— 覆盖使能/动作/夹爪/配置四类
DENY_SUBSTR = ("Ctrl", "Enable", "Disable", "Set", "Config", "Reset", "Stop",
               "Gripper", "Move", "Motion", "Emergency", "Zero", "Calib",
               "Teach", "Drag", "Install", "Update", "Master", "Init")


class ForbiddenCall(RuntimeError):
    pass


class _ReadOnlyProxy:
    """**属性访问层**的允许清单。取不到就抛 —— 调用方**没有机会**调到禁用方法。"""

    def __init__(self, inner, allow: set[str]):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_allow", set(allow))
        object.__setattr__(self, "_log", [])

    def __getattr__(self, name):
        allow = object.__getattribute__(self, "_allow")
        if name not in allow:
            # 禁用清单命中时给出更明确的信息
            why = "**禁用类**" if any(s in name for s in DENY_SUBSTR) else "不在允许清单"
            raise ForbiddenCall(
                f"⛔ 拒绝访问 `{name}`（{why}）。R 阶段只允许：{sorted(allow)}")
        object.__getattribute__(self, "_log").append(name)
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        raise ForbiddenCall(f"⛔ 拒绝写属性 `{name}` —— R 阶段不修改任何参数")

    def calls(self):
        return list(object.__getattribute__(self, "_log"))


# ══════════════════════════════════════════════════════════════════════
def guard_selftest() -> bool:
    """离线自检：**不碰 CAN**，只验证代理确实拦得住。"""
    print("=" * 78 + "\n离线自检：允许清单代理（**不连接 CAN**）\n" + "=" * 78)

    class _Fake:
        def __getattr__(self, n):
            return lambda *a, **k: f"CALLED:{n}"

    ok = {}

    def chk(k, cond, d=""):
        ok[k] = bool(cond); print(f"  {'✓' if cond else '**✗**'} {k}{('   ' + d) if d else ''}")

    p = _ReadOnlyProxy(_Fake(), ALLOW_READ | ALLOW_CONN)
    chk("1 允许清单内的读方法可访问", p.GetArmStatus() == "CALLED:GetArmStatus")

    forbidden = ["EnablePiper", "EnableArm", "DisableArm", "JointCtrl", "GripperCtrl",
                 "MotionCtrl_1", "MotionCtrl_2", "ModeCtrl", "EmergencyStop",
                 "JointConfig", "JointMaxAccConfig", "MotorMaxSpdSet",
                 "MotorAngleLimitMaxSpdSet", "SetSDKJointLimitParam", "ResetPiper",
                 "EndPoseCtrl", "JointMitCtrl", "MasterSlaveConfig", "PiperInit"]
    blocked = []
    for n in forbidden:
        try:
            getattr(p, n)
            blocked.append(f"**{n} 未被拦截**")
        except ForbiddenCall:
            pass
    chk(f"2 {len(forbidden)} 个使能/动作/夹爪/配置方法全部被拦截",
        not blocked, "; ".join(blocked))

    try:
        p.SearchAllMotorMaxAccLimit(); q_blocked = False
    except ForbiddenCall:
        q_blocked = True
    chk("3 查询帧方法**默认也被拦截**（须 --send-query-frames 才加入清单）", q_blocked)

    try:
        p.foo = 1; w_blocked = False
    except ForbiddenCall:
        w_blocked = True
    chk("4 写属性被拒绝", w_blocked)

    p2 = _ReadOnlyProxy(_Fake(), ALLOW_READ | ALLOW_CONN | ALLOW_QUERY)
    chk("5 显式开启后查询帧方法才可访问",
        p2.SearchAllMotorMaxAccLimit() == "CALLED:SearchAllMotorMaxAccLimit")
    try:
        p2.JointCtrl; still = False
    except ForbiddenCall:
        still = True
    chk("6 即便开了查询帧，动作方法**仍然**被拦截", still)

    chk("7 不导入 lerobot 的 robot 包装（连接流程会使能+归位）",
        "lerobot.common.robot_devices.robots" not in sys.modules)
    print(f"\n  ⇒ 自检 {sum(ok.values())}/{len(ok)} 通过")
    return all(ok.values())


def can_state():
    """`ip -details link show can0` —— **只看不改**。"""
    try:
        r = subprocess.run(["ip", "-details", "link", "show", "can0"],
                           capture_output=True, text=True, timeout=5)
        return {"rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def dump(o):
    """把 SDK 的反馈对象转成可落盘的结构（尽量保留原始字段名）。"""
    if o is None or isinstance(o, (int, float, str, bool)):
        return o
    for attr in ("__dict__",):
        d = getattr(o, attr, None)
        if isinstance(d, dict) and d:
            return {k: dump(v) for k, v in d.items() if not k.startswith("_abc")}
    if hasattr(o, "__slots__"):
        return {k: dump(getattr(o, k, None)) for k in o.__slots__}
    return repr(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="离线自检，**不碰 CAN**")
    ap.add_argument("--i-confirm-arm-is-idle", action="store_true",
                    help="现场已确认：机械臂静止、无其他控制程序发送动作、**未切换电机状态**")
    ap.add_argument("--send-query-frames", action="store_true",
                    help="⚠ 允许发 CAN **查询**帧（非运动指令，但仍是总线写入）")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--settle", type=float, default=1.5, help="发查询后等反馈的秒数")
    a = ap.parse_args()

    if not guard_selftest():
        print("\n**✗ 自检未过 —— 不连接设备**")
        return 1
    if a.dry_run:
        print("\n  （--dry-run：到此为止，**未连接 CAN**）")
        return 0
    if not a.i_confirm_arm_is_idle:
        print("\n⛔ 未给 `--i-confirm-arm-is-idle` ⇒ **不连接设备**。")
        print("   请先在现场确认：机械臂静止 · 没有其他控制程序在发送动作 · "
              "**不要为这一步切换电机状态**。")
        return 2

    rec = {"stage": "R (read-only)", "utc": datetime.now(timezone.utc).isoformat(),
           "host": platform.node(), "python": platform.python_version(),
           "sent_query_frames": bool(a.send_query_frames),
           "can_link": can_state()}
    print("\n" + "=" * 78 + f"\nR 阶段读取（can={a.can}）\n" + "=" * 78)

    from piper_sdk import C_PiperInterface_V2
    allow = ALLOW_READ | ALLOW_CONN | (ALLOW_QUERY if a.send_query_frames else set())
    raw = C_PiperInterface_V2(a.can)
    p = _ReadOnlyProxy(raw, allow)
    try:
        # ⚠ piper_init=False：**不让它自动发那三条查询帧**
        p.ConnectPort(can_init=False, piper_init=False, start_thread=True)
        time.sleep(0.5)
        rec["connected"] = bool(p.get_connect_status())
        print(f"  连接读线程：{rec['connected']}（**piper_init=False，未自动发帧**）")

        if a.send_query_frames:
            print("  ⚠ 发送 CAN **查询**帧（固件版本 / 角度限位+最大速度 / 最大加速度）…")
            p.SearchPiperFirmwareVersion()
            p.SearchAllMotorMaxAngleSpd()
            p.SearchAllMotorMaxAccLimit()
            time.sleep(a.settle)
        else:
            print("  · **未发查询帧**（无 --send-query-frames）⇒ "
                  "固件侧限值多半读不到，只能拿到 SDK 侧的值")

        reads = {
            "firmware_version": "GetPiperFirmwareVersion",
            "arm_status": "GetArmStatus",
            "enable_status": "GetArmEnableStatus",
            "mode_ctrl_feedback": "GetArmModeCtrl",
            "joint_now": "GetArmJointMsgs",
            "angle_limit_max_spd_all": "GetAllMotorAngleLimitMaxSpd",
            "max_acc_limit_all": "GetAllMotorMaxAccLimit",
            "cur_angle_limit_max_vel": "GetCurrentMotorAngleLimitMaxVel",
            "cur_max_acc_limit": "GetCurrentMotorMaxAccLimit",
            "end_vel_acc_param": "GetCurrentEndVelAndAccParam",
            "crash_protection": "GetCrashProtectionLevelFeedback",
            "sdk_joint_limit": "GetSDKJointLimitParam",
            "sdk_gripper_range": "GetSDKGripperRangeParam",
            "protocol_version": "GetCurrentProtocolVersion",
            "sdk_version": "GetCurrentSDKVersion",
            "interface_version": "GetCurrentInterfaceVersion",
            "can_fps": "GetCanFps",
        }
        rec["reads"] = {}
        for key, meth in reads.items():
            try:
                rec["reads"][key] = {"method": meth, "value": dump(getattr(p, meth)())}
                print(f"  ✓ {key:26} ← {meth}")
            except ForbiddenCall as e:
                rec["reads"][key] = {"method": meth, "error": f"FORBIDDEN: {e}"}
                print(f"  ⛔ {key:26} {e}")
            except Exception as e:
                rec["reads"][key] = {"method": meth, "error": f"{type(e).__name__}: {e}"}
                print(f"  ⚠ {key:26} 读取失败: {type(e).__name__}: {e}")

        # 静止确认（两次关节读数之差）
        try:
            j0 = dump(p.GetArmJointMsgs()); time.sleep(1.0)
            j1 = dump(p.GetArmJointMsgs())
            rec["idle_check"] = {"t0": j0, "t1": j1,
                                 "note": "两次读数应一致/极接近 ⇒ 臂静止"}
            print("  ✓ 静止确认：已记录相隔 1 s 的两次关节读数")
        except Exception as e:
            rec["idle_check"] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            p.DisconnectPort()          # 源码已核：只停读线程，不发动作
            print("  ✓ 已断开（`DisconnectPort` 只停读线程，**不发动作**）")
        except Exception as e:
            print(f"  ⚠ 断开异常: {type(e).__name__}: {e}")

    rec["calls_made"] = p.calls()
    rec["applicability"] = {
        "__instructions": "**下列各项必须结合厂商说明逐条填写，任何一项未确认就保留 null，"
                          "不以推测填充。**",
        "control_mode": None,
        "jointctrl_input_semantics": None,      # 位置目标 or 增量？单位是否 0.001°？
        "motionctrl2_third_param_100": None,    # 速度百分比？量程？上限？
        "speed_limit_applies_to": None,         # 下发目标变化 / 控制器内部轨迹
        "acc_limit_applies_to": None,           # 同上
        "units": None,
        "is_factory_default_or_modified": None,
        "firmware_doc_ref": None,
    }
    rec["verdict"] = {
        "released": False,
        "note": "**读到数字不自动放行。** 本脚本只产出事实；"
                "适用对象须结合厂商说明另行判断。复放阻断**保持生效**。",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    print(f"\n→ {OUT}")
    print("\n" + "=" * 78)
    print("⚠ **读到数字不自动放行。**")
    print("   `applicability` 各项**尚未填写** —— 必须结合厂商说明逐条确认，")
    print("   尤其是「限速/加速度约束的是**下发目标的变化**还是"
          "**控制器内部生成的轨迹**」。")
    print("   ③c 复放阻断**保持生效**；是否解除**另行决定**。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
