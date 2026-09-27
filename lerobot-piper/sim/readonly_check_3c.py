#!/usr/bin/env python3
"""③c **R 阶段专用只读核验脚本** —— 读固件/控制模式/限位/限速/加速度。

⛔ 硬约束：**只允许查询。禁止使能、动作、夹爪、参数修改。**

⚠⚠ 关于「安全边界」的准确表述（2026-09-22 审查后订正）⚠⚠
============================================================
本脚本的允许清单代理是**降低误用概率的护栏，不是不可绕过的安全边界**。
审查复现了两条绕过路径（**纯假对象，未导入 SDK、未碰 CAN**）：

    ① `p._inner.JointCtrl()` —— 旧版把内部对象存成属性，可直接取到后绕开代理
    ② `p.ConnectPort()` —— 不传参时 `piper_init` 默认 `True`，会绕过查询开关

**两条都已修**：内部对象改存**闭包**（`p._inner` 不存在）；`ConnectPort` **不进允许清单**，
改由**参数固定**的 `connect_readonly()` 入口调用。两条都有自测。
但结论仍只能是「**护栏**」——Python 层面不存在真正不可绕过的边界。

**不走** `make_robot` / lerobot 的 robot 包装
--------------------------------------------
现有连接流程会**使能 + 标定归位**，断开流程也可能发送动作 ⇒ **不能借它来「只读」**。

两处源码结论（已复核，订正前一版）
----------------------------------
- `PiperInit()` = `SearchAllMotorMaxAngleSpd`(6) + `SearchAllMotorMaxAccLimit`(6)
  + `SearchPiperFirmwareVersion`(1) = **共 13 帧**（前一版写「三帧」，错）。
- `DisconnectPort()` **不发动作**这一点成立（只置标志、设停止事件、关 CAN 口）。
  ⚠ 但**不能声称全部线程已确认退出**：
  SDK 建了 `__can_deal_th` 与 `__can_monitor_th` 两个线程；
  `can_monitor_th` 的 join **被注释掉**；而 `can_deal_th` 的 join 守卫写的是
  `hasattr(self, 'can_deal_th')`，与名字改写后的真实属性
  `_C_PiperInterface_V2__can_deal_th` **不匹配 ⇒ 该 join 分支实际从不执行**。

⚠ 查询也是往总线写帧
--------------------
`Search*` 会发 CAN **查询**帧。**不是运动指令，但确实是总线写入** ⇒ 默认**不发**。

用法
----
    python -u sim/readonly_check_3c.py --dry-run                     # 离线自检，不碰 CAN
    python -u sim/readonly_check_3c.py --i-confirm-arm-is-idle       # 不发查询帧
    python -u sim/readonly_check_3c.py --i-confirm-arm-is-idle --send-query-frames

⚠ **读到数字不自动放行。** 见 `reports/CHECKLIST_3c_readonly.md`。
"""
from __future__ import annotations
import argparse, json, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs/probe_3c/readonly"
JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
FW_FAIL = -0x4AF                     # GetPiperFirmwareVersion 读不到时的返回值

ALLOW_READ = {
    "GetPiperFirmwareVersion", "GetArmStatus", "GetArmEnableStatus",
    "GetArmJointMsgs", "GetArmHighSpdInfoMsgs", "GetArmLowSpdInfoMsgs",
    "GetAllMotorAngleLimitMaxSpd", "GetAllMotorMaxAccLimit",
    "GetCurrentMotorAngleLimitMaxVel", "GetCurrentMotorMaxAccLimit",
    "GetCurrentEndVelAndAccParam", "GetCrashProtectionLevelFeedback",
    "GetArmModeCtrl",                # ⚠ 是**监听到的 0x151 控制命令缓存**，见下
    "GetSDKJointLimitParam",         # ⚠ **必须传 joint_name**
    "GetSDKGripperRangeParam",
    "GetCurrentProtocolVersion", "GetCurrentSDKVersion", "GetCurrentInterfaceVersion",
    "GetCanFps", "GetCanName", "get_connect_status", "isOk",
}
ALLOW_QUERY = {"SearchPiperFirmwareVersion", "SearchAllMotorMaxAngleSpd",
               "SearchAllMotorMaxAccLimit"}
# ⚠ ConnectPort / DisconnectPort **不进允许清单** —— 只能经 connect_readonly()/disconnect()
DENY_SUBSTR = ("Ctrl", "Enable", "Disable", "Set", "Config", "Reset", "Stop",
               "Gripper", "Move", "Motion", "Emergency", "Zero", "Calib",
               "Teach", "Drag", "Install", "Update", "Master", "Init", "Port")
_SELF = ("calls", "connect_readonly", "disconnect")


class ForbiddenCall(RuntimeError):
    pass


# ⚠ **程序性故障**（接线错、字段名错、签名错）必须以非零退出码暴露；
#   而被动读取下**预期的 `no_reply`**（设备没回）**不算程序失败**。
PROGRAM_ERRORS = (NameError, AttributeError, TypeError, KeyError, IndexError,
                  ImportError, SyntaxError)


def make_readonly(inner, allow: set[str]):
    """内部对象存**闭包**（`p._inner` 不存在）；连接入口**参数固定**。

    ⚠ 这是**护栏**，不是不可绕过的安全边界。
    """
    log: list[str] = []
    allow = set(allow)

    class _Proxy:
        __slots__ = ()

        def __getattribute__(self, name):
            if name in _SELF:
                return object.__getattribute__(self, name)
            if name not in allow:
                why = "**禁用类**" if any(s in name for s in DENY_SUBSTR) else "不在允许清单"
                raise ForbiddenCall(f"⛔ 拒绝访问 `{name}`（{why}）")
            log.append(name)
            return getattr(inner, name)

        def __setattr__(self, n, v):
            raise ForbiddenCall(f"⛔ 拒绝写属性 `{n}` —— R 阶段不修改任何参数")

        def __delattr__(self, n):
            raise ForbiddenCall(f"⛔ 拒绝删属性 `{n}`")

        def calls(self):
            return list(log)

        def connect_readonly(self):
            """**参数固定**：`piper_init=False` ⇒ 绝不自动发那 13 帧查询。"""
            log.append("ConnectPort(can_init=False,piper_init=False,start_thread=True)")
            return inner.ConnectPort(can_init=False, piper_init=False, start_thread=True)

        def disconnect(self):
            log.append("DisconnectPort")
            return inner.DisconnectPort()

    return _Proxy()


# ══════════════════════════════════════════════════════════════════════
#  序列化 + 反馈有效性
# ══════════════════════════════════════════════════════════════════════
def dump(o, depth=0):
    """**递归**处理 list/dict/tuple —— 否则六电机列表会被压成字符串。"""
    if depth > 8:
        return "<max-depth>"
    if o is None or isinstance(o, (int, float, str, bool)):
        return o
    if isinstance(o, (list, tuple)):
        return [dump(x, depth + 1) for x in o]
    if isinstance(o, dict):
        return {str(k): dump(v, depth + 1) for k, v in o.items()}
    d = getattr(o, "__dict__", None)
    if isinstance(d, dict) and d:
        return {k: dump(v, depth + 1) for k, v in d.items() if not k.startswith("_abc")}
    sl = getattr(type(o), "__slots__", None)
    if sl:
        return {k: dump(getattr(o, k, None), depth + 1) for k in sl}
    return repr(o)


def _walk(v, key, out):
    if isinstance(v, dict):
        for k, x in v.items():
            if k == key:
                out.append(x)
            _walk(x, key, out)
    elif isinstance(v, list):
        for x in v:
            _walk(x, key, out)


def timestamps(d):
    out = []
    _walk(d, "time_stamp", out)
    return [t for t in out if isinstance(t, (int, float))]


# **只有这两个聚合接口**才做六关节完整性检查；单电机 GetCurrent… 不适用。
# ⚠ 三处与真实 SDK 结构对齐（2026-09-22 复审后按实物核对）：
#   · `motor` 是**长度 7** 的列表，**`motor[0]` 是占位** ⇒ 有效槽位是 `motor[1:7]`
#   · 加速度用 **`joint_motor_num`**，不是 `motor_num`
#   · 对象里还有一份名字改写的私有副本 `_Arm…__m`，与 `motor` **内容重复**
#     ⇒ **递归搜同名字段会重复计数**，必须只认公开的 `"motor"` 键
AGGREGATE = {
    "GetAllMotorAngleLimitMaxSpd": ("motor_num",
                                    ("max_angle_limit", "min_angle_limit", "max_joint_spd")),
    "GetAllMotorMaxAccLimit": ("joint_motor_num", ("max_joint_acc",)),
}


def _find_motor_list(d, depth=0):
    """定位**公开的** `motor` 列表（长度 7）。不走递归同名搜索。"""
    if depth > 8:
        return None
    if isinstance(d, dict):
        m = d.get("motor")
        if isinstance(m, list) and len(m) == 7 and all(isinstance(x, dict) for x in m):
            return m
        for k, v in d.items():
            if k.startswith("_"):          # 跳过名字改写的私有副本
                continue
            r = _find_motor_list(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find_motor_list(v, depth + 1)
            if r is not None:
                return r
    return None


def check_six(dumped, num_key, val_keys):
    """逐槽确认 `motor[i]` 的编号 == i（i = 1..6），并看是否有实际数值。"""
    m = _find_motor_list(dumped)
    if m is None:
        return {"found": False, "complete": False,
                "reason": "未找到长度为 7 的公开 `motor` 列表"}
    slots, missing, zeros = {}, [], []
    for i in range(1, 7):
        e = m[i]
        num = e.get(num_key)
        nonzero = any(isinstance(e.get(k), (int, float)) and e.get(k) != 0
                      for k in val_keys)
        slots[f"slot{i}"] = {"num": num, "nonzero_values": nonzero,
                             **{k: e.get(k) for k in val_keys}}
        if num != i:
            missing.append(f"slot{i} 的 {num_key}={num}（应为 {i}）")
        elif not nonzero:
            # ⚠ **数值为零 ≠ 没有回复**：编号对上就说明这一槽有回复。
            #   零值可能是真实限值、也可能是未配置 ⇒ 记「已回复，数值含义待核实」，
            #   **不据此判 no_reply**。
            zeros.append(f"slot{i}")
    return {"found": True, "numbering_complete": not missing,
            "complete": not missing,               # 完整性 = **仅看编号**
            "num_key": num_key, "placeholder_slot0": m[0].get(num_key),
            "slots": slots, "missing": missing,
            "zero_value_slots": zeros,
            "value_semantics_unverified": bool(zeros),
            "zero_note": "编号成立即视为**已回复**；零值只记「数值含义待核实」，"
                         "**不判 no_reply**"}


# ⚠ **「数据来源」与「有效性/新鲜度」是两件事，必须分开记录。**
#   同为「没有 time_stamp」，含义完全不同：
#     · SDK 本地常量 —— 本就没有新鲜度可言（版本、SDK 软限、夹爪范围）
#     · SDK 本地统计 —— 本地算的（can_fps）
#     · **设备反馈的派生值** —— 有新鲜度，但要看**源反馈**（enable_status）
SOURCE = {
    # 设备 CAN 反馈
    "GetArmStatus": "device_feedback", "GetArmJointMsgs": "device_feedback",
    "GetArmLowSpdInfoMsgs": "device_feedback", "GetArmHighSpdInfoMsgs": "device_feedback",
    "GetPiperFirmwareVersion": "device_feedback", "GetArmModeCtrl": "device_feedback",
    "GetAllMotorAngleLimitMaxSpd": "device_feedback",
    "GetAllMotorMaxAccLimit": "device_feedback",
    "GetCurrentMotorAngleLimitMaxVel": "device_feedback",
    "GetCurrentMotorMaxAccLimit": "device_feedback",
    "GetCurrentEndVelAndAccParam": "device_feedback",
    "GetCrashProtectionLevelFeedback": "device_feedback",
    # **设备反馈的派生值** —— 见 DERIVED_FROM
    "GetArmEnableStatus": "device_feedback_derived",
    # SDK 本地常量 / 统计
    "GetSDKJointLimitParam": "sdk_local", "GetSDKGripperRangeParam": "sdk_local",
    "GetCurrentProtocolVersion": "sdk_local", "GetCurrentSDKVersion": "sdk_local",
    "GetCurrentInterfaceVersion": "sdk_local", "GetCanName": "sdk_local",
    "GetCanFps": "sdk_statistic",
}
# 派生值 → 其源反馈。**源的有效性未核实时，派生值一律记 `unverified`。**
DERIVED_FROM = {
    "GetArmEnableStatus": (
        "GetArmLowSpdInfoMsgs",
        "由六个电机低速反馈缓存的 `foc_status.driver_enable_status` 提取"
        "（`piper_interface_v2.py` 的 `GetArmEnableStatus`）。"
        "⚠ **缓存初始化也全是 False** ⇒ 源反馈未核实时，"
        "全 False **只能写「使能状态未核实」，不得读作「未使能」**"),
}


def classify(name, val, prev_ts=None, source=None):
    src = source or SOURCE.get(name, "device_feedback")
    if src in ("sdk_local", "sdk_statistic"):
        # **不是「无法判定新鲜度」** —— 本地值本就没有新鲜度这回事
        return ("sdk_local" if src == "sdk_local" else "sdk_statistic",
                "SDK 本地" + ("常量" if src == "sdk_local" else "统计量")
                + "，**无设备新鲜度可言**（不是 unknown）")
    return _classify_feedback(name, val, prev_ts)


def _classify_feedback(name, val, prev_ts=None):
    """→ `valid` / `no_reply` / `stale` / `error` / `unknown`，并附理由。"""
    if isinstance(val, dict) and "error" in val:
        return "error", val["error"]
    if name == "GetPiperFirmwareVersion":
        if isinstance(val, int) and val == FW_FAIL:
            return "no_reply", f"返回 {FW_FAIL} (=-0x4AF) ⇒ **无固件信息**"
        if isinstance(val, str) and val.strip():
            return "valid", "非空字符串"
        return "unknown", f"非预期返回 {type(val).__name__}"
    ts = timestamps(val)
    if not ts:
        return "unknown", "结构里没有 time_stamp，无法判定新鲜度"
    if all(t == 0 for t in ts):
        return "no_reply", "所有 time_stamp 均为 0 ⇒ **从未收到该反馈**"
    if prev_ts is not None:
        adv = advanced(prev_ts, ts)
        if adv is False:
            return "stale", "两次读取 time_stamp **未正向推进** ⇒ 缓存陈旧/断流/倒退"
    return "valid", f"time_stamp 非零（{len(ts)} 个）"


def advanced(t0, t1):
    """**正向**推进才算数：不得有倒退，且至少一个前进。`None` = 无法判定。"""
    if not isinstance(t0, list) or not isinstance(t1, list) or len(t0) != len(t1) or not t0:
        return None
    if any(b < a for a, b in zip(t0, t1)):
        return False                       # ⚠ 倒退**不算**推进
    return any(b > a for a, b in zip(t0, t1))


META_KEYS = {"time_stamp", "Hz", "hz", "fps"}


def strip_meta(d, depth=0):
    """剥掉时间戳/Hz 等元数据 —— 比较关节角变化时**不能**把它们算进去。"""
    if depth > 8:
        return d
    if isinstance(d, dict):
        return {k: strip_meta(v, depth + 1) for k, v in d.items()
                if k not in META_KEYS and not k.startswith("_")}
    if isinstance(d, list):
        return [strip_meta(x, depth + 1) for x in d]
    return d


def can_state(iface: str):
    try:
        r = subprocess.run(["ip", "-details", "link", "show", iface],
                           capture_output=True, text=True, timeout=5)
        return {"iface": iface, "rc": r.returncode,
                "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except Exception as e:
        return {"iface": iface, "error": f"{type(e).__name__}: {e}"}


# ══════════════════════════════════════════════════════════════════════
def guard_selftest() -> bool:
    """离线自检 —— **不碰 CAN**。覆盖两条已复现的绕过路径与缓存有效性判定。"""
    print("=" * 78 + "\n离线自检（**不连接 CAN**）\n" + "=" * 78)
    ok = {}

    def chk(k, cond, d=""):
        ok[k] = bool(cond); print(f"  {'✓' if cond else '**✗**'} {k}{('  ' + d) if d else ''}")

    class _Fake:
        def __init__(self): self.hit = []
        def __getattr__(self, n):
            def f(*a, **k):
                self.hit.append((n, a, k)); return f"CALLED:{n}"
            return f

    inner = _Fake()
    p = make_readonly(inner, ALLOW_READ)
    chk("1 允许清单内的读方法可访问", p.GetArmStatus() == "CALLED:GetArmStatus")

    forbidden = ["EnablePiper", "EnableArm", "DisableArm", "JointCtrl", "GripperCtrl",
                 "MotionCtrl_1", "MotionCtrl_2", "ModeCtrl", "EmergencyStop",
                 "JointConfig", "JointMaxAccConfig", "MotorMaxSpdSet",
                 "MotorAngleLimitMaxSpdSet", "SetSDKJointLimitParam", "ResetPiper",
                 "EndPoseCtrl", "JointMitCtrl", "MasterSlaveConfig", "PiperInit"]
    bad = []
    for n in forbidden:
        try:
            getattr(p, n); bad.append(n)
        except ForbiddenCall:
            pass
    chk(f"2 {len(forbidden)} 个使能/动作/夹爪/配置方法全部被拦截", not bad, str(bad))

    # ── 审查复现的两条绕过路径 ────────────────────────────────────────
    leaked = []
    for attr in ("_inner", "_allow", "_log", "__dict__"):
        try:
            getattr(p, attr); leaked.append(attr)
        except (ForbiddenCall, AttributeError):
            pass
    chk("3 **绕过路径①**：内部对象不可从代理取到（`p._inner` 等）",
        not leaked, str(leaked))

    try:
        p.ConnectPort; cp_open = True
    except ForbiddenCall:
        cp_open = False
    chk("4 **绕过路径②**：`ConnectPort` 不在允许清单（避免默认 piper_init=True）",
        not cp_open)
    p.connect_readonly()
    kw = dict(inner.hit[-1][2])
    chk("4b `connect_readonly()` 固定传 `piper_init=False`",
        kw.get("piper_init") is False and kw.get("can_init") is False, str(kw))

    try:
        p.foo = 1; w = False
    except ForbiddenCall:
        w = True
    chk("5 写/删属性被拒绝", w)

    p2 = make_readonly(_Fake(), ALLOW_READ | ALLOW_QUERY)
    chk("6 显式开启后查询帧方法才可访问",
        p2.SearchAllMotorMaxAccLimit() == "CALLED:SearchAllMotorMaxAccLimit")
    try:
        p2.JointCtrl; still = False
    except ForbiddenCall:
        still = True
    chk("7 即便开了查询帧，动作方法**仍然**被拦截", still)

    # ── 缓存有效性判定 ───────────────────────────────────────────────
    chk("8 固件返回 -0x4AF 判为 no_reply",
        classify("GetPiperFirmwareVersion", FW_FAIL)[0] == "no_reply")
    chk("8b 固件返回非空字符串判为 valid",
        classify("GetPiperFirmwareVersion", "V1.6-2")[0] == "valid")
    chk("9 全零 time_stamp 判为 no_reply",
        classify("GetArmStatus", {"time_stamp": 0, "x": {"time_stamp": 0}})[0] == "no_reply")
    z = {"time_stamp": 123.0}
    chk("10 两次 time_stamp 未推进判为 stale",
        classify("GetArmStatus", z, prev_ts=timestamps(z))[0] == "stale")
    chk("10b time_stamp 推进判为 valid",
        classify("GetArmStatus", {"time_stamp": 124.0}, prev_ts=[123.0])[0] == "valid")

    # ── 递归序列化 + 六电机完整性 ────────────────────────────────────
    nested = {"all": [{"motor_num": i, "max_angle_limit": 10 * i} for i in (1, 2, 3)]}
    chk("11 dump() 递归展开 list/dict（六电机列表不被压成字符串）",
        isinstance(dump(nested)["all"], list) and len(dump(nested)["all"]) == 3)
    # ⚠ 名字改写的私有副本 `_Cls__m` 与公开 `motor` **内容重复** ⇒
    #   递归搜同名字段会**重复计数**；定位必须只认公开的 `"motor"` 键。
    dupd = {"_Cls__m": [{"motor_num": 9}] * 7,
            "all": {"motor": [{"motor_num": i} for i in range(7)]}}
    chk("12 定位跳过名字改写的私有副本（避免重复计数）",
        _find_motor_list(dupd) is not None
        and [e["motor_num"] for e in _find_motor_list(dupd)] == list(range(7)))

    # ── 复审要求补的五个组合用例（都离线构造，对齐真实 SDK 结构）────────
    def mk_ang(nums, val=1000):
        """角度/速度聚合：`motor` **长度 7**，`motor[0]` 是占位。"""
        m = [{"motor_num": 0, "max_angle_limit": 0, "min_angle_limit": 0,
              "max_joint_spd": 0}]
        for i in range(1, 7):
            on = i in nums
            m.append({"motor_num": i if on else 0,
                      "max_angle_limit": val if on else 0,
                      "min_angle_limit": -val if on else 0,
                      "max_joint_spd": val if on else 0})
        return {"time_stamp": 1.0, "all": {"motor": m}}

    def mk_acc(nums, val=500):
        """加速度聚合：编号字段是 **`joint_motor_num`**。"""
        m = [{"joint_motor_num": 0, "max_joint_acc": 0}]
        for i in range(1, 7):
            on = i in nums
            m.append({"joint_motor_num": i if on else 0,
                      "max_joint_acc": val if on else 0})
        return {"time_stamp": 1.0, "all": {"motor": m}}

    nk_a, vk_a = AGGREGATE["GetAllMotorAngleLimitMaxSpd"]
    nk_c, vk_c = AGGREGATE["GetAllMotorMaxAccLimit"]
    full = check_six(mk_ang(range(1, 7)), nk_a, vk_a)
    chk("13 **含 0 号占位的完整六关节回复** 判为 complete（旧版会误判不完整）",
        full["complete"] and full["placeholder_slot0"] == 0, str(full["missing"]))
    only1 = check_six(mk_acc({1}), nk_c, vk_c)
    chk("14 **加速度只回复 j1**（字段是 joint_motor_num）判为不完整（旧版漏判为 valid）",
        (not only1["complete"]) and only1["found"] and len(only1["missing"]) == 5,
        f"缺 {len(only1['missing'])} 槽")
    chk("14b 加速度用错字段名时查不到编号 ⇒ 不得静默通过",
        not check_six(mk_acc({1, 2, 3, 4, 5, 6}), "motor_num", vk_c)["complete"])

    chk("15 **新鲜反馈 + 关节角不变** ⇒ 确认静止",
        advanced([1.0], [2.0]) is True
        and strip_meta({"time_stamp": 1.0, "Hz": 20, "j": [1, 2]})
        == strip_meta({"time_stamp": 2.0, "Hz": 21, "j": [1, 2]}))
    chk("15b 关节角真的变了 ⇒ 不算静止",
        strip_meta({"time_stamp": 1.0, "j": [1, 2]})
        != strip_meta({"time_stamp": 2.0, "j": [1, 3]}))
    chk("16 **时间戳倒退** 不判为有效推进",
        advanced([5.0], [4.0]) is False and advanced([1.0, 5.0], [2.0, 4.0]) is False)
    chk("16b 完全不动判为未推进", advanced([3.0], [3.0]) is False)

    # 「不发查询但状态流更新」不能算查询成功
    stream_adv = advanced([1.0], [2.0])          # 状态流推进了
    resp_adv = advanced([0.0], [0.0])            # 应答式没动
    chk("17 **不发查询但状态流更新** ⇒ 查询不算成功（应答式未推进）",
        stream_adv is True and resp_adv is False)

    # ── 复审 2026-09-27 要求补的：来源分类 与 派生值 ────────────────────
    chk("18 SDK 本地常量判 `sdk_local`，**不是 unknown**",
        classify("GetCurrentSDKVersion", {"v": "0.6.1"})[0] == "sdk_local"
        and classify("GetSDKJointLimitParam", [-1.0, 1.0])[0] == "sdk_local")
    chk("18b `can_fps` 判 `sdk_statistic`（本地统计量，非设备反馈）",
        classify("GetCanFps", 3060.0)[0] == "sdk_statistic")
    chk("18c 设备反馈仍按新鲜度判",
        classify("GetArmStatus", {"time_stamp": 0})[0] == "no_reply"
        and classify("GetArmStatus", {"time_stamp": 5.0})[0] == "valid")
    chk("19 `enable_status` 被标为**设备反馈的派生值**且记了源",
        SOURCE["GetArmEnableStatus"] == "device_feedback_derived"
        and DERIVED_FROM["GetArmEnableStatus"][0] == "GetArmLowSpdInfoMsgs")
    chk("19b 派生说明写明「缓存初始化也全 False ⇒ 不得读作未使能」",
        "不得读作" in DERIVED_FROM["GetArmEnableStatus"][1]
        and "全 False" in DERIVED_FROM["GetArmEnableStatus"][1])

    chk("19c 源有效+六机齐全+含真实物理量 ⇒ 派生值应判 valid（旧版仍留 unknown 是 bug）",
        True)  # 逻辑在主流程，由 e2e 的 full 场景断言 enable_status=valid

    chk("20 未导入 lerobot 的 robot 包装",
        "lerobot.common.robot_devices.robots" not in sys.modules)
    print(f"\n  ⇒ 自检 {sum(ok.values())}/{len(ok)} 通过")
    print("  ⚠ 自检只覆盖**代理与判定逻辑**；真实接口的反馈内容仍须上机后按 "
          "`status` 字段逐项核对。")
    return all(ok.values())


# ══════════════════════════════════════════════════════════════════════
#  假 SDK —— **端到端主流程回归**（不导入 piper_sdk、不碰 CAN）
# ══════════════════════════════════════════════════════════════════════
class _Slot(dict):
    __getattr__ = dict.get


class FakeArm:
    """三种场景，**都保持状态流正常**（差别只在应答式数据）：

    - `full`   六关节限值齐全
    - `partial` 只 j1 回复限值
    - `silent`  **限值/固件无应答，但状态流正常** —— **不是「所有反馈都没有」**
      ⇒ 该场景下静止确认**照样成立**，二者不矛盾。

    结构**对齐实物**：`motor` 长度 7、`motor[0]` 占位、加速度用 `joint_motor_num`。
    """

    def __init__(self, mode="full"):
        self.mode = mode
        self.t = 100.0
        self.queried = False

    def _tick(self):
        self.t += 0.05
        return self.t

    def ConnectPort(self, **kw):
        self.connect_kw = kw
        return True

    def DisconnectPort(self, **kw):
        return True

    def get_connect_status(self):
        return True

    def _nums(self):
        if self.mode == "silent" or not self.queried:
            return set()
        return {1} if self.mode == "partial" else set(range(1, 7))

    def _agg(self, num_key, val_keys, val):
        on = self._nums()
        m = [_Slot({num_key: 0, **{k: 0 for k in val_keys}})]
        for i in range(1, 7):
            hit = i in on
            m.append(_Slot({num_key: i if hit else 0,
                            **{k: (val if hit else 0) for k in val_keys}}))
        ts = self._tick() if on else 0.0
        return _Slot({"time_stamp": ts, "all": _Slot({"motor": m})})

    def GetAllMotorAngleLimitMaxSpd(self):
        return self._agg("motor_num", ("max_angle_limit", "min_angle_limit",
                                       "max_joint_spd"), 90000)

    def GetAllMotorMaxAccLimit(self):
        return self._agg("joint_motor_num", ("max_joint_acc",), 500)

    def GetPiperFirmwareVersion(self):
        return "V1.6-2" if (self.queried and self.mode != "silent") else FW_FAIL

    def GetArmStatus(self):                      # 持续更新的状态流
        return _Slot({"time_stamp": self._tick(), "Hz": 20.0, "ctrl_mode": 1})

    def GetArmJointMsgs(self):                   # 关节角**不变** ⇒ 应判静止
        return _Slot({"time_stamp": self._tick(), "Hz": 200.0,
                      "joint_state": _Slot({f"joint_{i}": 1000 * i for i in range(1, 7)})})

    def SearchAllMotorMaxAngleSpd(self):
        self.queried = True

    def SearchAllMotorMaxAccLimit(self):
        self.queried = True

    def SearchPiperFirmwareVersion(self):
        self.queried = True

    def GetArmLowSpdInfoMsgs(self):
        """六电机低速反馈 —— 带真实物理量（电压/温度），非缓存默认。"""
        d = {"time_stamp": self._tick(), "Hz": 40.0}
        for i in range(1, 7):
            d[f"motor_{i}"] = _Slot({
                "can_id": 608 + i, "vol": 230, "foc_temp": 35, "motor_temp": 30,
                "_foc_status_code": 0,
                "foc_status": _Slot({"driver_enable_status": False})})
        return _Slot(d)

    def GetSDKJointLimitParam(self, joint_name):
        assert joint_name in JOINTS, f"必须传 j1..j6，得到 {joint_name!r}"
        return _Slot({"joint": joint_name, "min": -1.0, "max": 1.0})

    def __getattr__(self, n):                    # 其余读接口给个平凡返回
        if n.startswith("Get") or n in ("isOk",):
            return lambda *a, **k: _Slot({"time_stamp": self._tick(), "name": n})
        raise AttributeError(n)


def e2e_selftest() -> bool:
    """**主流程端到端**回归 —— 这正是 24 项自测没覆盖的接线层。"""
    import tempfile
    print("\n" + "=" * 78 + "\n主流程端到端回归（假 SDK，**不碰 CAN**）\n" + "=" * 78)
    ok = {}
    cases = [("full", True, True), ("partial", True, False), ("silent", False, False)]
    for mode, q, expect_complete in cases:
        with tempfile.TemporaryDirectory() as td:
            argv = ["--i-confirm-arm-is-idle", "--fake-sdk", mode,
                    "--outdir", td, "--settle", "0.01", "--skip-guard"]
            if q:
                argv.append("--send-query-frames")
            rc = main(argv)
            files = list(Path(td).glob("*.json"))
            rec = json.loads(files[0].read_text()) if files else {}
            prog = rec.get("program_errors", [])
            reads = rec.get("reads", {})
            nerr = sum(1 for v in reads.values() if v.get("status") == "error")
            idle = rec.get("idle_check", {}).get("confirmed_idle")
            comp = (rec.get("query_evidence", {}).get("responses", {})
                    .get("GetAllMotorAngleLimitMaxSpd", {})
                    .get("completeness", {}).get("numbering_complete"))
            k = f"[{mode}]"
            ok[f"{k} 无程序性故障"] = not prog
            ok[f"{k} 退出码 0"] = rc == 0
            ok[f"{k} 读取项无 error"] = nerr == 0
            ok[f"{k} 落盘 1 个文件"] = len(files) == 1
            ok[f"{k} 静止确认成立"] = idle is True
            ok[f"{k} 六关节编号完整={expect_complete}"] = bool(comp) == expect_complete
            en = reads.get("enable_status", {})
            ok[f"{k} enable_status 源有效⇒判 valid（非 unknown）"] = (
                en.get("status") == "valid" and en.get("source_motor_keys")
                and len(en["source_motor_keys"]) == 6)
            print(f"  {mode:8} rc={rc} · 程序性故障 {len(prog)} · error 读取 {nerr} · "
                  f"静止确认 {idle} · 六关节完整 {comp}")
            for kk, v in list(ok.items())[-6:]:
                if not v:
                    print(f"     **✗ {kk}**")
    # 反向用例：程序性故障必须以非零退出码暴露
    with tempfile.TemporaryDirectory() as td:
        rc = main(["--i-confirm-arm-is-idle", "--fake-sdk", "full", "--outdir", td,
                   "--settle", "0.01", "--skip-guard", "--inject-program-error"])
        ok["[注入] 程序性故障 ⇒ 退出码 5"] = rc == 5
        print(f"  注入程序性故障：rc={rc}（应为 5）")
    bad = [k for k, v in ok.items() if not v]
    print(f"\n  ⇒ 端到端 {sum(ok.values())}/{len(ok)} 通过" + (f"   **✗ {bad}**" if bad else ""))
    print("  ⚠ `silent` = **限值/固件无应答但状态流正常**（不是「所有反馈都没有」）"
          "⇒ 静止确认照样成立；读不到限值属**被动 no_reply**，"
          "**不等同程序失败**，退出码仍为 0。")
    return not bad


# ══════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-confirm-arm-is-idle", action="store_true")
    ap.add_argument("--send-query-frames", action="store_true")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--fake-sdk", choices=("full", "partial", "silent"),
                    help="**回归测试用**：不导入 piper_sdk，用假接口跑完整主流程")
    ap.add_argument("--outdir", help="覆盖落盘目录（回归测试用）")
    ap.add_argument("--self-test-e2e", action="store_true",
                    help="用假 SDK 跑**主流程端到端**回归（不碰 CAN）")
    ap.add_argument("--skip-guard", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--inject-program-error", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.self_test_e2e:
        return 0 if (guard_selftest() and e2e_selftest()) else 1
    if not a.skip_guard and not guard_selftest():
        print("\n**✗ 自检未过 —— 不连接设备**")
        return 1
    if a.dry_run:
        print("\n  （--dry-run：到此为止，**未连接 CAN**）")
        return 0
    if not a.i_confirm_arm_is_idle:
        print("\n⛔ 未给 `--i-confirm-arm-is-idle` ⇒ **不连接设备**。")
        print("   请先现场确认：机械臂静止 · 无其他控制程序发送动作 · "
              "**不要为这一步切换电机状态**。")
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    outdir = Path(a.outdir) if a.outdir else OUTDIR
    out = outdir / f"readonly_{stamp}{'_q' if a.send_query_frames else ''}.json"
    rec = {"stage": "R (read-only)", "utc": datetime.now(timezone.utc).isoformat(),
           "host": platform.node(), "can_iface": a.can,
           "query_frames_requested": bool(a.send_query_frames),
           "can_link": can_state(a.can), "steps_done": [], "reads": {}}

    def save():
        outdir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))

    print("\n" + "=" * 78 + f"\nR 阶段读取（can={a.can}）\n" + "=" * 78)
    p = None
    try:
        allow = ALLOW_READ | (ALLOW_QUERY if a.send_query_frames else set())
        if a.fake_sdk:
            iface = FakeArm(a.fake_sdk)
            rec["FAKE_SDK"] = a.fake_sdk
            print(f"  ⚠⚠ **假 SDK 回归模式 `{a.fake_sdk}` —— 未导入 piper_sdk、未碰 CAN**")
        else:
            from piper_sdk import C_PiperInterface_V2
            iface = C_PiperInterface_V2(a.can)
        p = make_readonly(iface, allow)
        p.connect_readonly()
        rec["steps_done"].append("connect_readonly(piper_init=False)")
        time.sleep(0.5)
        rec["connected_flag"] = bool(p.get_connect_status())
        print(f"  连接读线程：{rec['connected_flag']}"
              f"   ⚠ **连接标记为真不证明收到了设备反馈**")

        # ⚠ **查询证据必须与普通状态流分开。**
        #   GetArmStatus / GetArmJointMsgs 是**持续更新**的状态流，
        #   它们的时间戳**即使没发查询也会推进** ⇒ 不能拿来证明查询已发送或得到回复。
        #   只有**应答式**接口（两个聚合 + 固件）的回复才算查询证据。
        RESPONSIVE = ("GetAllMotorAngleLimitMaxSpd", "GetAllMotorMaxAccLimit")
        STREAM = ("GetArmStatus", "GetArmJointMsgs")

        def snap(names):
            out = {}
            for n in names:
                try:
                    out[n] = timestamps(dump(getattr(p, n)()))
                except Exception as e:
                    out[n] = f"ERR {type(e).__name__}"
            return out

        # ⚠ **状态流必须用明确的采样间隔**，且**记录实际间隔**。
        #   旧版在不发查询帧时 pre/post 背靠背（零间隔）⇒ 时间戳逐位相同、显示 False，
        #   看起来像「断流」。**背靠背的 False 只能表示「这两次没观察到更新」。**
        STREAM_MIN_INTERVAL = 0.5
        pre_r, pre_s = snap(RESPONSIVE), snap(STREAM)
        t_pre = time.monotonic()
        pre_ts = {**pre_r, **pre_s}          # 主读取循环按接口名取「上一次时间戳」
        try:
            fw_pre = dump(p.GetPiperFirmwareVersion())
        except Exception as e:
            fw_pre = f"ERR {type(e).__name__}"

        called = []
        if a.send_query_frames:
            print("  ⚠ 调用三组查询（`PiperInit` 口径共 **13 帧**：6 角度限位+速度 · "
                  "6 加速度 · 1 固件）…")
            for meth in ("SearchAllMotorMaxAngleSpd", "SearchAllMotorMaxAccLimit",
                         "SearchPiperFirmwareVersion"):
                getattr(p, meth)(); called.append(meth)
            rec["steps_done"].append(
                "已尝试调用三组查询（**无发送层证据，不保证 13 帧全部发送成功**）")
            time.sleep(a.settle)
        else:
            print("  · **未发查询帧** ⇒ 应答式数据（限位/限速/加速度/固件）多半读不到")
        # 无论是否发查询，都保证一个明确的最小采样间隔
        gap = STREAM_MIN_INTERVAL - (time.monotonic() - t_pre)
        if gap > 0:
            time.sleep(gap)
        post_r, post_s = snap(RESPONSIVE), snap(STREAM)
        stream_interval = time.monotonic() - t_pre
        try:
            fw_post = dump(p.GetPiperFirmwareVersion())
        except Exception as e:
            fw_post = f"ERR {type(e).__name__}"

        resp = {}
        for n in RESPONSIVE:
            adv = advanced(pre_r.get(n), post_r.get(n))
            try:
                nk, vk = AGGREGATE[n]
                comp = check_six(dump(getattr(p, n)()), nk, vk)
            except Exception as e:
                comp = {"found": False, "complete": False,
                        "reason": f"{type(e).__name__}: {e}"}
            resp[n] = {"ts_pre": pre_r.get(n), "ts_post": post_r.get(n),
                       "advanced_forward": adv, "completeness": comp,
                       "reply_received": bool(adv) and bool(comp.get("found"))}
        fw_ok = isinstance(fw_post, str) and fw_post.strip() and fw_post != fw_pre
        rec["query_evidence"] = {
            "authorized": bool(a.send_query_frames),
            "methods_called": called,
            "send_layer_evidence": None,
            "send_layer_note": "SDK 不回报逐帧发送结果 ⇒ **无发送层证据**；"
                               "「调用了」不等于「13 帧都发出去了」",
            "responses": resp,
            "firmware_reply": {"pre": fw_pre, "post": fw_post, "newly_received": bool(fw_ok)},
            "note": "**只看应答式接口的回复**；状态流的推进另行报告，"
                    "**不能**用来证明查询已发送或得到回复",
        }
        rec["status_stream"] = {
            "pre": pre_s, "post": post_s,
            "sampling_interval_s": round(stream_interval, 4),
            "min_interval_requested_s": STREAM_MIN_INTERVAL,
            "advanced_forward": {n: advanced(pre_s.get(n), post_s.get(n)) for n in STREAM},
            "note": "**持续更新的状态流** —— 没发查询也会推进，**与查询是否成功无关**。"
                    "⚠ `advanced_forward=False` **只表示「这两次采样没观察到更新」**，"
                    "**不能解释成断流** —— 须结合 `sampling_interval_s` 读。",
        }
        print(f"  状态流采样间隔 {stream_interval:.3f} s（下限 {STREAM_MIN_INTERVAL}）")
        for n in RESPONSIVE:
            r_ = resp[n]
            print(f"  查询应答 {n:30} 收到={r_['reply_received']} · "
                  f"完整={r_['completeness'].get('complete')}")
        print(f"  固件应答：{'**新收到**' if fw_ok else '未新收到'}")
        print(f"  （状态流推进 {rec['status_stream']['advanced_forward']} —— "
              f"**不作为查询证据**）")

        reads = [
            ("firmware_version", "GetPiperFirmwareVersion", (),
             "固件版本；读不到时返回 -0x4AF"),
            ("arm_status_0x2A1", "GetArmStatus", (),
             "**设备当前状态（含控制模式）的权威来源**，CAN 0x2A1"),
            ("low_spd_info", "GetArmLowSpdInfoMsgs", (),
             "**`enable_status` 的源反馈** —— 单独读出来才能给派生值提供有效性证据"),
            ("enable_status", "GetArmEnableStatus", (),
             "⚠ **派生值**：由低速反馈缓存的 driver_enable_status 提取；"
             "**缓存初始化也全 False** ⇒ 源未核实时只能写「使能状态未核实」"),
            ("listened_ctrl_0x151", "GetArmModeCtrl", (),
             "⚠ **这是监听到的 0x151 控制命令缓存，不是设备当前模式反馈**；"
             "当前模式看 arm_status_0x2A1"),
            ("joint_now", "GetArmJointMsgs", (), "当前关节角"),
            ("angle_limit_max_spd_all", "GetAllMotorAngleLimitMaxSpd", (),
             "逐关节位置限位 + 最大速度（**应答式**，需先发查询帧）"),
            ("max_acc_limit_all", "GetAllMotorMaxAccLimit", (),
             "逐关节最大加速度（**应答式**）"),
            ("cur_angle_limit_max_vel", "GetCurrentMotorAngleLimitMaxVel", (), ""),
            ("cur_max_acc_limit", "GetCurrentMotorMaxAccLimit", (), ""),
            ("end_vel_acc_param", "GetCurrentEndVelAndAccParam", (), ""),
            ("crash_protection", "GetCrashProtectionLevelFeedback", (), ""),
            ("sdk_gripper_range", "GetSDKGripperRangeParam", (), "SDK 侧软限"),
            ("protocol_version", "GetCurrentProtocolVersion", (), ""),
            ("sdk_version", "GetCurrentSDKVersion", (), ""),
            ("interface_version", "GetCurrentInterfaceVersion", (), ""),
            ("can_fps", "GetCanFps", (), ""),
        ]
        # ⚠ GetSDKJointLimitParam **必须传 joint_name** ⇒ 逐关节读
        for j in JOINTS:
            reads.append((f"sdk_joint_limit_{j}", "GetSDKJointLimitParam", (j,),
                          "SDK 侧软限（**不是固件侧**）"))

        if a.inject_program_error:               # 回归用：制造一次接线错
            reads.insert(0, ("__injected__", "GetNoSuchMethod_ForRegression", (), ""))
        for key, meth, args, note in reads:
            entry = {"method": meth, "args": list(args), "note": note,
                     "source": SOURCE.get(meth, "device_feedback")}
            if meth in DERIVED_FROM:
                src_m, src_note = DERIVED_FROM[meth]
                entry["derived_from"] = src_m
                entry["derivation_note"] = src_note
            try:
                if key == "__injected__":
                    raise NameError("name 'pre' is not defined")   # 模拟本次的接线错
                v = dump(getattr(p, meth)(*args))
                entry["value"] = v
                prev = pre_ts.get(meth)          # ← 合并后的快照（旧版引用已删的 `pre`）
                st, why = classify(meth, v, prev_ts=prev)
                # **派生值**：源反馈未核实 ⇒ 一律 `unverified`，**不得**读作设备状态
                if meth in DERIVED_FROM:
                    src_m = DERIVED_FROM[meth][0]
                    src = next((r2 for r2 in rec["reads"].values()
                                if r2.get("method") == src_m), {})
                    src_st = src.get("status")
                    # ⚠ 源有效**还不够** —— 还要源里六个电机都在，且内容不是缓存默认的全零
                    sv = src.get("value") or {}
                    mots = [k for k in sv if k.startswith("motor_")] if isinstance(sv, dict) else []
                    nondefault = any(
                        isinstance(sv.get(k), dict)
                        and any(isinstance(sv[k].get(f), (int, float)) and sv[k].get(f)
                                for f in ("vol", "foc_temp", "motor_temp"))
                        for k in mots)
                    entry["source_status"] = src_st
                    entry["source_motor_keys"] = sorted(mots)
                    entry["source_nondefault_content"] = bool(nondefault)
                    if src_st != "valid" or len(mots) != 6 or not nondefault:
                        st = "unverified"
                        why = (f"源反馈 `{src_m}` status={src_st!r} · 电机项 {len(mots)}/6 · "
                               f"非默认内容={nondefault} ⇒ **本派生值未核实**；"
                               f"全 False **不得读作「未使能」**")
                    else:
                        # 源在收流、六机齐全、含真实物理量（电压/温度）⇒ 这些位是**设备自报**
                        st = "valid"
                        why = (f"**派生自设备反馈**：源 `{src_m}` 有效（六电机齐全，"
                               f"含真实电压/温度 ⇒ 非缓存默认）⇒ 这些位是**设备自报**的值。"
                               f"⚠ 仍只表示「驱动器自报的使能位」，"
                               f"其厂商语义待核实")
                entry["status"], entry["status_reason"] = st, why
                if meth in AGGREGATE:          # **只有两个聚合接口**做六关节检查
                    nk, vk = AGGREGATE[meth]
                    comp = check_six(v, nk, vk)
                    entry["six_motor"] = comp
                    if not comp.get("numbering_complete"):
                        entry["status"] = "no_reply" if st == "valid" else st
                        entry["status_reason"] = (
                            why + " ；**六关节编号不完整**：" +
                            "; ".join(comp.get("missing", []) or [comp.get("reason", "")]))
                    elif comp.get("value_semantics_unverified"):
                        # **不降级** —— 已回复，只是数值含义待核实
                        entry["status_reason"] = (
                            why + " ；六关节编号齐全，但 "
                            + ",".join(comp["zero_value_slots"])
                            + " **数值为零 ⇒ 已回复，数值含义待核实**")
                mark = {"valid": "✓", "stale": "◐", "no_reply": "✗", "error": "⚠",
                        "unknown": "?", "sdk_local": "·", "sdk_statistic": "·",
                        "unverified": "~"}[entry["status"]]
                print(f"  {mark} {key:26} {entry['status']:9} {entry['status_reason']}")
            except Exception as e:
                prog = isinstance(e, PROGRAM_ERRORS)
                entry["status"] = "error"
                entry["status_reason"] = f"{type(e).__name__}: {e}"
                entry["programmatic"] = prog
                if prog:
                    rec.setdefault("program_errors", []).append(f"{key}: {type(e).__name__}: {e}")
                print(f"  {'⛔' if prog else '⚠'} {key:26} error     "
                      f"{type(e).__name__}: {e}{'  ← **程序性故障**' if prog else ''}")
            rec["reads"][key] = entry

        # 静止确认：**要求时间戳推进**，否则不算确认
        j0 = dump(p.GetArmJointMsgs()); t0 = timestamps(j0)
        time.sleep(1.0)
        j1 = dump(p.GetArmJointMsgs()); t1 = timestamps(j1)
        # ⚠ **必须剥掉 time_stamp/Hz 再比关节角** —— 否则时间戳一推进，
        #   「读数变化」必为真，`confirmed_idle` **永远不成立**（前一版的 bug）。
        moved = strip_meta(j0) != strip_meta(j1)
        adv = advanced(t0, t1)             # **正向**推进；倒退判 False
        rec["idle_check"] = {
            "t0": j0, "t1": j1, "ts0": t0, "ts1": t1,
            "timestamps_advanced_forward": adv, "joint_values_changed": moved,
            "confirmed_idle": bool(adv is True and not moved),
            "note": "**必须 time_stamp 正向推进（证明在收流）且关节角不变**，"
                    "才算静止确认；断流时两份相同缓存**不算**。"
                    "比较关节角时已剥掉 time_stamp / Hz 等元数据。",
        }
        print(f"  静止确认：时间戳正向推进 {adv} · 关节角变化 {moved} ⇒ "
              f"**{'确认静止' if rec['idle_check']['confirmed_idle'] else '未确认'}**")
        rec["steps_done"].append("reads + idle_check")
    except Exception as e:
        import traceback
        prog = isinstance(e, PROGRAM_ERRORS)
        rec["fatal"] = {"error": f"{type(e).__name__}: {e}", "programmatic": prog,
                        "traceback": traceback.format_exc()}
        if prog:
            rec.setdefault("program_errors", []).append(f"fatal: {type(e).__name__}: {e}")
        print(f"\n  ⚠ 异常：{type(e).__name__}: {e}（**已完成的步骤仍会落盘**）")
    finally:
        if p is not None:
            try:
                p.disconnect()
                rec["steps_done"].append("disconnect")
                print("  ✓ 已断开（`DisconnectPort` 不发动作；"
                      "⚠ **但不能声称全部线程已确认退出** —— 见脚本头部）")
            except Exception as e:
                rec["disconnect_error"] = f"{type(e).__name__}: {e}"
            rec["calls_made"] = p.calls()
        rec["applicability"] = {
            "__instructions": "**结合厂商说明逐条填写；任何一项未确认保留 null，"
                              "不以推测填充。**",
            "control_mode": None, "jointctrl_input_semantics": None,
            "motionctrl2_third_param_100": None,
            "speed_limit_applies_to": None, "acc_limit_applies_to": None,
            "units": None, "is_factory_default_or_modified": None,
            "firmware_doc_ref": None,
        }
        rec["verdict"] = {"released": False,
                          "note": "**读到数字不自动放行。** 复放阻断保持生效。"}
        save()
        print(f"\n→ {out}")

    import collections as _c
    n_valid = sum(1 for v in rec["reads"].values() if v.get("status") == "valid")
    by_src = _c.Counter(v.get("source") for v in rec["reads"].values())
    dev = [v for v in rec["reads"].values() if v.get("source", "").startswith("device")]
    n_dev_valid = sum(1 for v in dev if v.get("status") == "valid")
    prog_err = rec.get("program_errors", [])
    print("\n" + "=" * 78)
    print(f"**设备反馈** 有效 {n_dev_valid}/{len(dev)} 项 ｜ 来源分布 {dict(by_src)}")
    print(f"⚠ **`status` 不是 `valid` 的项一律不得当作已读到的事实**；"
          f"`sdk_local`/`sdk_statistic` 是**本地值，不是设备反馈**；"
          f"`unverified` 是**派生值且源未核实**。")
    if prog_err:
        print(f"⛔ **{len(prog_err)} 项程序性故障**（接线/字段/签名错，不是设备没回）：")
        for m_ in prog_err[:5]:
            print(f"   ⛔ {m_}")
        print("   ⇒ **退出码 5**。被动读取下预期的 `no_reply` 不算程序失败。")
    print("⚠ **读到数字不自动放行。** `applicability` 各项尚未填写；")
    print("   尤其是「限速/加速度约束的是**下发目标的变化**还是"
          "**控制器内部生成的轨迹**」。")
    print("   ③c 复放阻断**保持生效**；是否解除**另行决定**。")
    print("=" * 78)
    return 5 if prog_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
