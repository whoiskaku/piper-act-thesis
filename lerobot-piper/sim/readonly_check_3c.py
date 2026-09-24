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
    slots, missing = {}, []
    for i in range(1, 7):
        e = m[i]
        num = e.get(num_key)
        has_val = any(isinstance(e.get(k), (int, float)) and e.get(k) != 0
                      for k in val_keys)
        slots[f"slot{i}"] = {"num": num, "has_value": has_val,
                             **{k: e.get(k) for k in val_keys}}
        if num != i:
            missing.append(f"slot{i} 的 {num_key}={num}（应为 {i}）")
        elif not has_val:
            missing.append(f"slot{i} 编号对但**数值全为 0/缺失**")
    return {"found": True, "complete": not missing, "num_key": num_key,
            "placeholder_slot0": m[0].get(num_key),
            "slots": slots, "missing": missing}


def classify(name, val, prev_ts=None):
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

    chk("18 未导入 lerobot 的 robot 包装",
        "lerobot.common.robot_devices.robots" not in sys.modules)
    print(f"\n  ⇒ 自检 {sum(ok.values())}/{len(ok)} 通过")
    print("  ⚠ 自检只覆盖**代理与判定逻辑**；真实接口的反馈内容仍须上机后按 "
          "`status` 字段逐项核对。")
    return all(ok.values())


# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-confirm-arm-is-idle", action="store_true")
    ap.add_argument("--send-query-frames", action="store_true")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--settle", type=float, default=1.5)
    a = ap.parse_args()

    if not guard_selftest():
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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUTDIR / f"readonly_{stamp}{'_q' if a.send_query_frames else ''}.json"
    rec = {"stage": "R (read-only)", "utc": datetime.now(timezone.utc).isoformat(),
           "host": platform.node(), "can_iface": a.can,
           "query_frames_requested": bool(a.send_query_frames),
           "can_link": can_state(a.can), "steps_done": [], "reads": {}}

    def save():
        OUTDIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))

    print("\n" + "=" * 78 + f"\nR 阶段读取（can={a.can}）\n" + "=" * 78)
    p = None
    try:
        from piper_sdk import C_PiperInterface_V2
        allow = ALLOW_READ | (ALLOW_QUERY if a.send_query_frames else set())
        p = make_readonly(C_PiperInterface_V2(a.can), allow)
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

        pre_r, pre_s = snap(RESPONSIVE), snap(STREAM)
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
        post_r, post_s = snap(RESPONSIVE), snap(STREAM)
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
            "advanced_forward": {n: advanced(pre_s.get(n), post_s.get(n)) for n in STREAM},
            "note": "**持续更新的状态流** —— 没发查询也会推进，"
                    "**与查询是否成功无关**，仅用于判断是否在收流",
        }
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
            ("enable_status", "GetArmEnableStatus", (), "使能状态（只读）"),
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

        for key, meth, args, note in reads:
            entry = {"method": meth, "args": list(args), "note": note}
            try:
                v = dump(getattr(p, meth)(*args))
                entry["value"] = v
                prev = pre.get(meth) if meth in pre else None
                st, why = classify(meth, v, prev_ts=prev)
                entry["status"], entry["status_reason"] = st, why
                if meth in AGGREGATE:          # **只有两个聚合接口**做六关节检查
                    nk, vk = AGGREGATE[meth]
                    comp = check_six(v, nk, vk)
                    entry["six_motor"] = comp
                    if not comp.get("complete"):
                        entry["status"] = "no_reply" if st == "valid" else st
                        entry["status_reason"] = (
                            why + " ；**六关节不完整**：" +
                            "; ".join(comp.get("missing", []) or [comp.get("reason", "")]))
                mark = {"valid": "✓", "stale": "◐", "no_reply": "✗",
                        "error": "⚠", "unknown": "?"}[entry["status"]]
                print(f"  {mark} {key:26} {entry['status']:9} {entry['status_reason']}")
            except Exception as e:
                entry["status"], entry["status_reason"] = "error", f"{type(e).__name__}: {e}"
                print(f"  ⚠ {key:26} error     {type(e).__name__}: {e}")
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
        rec["fatal"] = {"error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()}
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

    n_valid = sum(1 for v in rec["reads"].values() if v.get("status") == "valid")
    print("\n" + "=" * 78)
    print(f"有效反馈 {n_valid}/{len(rec['reads'])} 项 —— "
          f"**`status` 不是 `valid` 的项一律不得当作已读到的事实**")
    print("⚠ **读到数字不自动放行。** `applicability` 各项尚未填写；")
    print("   尤其是「限速/加速度约束的是**下发目标的变化**还是"
          "**控制器内部生成的轨迹**」。")
    print("   ③c 复放阻断**保持生效**；是否解除**另行决定**。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
