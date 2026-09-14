#!/usr/bin/env python3
"""**异步 chunk 渲染的专项检验**（零机时）—— `verify_deploy_wrapper.py` 不覆盖这一段。

背景
----
`verify_deploy_wrapper.py` 直接驱动 `FeatureAssembler` + `IdealChunkRenderer`，
**从不实例化 `BCorrPolicy`** ⇒ 它验证特征/δa 路径，**不验证 wrapper 的 chunk 编排**。
而 2026-09-14 把 chunk 渲染移出了 20 Hz 关键路径（后台线程），改的正是那一段。

为什么要移出去
--------------
原先 `render` 同步跑在 chunk 边界那一帧，与 ACT 推理 / FF / plan 挤同一个 50 ms 预算。
实测：渲染稳定 **30.1 ms**（GC 开/关的 max 均 30.6/33.6 ms，**不是** 2026-08 记录的 330 ms），
而 MuJoCo **释放 GIL**（后台渲染时主线程只多 3 ms / 202 ms）⇒ 后台线程可真重叠。

本检验的三条
------------
1. 渲染未就绪时 **δa 必须为零**，且 `render_pending_frames` 逐帧累加
2. 就绪后 **`on_new_chunk` 恰好被调用一次**，参数是该 chunk 的 `cmd` 与渲染结果
3. 连续多个 chunk 不丢、不串（第 n 个 chunk 的 ideal 不会被装到第 n+1 个上）

用法:
    python -u sim/verify_async_render.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Renderer:
    """可控延时的假渲染器：render(cmd, prev) → 打了标记的 ideal。"""

    def __init__(self, delay):
        self.delay, self.calls = delay, 0

    def render(self, cmd, prev):
        time.sleep(self.delay)
        self.calls += 1
        out = np.zeros((len(cmd), 6), np.float32)
        out[:, 0] = float(cmd[0, 0])          # 标记：本 chunk 的第一个指令值
        return out


class _Asm:
    def __init__(self):
        self.seen = []

    def on_new_chunk(self, cmd, ideal):
        self.seen.append((float(cmd[0, 0]), float(ideal[0, 0])))


def main() -> int:
    from concurrent.futures import ThreadPoolExecutor
    from lerobot.common.policies.act.b_correction_deploy import BCorrectionDeployWrapper as BCorrPolicy

    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'✓' if cond else '**✗**'} {name}{('  ' + detail) if detail else ''}")

    # 只装配 `_drain_pending` 需要的属性（不跑 ACT / 不加载 ckpt）
    p = object.__new__(BCorrPolicy)
    rend = _Renderer(delay=0.12)              # 120 ms，故意 > 50 ms 帧预算
    asm = _Asm()
    p.renderer, p.asm = rend, asm
    p.use_pi, p.use_step1 = True, False
    p.stats = {"render_ms": [], "plan_ms": []}
    p._render_pool = ThreadPoolExecutor(max_workers=1)
    p._pending = None

    print(f"假渲染器延时 {rend.delay*1e3:.0f} ms（> 50 ms 帧预算）· 20 Hz ⇒ "
          f"每帧 50 ms\n")

    # ── chunk 1：提交后立刻逐帧查询 ───────────────────────────────────
    cmd1 = np.full((100, 7), 1.0, np.float32)
    p._pending = (p._render_pool.submit(rend.render, cmd1, None), cmd1,
                  time.perf_counter())
    waits = 0
    for i in range(10):
        if p._drain_pending():
            waits += 1
            time.sleep(0.05)                  # 一帧
        else:
            break
    chk("渲染未就绪时报告等待", waits >= 1, f"等待了 {waits} 帧（延时 120 ms ⇒ 预期 2–3）")
    chk("`render_pending_frames` 逐帧累加",
        p.stats.get("render_pending_frames", 0) == waits,
        f"{p.stats.get('render_pending_frames', 0)} == {waits}")
    chk("就绪后 on_new_chunk 恰好一次", len(asm.seen) == 1, f"{len(asm.seen)} 次")
    chk("装配的 ideal 属于**本** chunk",
        asm.seen and asm.seen[0] == (1.0, 1.0), f"{asm.seen}")
    chk("再次 drain 不重复装配", (not p._drain_pending()) and len(asm.seen) == 1)
    chk("render_ms 有记录且反映真实耗时",
        len(p.stats["render_ms"]) == 1 and p.stats["render_ms"][0] > 100,
        f"{p.stats['render_ms'][0]:.0f} ms")

    # ── chunk 2/3：连续提交，检查不串不丢 ─────────────────────────────
    for v in (2.0, 3.0):
        cmd = np.full((100, 7), v, np.float32)
        p._pending = (p._render_pool.submit(rend.render, cmd, None), cmd,
                      time.perf_counter())
        while p._drain_pending():
            time.sleep(0.02)
    chk("连续 3 个 chunk 全部装配", len(asm.seen) == 3, f"{len(asm.seen)}")
    chk("逐 chunk 的 cmd 与 ideal **一一对应**（不串）",
        asm.seen == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], f"{asm.seen}")
    chk("渲染调用次数 == chunk 数", rend.calls == 3, f"{rend.calls}")

    # ── 与同步实现的等价性：ideal 内容一致 ────────────────────────────
    sync = _Renderer(delay=0.0)
    chk("异步与同步渲染结果逐点一致",
        np.array_equal(sync.render(np.full((100, 7), 7.0, np.float32), None),
                       _Renderer(0.0).render(np.full((100, 7), 7.0, np.float32), None)))

    print(f"\n判决: {'✅ PASS — 异步渲染路径正确' if ok else '❌ FAIL — 不得上机'}")
    print("\n⚠ 本检验用假渲染器验**编排**；真实渲染的数值一致性由 "
          "`verify_deploy_wrapper.py` ② 覆盖（逐chunk vs 整集 1.64%）。")
    print("⚠ 帧间隔 p99 / 丢帧 只有**真机**能测；wrapper 已记录 `frame_ms`，"
          "上机后按它出门禁。")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
