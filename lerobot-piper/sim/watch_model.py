#!/usr/bin/env python3
"""实时监视 sim/piper_model.xml 的修改 — 文件保存即自动重载 MuJoCo viewer。

用法(在你自己的带图形界面的终端, lerobot env):
    cd ~/claude-test-2/lerobot-piper && python sim/watch_model.py [可选: 模型路径]

行为:
  - 打开 MuJoCo 窗口显示当前模型(静态初始位姿, 画出所有 site 坐标轴 → 可查看 IMU 位姿);
  - 检测到 XML 保存 → 自动关窗重载(≈0.5s), 你当场看到修改效果;
  - XML 写错时不崩: 打印错误, 3s 后重试, 修好即恢复;
  - 手动关窗且文件无新修改 → 退出。
"""
import os
import sys
import time

import mujoco
import mujoco.viewer

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "piper_model.xml")

print(f"[watch] 监视 {path}")
print("[watch] 保存修改将自动重载; 关窗且无修改则退出")

while True:
    try:
        mtime = os.path.getmtime(path)
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        print(f"[watch] 已加载: nbody={model.nbody} nsite={model.nsite} "
              f"nsensor={model.nsensor} nu={model.nu}")
    except Exception as e:  # XML 编辑到一半/语法错: 等待修复
        print(f"[watch] 加载失败: {e}\n[watch] 3s 后重试…")
        time.sleep(3)
        continue

    reload_needed = False
    with mujoco.viewer.launch_passive(model, data) as v:
        v.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # 画 site 坐标轴(IMU 朝向可见)
        v.sync()
        while v.is_running():
            try:
                if os.path.getmtime(path) != mtime:
                    time.sleep(0.3)  # 等编辑器写完整个文件
                    reload_needed = True
                    print("[watch] 检测到修改 → 重载")
                    break
            except FileNotFoundError:
                pass  # 编辑器原子写入的瞬间文件可能暂时不在
            v.sync()
            time.sleep(0.25)

    if not reload_needed:
        print("[watch] 窗口关闭且无修改 → 退出")
        break
