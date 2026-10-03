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
