"""`sim/q1` 的 pytest 配置。

MuJoCo + scipy 共存时 `json.loads` 期间的 GC 会 SIGSEGV（见 infra 记录）⇒ 导入即关 GC。
"""
import gc
import sys
from pathlib import Path

import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 读 100 集真实数据或跑 G-2/MuJoCo 的集成测试")


@pytest.fixture(scope="session")
def bare_eps():
    import q1_data as D
    return D.load_bare()


@pytest.fixture(scope="session")
def calib():
    import q1_data as D
    return D.calibration()
