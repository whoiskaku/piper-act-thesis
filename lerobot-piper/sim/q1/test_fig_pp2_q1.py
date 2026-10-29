import numpy as np

import fig_pp2_q1 as F


def test_best_shift_recovers_known_lag():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    pairs = [(x[1:301], x[:300])]                  # real[k] = x[k+1] ⇒ real[k−1] 对 pred[k]，L = −1
    assert F.best_shift(pairs) == -1
    assert F.ratio(x[:300], x[:300], 0) == 0.0


def test_plot_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    rows = [{"name": f"ep{i}", "real": rng.normal(size=(300, 3)), "new": rng.normal(size=(300, 3)),
             "g2": rng.normal(size=(300, 3)), "r_new": float(i), "r_g2": 1.0} for i in range(5)]
    p = tmp_path / "f.png"
    F.plot(rows, "2b", 0, 1, p)
    assert p.exists() and p.stat().st_size > 10_000
