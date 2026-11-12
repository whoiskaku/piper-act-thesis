"""14 次响应 pilot 整理脚本的测试：合成数据，不读真实记录。"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import analyze_pilot_response14 as A   # noqa: E402

T = 2000


def _noise(rng, s=0.03):
    return rng.normal(0, s, (T, 3))


def test_pooled_sigma_recovers_the_single_replay_scatter():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.3, (T, 3))
    s, df = A.pooled_sigma([base + _noise(rng) for _ in range(5)])
    assert df == 4 and abs(s - 0.03) / 0.03 < 0.05
    assert abs(A.pair_sigma(base + _noise(rng), base + _noise(rng)) - 0.03) / 0.03 < 0.05


def test_responses_are_reported_per_sign_and_the_odd_part_cancels_the_baseline():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.3, (T, 3))
    r = 0.05 * np.sin(np.arange(T) / 7.0)[:, None] * np.array([1.0, 0.5, -0.3])
    nz = lambda: _noise(rng, 0.005)                                     # 信噪比约 5（r̄ 的噪声 = σ·√(1/2+1/5)）
    bare = [base + nz() for _ in range(5)]
    plus, minus = [base + r + nz() for _ in range(2)], [base - r + nz() for _ in range(2)]
    est = A.responses(plus, minus, np.mean(bare, 0))
    assert np.corrcoef(est["plus"].ravel(), r.ravel())[0, 1] > 0.9
    assert np.corrcoef(est["minus"].ravel(), -r.ravel())[0, 1] > 0.9
    assert np.corrcoef(est["odd"].ravel(), r.ravel())[0, 1] > 0.95
    assert A.rms(est["even"]) < 0.5 * A.rms(r)


def test_shared_baseline_uncertainty_matches_the_frozen_table():
    u = A.uncertainty(1.0, n=2, M=5)
    assert abs(u["var_r"] - 0.7) < 1e-12 and abs(u["cov"] - 0.2) < 1e-12 and abs(u["corr"] - 2 / 7) < 1e-12
    assert abs(u["var_ratio_mean4"] - 0.325 / 0.175) < 1e-12 and abs(u["se_ratio_mean4"] - np.sqrt(0.325 / 0.175)) < 1e-12
    assert abs(u["var_odd"] - 0.25) < 1e-12 and abs(u["var_even"] - 0.45) < 1e-12


def test_chi2_intervals_match_the_frozen_table():
    lo, hi = A.chi2_ci(4)
    assert abs(lo - 0.599) < 2e-3 and abs(hi - 2.87) < 1e-2
    lo, hi = A.chi2_ci(1)
    assert abs(lo - 0.446) < 2e-3 and abs(hi - 31.9) < 0.1


def test_linear_drift_slope_and_the_bias_it_puts_on_a_response():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.3, (T, 3))
    slope = 0.002 * np.cos(np.arange(T) / 11.0)[:, None] * np.ones(3)      # rad/s 每分钟
    times = np.array([0.0, 2.0, 8.5, 15.0, 16.5])
    bare = [base + slope * t + _noise(rng, 0.005) for t in times]
    d = A.drift(bare, times)
    assert np.corrcoef(d["slope"].ravel(), slope.ravel())[0, 1] > 0.95
    assert abs(d["t_mean"] - times.mean()) < 1e-12
    bias = A.drift_bias(d, 5.0)
    assert np.allclose(bias, d["slope"] * (5.0 - times.mean()))
    assert set(d["groups"]) == {"A12", "A3", "A45"} and d["diff_rms"]["A45-A12"] > 0


def test_abba_mean_times_use_actual_timestamps():
    att = [{"run_id": "BP_p_r1", "started_utc": "2026-10-05T03:03:01+00:00", "ended_utc": "2026-10-05T03:03:57+00:00"},
           {"run_id": "BP_m_r1", "started_utc": "2026-10-05T03:04:17+00:00", "ended_utc": "2026-10-05T03:05:13+00:00"},
           {"run_id": "BP_m_r2", "started_utc": "2026-10-05T03:05:33+00:00", "ended_utc": "2026-10-05T03:06:29+00:00"},
           {"run_id": "BP_p_r2", "started_utc": "2026-10-05T03:06:47+00:00", "ended_utc": "2026-10-05T03:07:43+00:00"}]
    ab = A.abba(att)
    assert abs(ab["P"]["plus_minus_gap_s"]) < 2.0 and ab["P"]["n_plus"] == 2 and ab["P"]["n_minus"] == 2


def test_records_must_match_the_ledger_hashes(tmp_path):
    rec = tmp_path / "records"
    rec.mkdir()
    (rec / "replay_A1.json").write_text(json.dumps({"imu": [[0, 0, 0, 1, 2, 3]] * 4}))
    import hashlib
    good = hashlib.sha256((rec / "replay_A1.json").read_bytes()).hexdigest()
    att = [{"run_id": "A1", "status": "complete", "record_sha256": good}]
    assert A.load_runs(tmp_path, att)["A1"].shape == (4, 3)
    att[0]["record_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        A.load_runs(tmp_path, att)


def test_setup_episode_takes_the_replay_gyro_and_rebuilds_the_gap():
    import q1_testing as QT
    e0 = QT.synthetic_episode(0, T=120)
    w = np.random.default_rng(3).normal(0, 0.2, (e0.T, 3))
    e = A.replay_episode(e0, w)
    assert np.array_equal(e.w, w) and np.allclose(e.gap, w[3:3 + e0.n] - e0.ideal[:e0.n])
    assert e.ideal is e0.ideal and e.cmd is e0.cmd
