import copy

import numpy as np
import pytest
import torch

import q1_windows as W
import q2_models as Mo
import q2_testing as T
import q2_train as TR


@pytest.fixture(scope="module")
def syn():
    bare = [T.synthetic(s, "ab6", "A", "bare") for s in range(3)]
    corr = [T.synthetic(10 + s, "ab6", "D", "corr") for s in range(3)]
    nom = TR.train_nominal(bare, 0, ckpts=(2,))
    return bare, corr, (nom["model"], nom["stats"])


def test_code_fingerprint_covers_the_roster_and_reused_stage1_code():
    assert "q2_roster.json" in TR.CODE_FILES_Q2 and "q1_train.py" in TR.CODE_FILES_Q1
    assert len(TR.code_sha()) == 64 and TR.code_sha() == TR.code_sha()


def test_config_hash_tracks_the_frozen_rules(monkeypatch):
    h = TR.config_hash()
    for key, val in (("gate_d", 0.04), ("nonlin_margin", 0.03), ("ckpts", (0, 10000)), ("basis_sd", 0.1)):
        monkeypatch.setitem(TR.CONFIG, key, val)
        assert TR.config_hash() != h
        monkeypatch.undo()
    assert TR.config_hash() == h


def test_nominal_is_stage1_train_one():
    import q1_train as TR1
    bare = [T.synthetic(s, "ab6", "A", "bare") for s in range(2)]
    a = TR.train_nominal(bare, 1, ckpts=(3,))
    b = TR1.train_one("2a", bare, 1, ckpts=(3,))
    assert TR1.state_dict_sha(a["ckpt"][3]) == TR1.state_dict_sha(b["ckpt"][3])


@pytest.mark.parametrize("form", Mo.FORMS)
def test_increment_training_is_reproducible_and_seed_dependent(form, syn):
    import q1_train as TR1
    _, corr, nom = syn
    a = TR.train_increment(form, nom, corr, 0, ckpts=(3,))
    b = TR.train_increment(form, nom, corr, 0, ckpts=(3,))
    c = TR.train_increment(form, nom, corr, 1, ckpts=(3,))
    sha = lambda r: TR1.state_dict_sha(r["ckpt"][3])
    assert sha(a) == sha(b) and a["loss"] == b["loss"] and sha(a) != sha(c)


@pytest.mark.parametrize("form", Mo.FORMS)
def test_checkpoints_are_prefixes_and_step_zero_is_zero_response(form, syn):
    import q1_train as TR1
    _, corr, nom = syn
    long = TR.train_increment(form, nom, corr, 0, ckpts=(0, 2, 4))
    short = TR.train_increment(form, nom, corr, 0, ckpts=(2,))
    assert TR1.state_dict_sha(long["ckpt"][2]) == TR1.state_dict_sha(short["ckpt"][2])
    e = corr[0]
    m0 = TR.with_increment(form, long["ckpt"][0])
    d0 = TR.increment_episode(form, m0, long["st_inc"], long["scale"], nom[1], e)
    assert (d0 == 0).all()                                            # 第 0 步 = 零响应
    m4 = TR.with_increment(form, long["ckpt"][4])
    assert np.abs(TR.increment_episode(form, m4, long["st_inc"], long["scale"], nom[1], e)).max() > 0


def test_increment_statistics_come_only_from_the_given_training_runs(syn):
    _, corr, nom = syn
    a = TR.train_increment("condlinear", nom, corr[:2], 0, ckpts=(1,))
    b = TR.train_increment("condlinear", nom, corr, 0, ckpts=(1,))
    assert not np.allclose(a["scale"], b["scale"]) and a["norms"]["c_w"] != b["norms"]["c_w"]
    assert np.allclose(a["scale"], Mo.da_scale(corr[:2]))


def test_predictor_shapes_and_h_equals_the_whole_episode_convolution(syn):
    import q2_windows as W2
    _, corr, nom = syn
    e = corr[1]
    H = np.random.default_rng(0).normal(0, 1, (16, 3, 6))
    assert TR.nominal_episode(*nom, e).shape == (e.n, 3)
    assert np.allclose(TR.h_episode(H, e), W2.h_full(H, e)[:e.n])


def test_metrics_and_beta_parts():
    e = T.synthetic(0)
    K = W.kernel()
    zero = np.zeros_like(e.gap)
    r = TR.ep_metrics(e, zero, K)
    assert r["num"] == r["den"] and set(r["lead"]) == {"3-6", "7-12", "13-27", "28-52", "53-96", "97-102"}
    assert (r["name"], r["fold"], r["arm"]) == (e.name, e.fold, e.arm)
    num, den = TR.beta_parts(e, zero, e.gap, K)                       # 增量 = 残差 ⇒ β = 1
    assert np.isclose(num / den, 1.0)
    num, den = TR.beta_parts(e, zero, zero, K)                        # 增量恒为 0 ⇒ 分母 0（β 不定义）
    assert den == 0.0


def test_pick_step_earliest_minimum_within_full_precision_ties():
    assert TR.pick_step({0: 0.5, 250: 0.4, 500: 0.4 + 1e-17}) == 250
    assert TR.pick_step({0: 0.4, 250: 0.4}) == 0
    assert TR.pick_step({10000: 0.1, 0: 0.2}) == 10000


def test_condlinear_learns_a_planted_linear_response():
    """合成：gap 里加入已知的线性响应 Hs ⊛ δa；条件线性训练后应在留出集上与之高度相关。"""
    import fit_step1_candidate as S1
    rng = np.random.default_rng(0)
    Hs = rng.normal(0, 30.0, (16, 3, 6))
    def plant(e):
        resp = S1.causal_conv(Hs, e.da)[:e.n]
        e.gap = e.gap + resp
        return e, resp
    bare = [T.synthetic(s, "ab6", "A", "bare") for s in range(2)]
    corr = [plant(T.synthetic(20 + s, "ab6", "D", "corr"))[0] for s in range(4)]
    held, resp = plant(T.synthetic(99, "ab6", "D", "corr"))
    nom = TR.train_nominal(bare, 0, ckpts=(1,))
    r = TR.train_increment("condlinear", (nom["model"], nom["stats"]), corr, 0, ckpts=(400,))
    d = TR.increment_episode("condlinear", r["model"], r["st_inc"], r["scale"], nom["stats"], held)
    P = lambda x: W.P(x, W.kernel())
    c = float((P(d) * P(resp)).sum() / np.sqrt((P(d) ** 2).sum() * (P(resp) ** 2).sum()))
    assert c > 0.5, c
