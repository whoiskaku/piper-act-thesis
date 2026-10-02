import copy

import numpy as np
import pytest

import q1_sensor as S
import q1_testing as T
import q1_train as TR
import q1_windows as W


@pytest.fixture(scope="module")
def syn():
    tr = [T.synthetic_episode(s, sess=f"s{s % 4}/A") for s in range(8)]
    te = T.synthetic_episode(99, sess="hold/A")
    R = np.eye(3)
    Wrs = S.fit_rs(tr, R)
    return tr, te, R, Wrs, S.fit_cmap(tr, R, Wrs)


@pytest.mark.parametrize("mid", list(TR.MODELS))
def test_train_is_bit_reproducible_and_predicts(mid, syn):
    import torch
    tr, te, R, Wrs, C = syn
    a = TR.train_one(mid, tr, seed=0, ckpts=(3,))
    assert torch.get_num_threads() == TR.CONFIG["threads"]
    b = TR.train_one(mid, tr, seed=0, ckpts=(3,))
    assert TR.state_sha256(a["model"]) == TR.state_sha256(b["model"]) and a["loss"] == b["loss"]
    c = TR.train_one(mid, tr, seed=1, ckpts=(3,))
    assert TR.state_sha256(c["model"]) != TR.state_sha256(a["model"])
    eh = TR.predict_episode(mid, a["model"], a["stats"], te, W.chunk_windows(te), R, Wrs, C)
    assert eh.shape == (te.n, 3) and np.isfinite(eh).all()


@pytest.mark.parametrize("mid", ["1b", "2b", "3b"])
def test_prediction_ignores_measurements_after_origin(mid, syn):
    tr, te, R, Wrs, C = syn
    out = TR.train_one(mid, tr, seed=0, ckpts=(2,))
    ws = W.chunk_windows(te)
    base = TR.predict_episode(mid, out["model"], out["stats"], te, ws, R, Wrs, C)
    w = ws[3]
    e2 = copy.copy(te)
    q = te.q.copy(); q[w.t0 + 1:] += 0.05; e2.q = q
    ww = te.w.copy(); ww[w.t0 + 1:] += 0.5; e2.w = ww
    p2 = TR.predict_episode(mid, out["model"], out["stats"], e2, ws, R, Wrs, C)
    assert np.array_equal(p2[w.t0:w.s1 + 1], base[w.t0:w.s1 + 1])


@pytest.mark.parametrize("mid", list(TR.MODELS))
def test_forward_windows_ignore_commands_after_t1(mid, syn):
    """训练用的前向（进联合损失的 q̂ 与 ê）也不得读下一 chunk 的指令/理想陀螺。"""
    import torch
    tr, te, R, Wrs, C = syn
    torch.manual_seed(0)
    model, st = TR.build(mid, tr)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ws = W.chunk_windows(te)
    w = ws[2]

    def fw(ep):
        with torch.no_grad():
            Ee, Qq, _ = TR.forward_windows(mid, model, st, ep, ws)
        return (None if Ee is None else Ee[2].numpy()), Qq[2].numpy()

    bE, bQ = fw(te)
    e1 = copy.copy(te)
    c = te.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    i_ = te.ideal.copy(); i_[w.t1 + 1:] += 0.5; e1.ideal = i_
    E1, Q1 = fw(e1)
    assert np.array_equal(Q1, bQ) and (bE is None or np.array_equal(E1, bE))


def test_split_fold_excludes_held_out_session():
    eps = [T.synthetic_episode(s, sess=f"s{s % 3}/A") for s in range(6)]
    tr, te = TR.split_fold(eps, "s1/A")
    assert {e.sess for e in te} == {"s1/A"} and "s1/A" not in {e.sess for e in tr}
    with pytest.raises(ValueError):
        TR.split_fold(eps, "nope/A")


def test_config_hash_tracks_the_selection_rule(monkeypatch):
    """配置指纹锁定的是选模规则（检查点、平局容差、上限……），不是某个作业选中的步数。"""
    h = TR.config_hash()
    assert TR.config_hash() == h and len(h) == 16
    assert TR.CONFIG["ckpts"][-1] == TR.CONFIG["max_steps"] == 10000 and len(TR.CONFIG["ckpts"]) == 11
    monkeypatch.setitem(TR.CONFIG, "ckpts", (0, 10000))
    assert TR.config_hash() != h
    monkeypatch.undo()
    monkeypatch.setitem(TR.CONFIG, "tie_rtol", 1e-9)
    assert TR.config_hash() != h


@pytest.mark.parametrize("mid", ["1a", "2b"])
def test_checkpoints_are_prefixes_of_one_run(mid, syn):
    """同种子 ⇒ 训到第 s 步的权重 = 长跑中第 s 步的快照（终训只训到 step* 才成立）。"""
    tr, te, R, Wrs, C = syn
    long = TR.train_one(mid, tr, seed=0, ckpts=(0, 2, 4))
    short = TR.train_one(mid, tr, seed=0, ckpts=(2,))
    zero = TR.train_one(mid, tr, seed=0, ckpts=(0,))
    assert sorted(long["ckpt"]) == [0, 2, 4] and sorted(short["ckpt"]) == [2]
    assert TR.state_dict_sha(long["ckpt"][2]) == TR.state_dict_sha(short["ckpt"][2])
    assert TR.state_dict_sha(long["ckpt"][0]) == TR.state_dict_sha(zero["ckpt"][0])
    assert TR.state_dict_sha(long["ckpt"][4]) == TR.state_sha256(long["model"])
    assert TR.state_dict_sha(long["ckpt"][2]) != TR.state_dict_sha(long["ckpt"][4])
    m2 = TR.with_weights(mid, long["stats"], long["ckpt"][2])
    assert TR.state_sha256(m2) == TR.state_dict_sha(short["ckpt"][2])


def test_pick_step_takes_the_earliest_minimum_within_full_precision_ties():
    assert TR.pick_step({0: 0.5, 250: 0.4, 500: 0.4 + 1e-17, 1000: 0.41}) == 250
    assert TR.pick_step({0: 0.4, 250: 0.4, 500: 0.39999}) == 500
    assert TR.pick_step({0: 0.4, 250: 0.4}) == 0
    assert TR.pick_step({10000: 0.1, 0: 0.2}) == 10000


def test_step_zero_is_not_uniformly_a_zero_prediction(syn):
    """第 0 步：端到端的误差输出恰为 0；分层模型仍有伺服先验与已拟合的传感器层 ⇒ 非零。"""
    tr, te, R, Wrs, C = syn
    ws = W.chunk_windows(te)
    for mid in ("2a", "3b"):
        r = TR.train_one(mid, tr, seed=0, ckpts=(0,))
        assert (TR.predict_episode(mid, r["model"], r["stats"], te, ws, R, Wrs, C) == 0).all()
    for mid in ("1a", "1b"):
        r = TR.train_one(mid, tr, seed=0, ckpts=(0,))
        assert np.abs(TR.predict_episode(mid, r["model"], r["stats"], te, ws, R, Wrs, C)).max() > 0


def test_session_scores_pool_within_each_session(syn):
    tr, te, R, Wrs, C = syn
    r = TR.train_one("2a", tr, seed=0, ckpts=(2,))
    eps = tr[:4]
    got = TR.session_scores("2a", r["model"], r["stats"], eps, R, Wrs, C)
    K = W.kernel()
    for sess in {e.sess for e in eps}:
        num = den = 0.0
        for e in eps:
            if e.sess == sess:
                n_, d_, _ = W.score(e.gap, TR.predict_episode("2a", r["model"], r["stats"], e,
                                                              W.chunk_windows(e), R, Wrs, C), K)
                num += n_; den += d_
        assert np.isclose(got[sess][0], num, rtol=1e-12) and np.isclose(got[sess][1], den, rtol=1e-12)


@pytest.mark.parametrize("mid", ["1a", "1b"])
@pytest.mark.parametrize("which", ["main", "mid"])
def test_layered_scored_output_ignores_next_chunk(mid, which, syn):
    """分层模型被计分的 ê 由传感器层分支构造（u_span / i_span）：改 t1 之后的指令或理想陀螺，
    每个窗口（含末尾截断的 chunk、中段起点）都必须逐位不变。"""
    import torch
    tr, te, R, Wrs, C = syn
    torch.manual_seed(0)
    model, st = TR.build(mid, tr)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ws = W.chunk_windows(te) if which == "main" else W.mid_windows(te)
    base = TR.predict_windows(mid, model, st, te, ws, R, Wrs, C)
    for field, delta in (("cmd", 0.05), ("ideal", 0.5)):
        for i, w in enumerate(ws):
            e1 = copy.copy(te)
            arr = getattr(te, field).copy()
            arr[w.t1 + 1:] += delta
            setattr(e1, field, arr)
            assert np.array_equal(TR.predict_windows(mid, model, st, e1, ws, R, Wrs, C)[i], base[i]), (field, w)
