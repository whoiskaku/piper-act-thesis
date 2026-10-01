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
    a = TR.train_one(mid, tr, seed=0, steps=3)
    assert torch.get_num_threads() == TR.CONFIG["threads"]
    b = TR.train_one(mid, tr, seed=0, steps=3)
    assert TR.state_sha256(a["model"]) == TR.state_sha256(b["model"]) and a["loss"] == b["loss"]
    c = TR.train_one(mid, tr, seed=1, steps=3)
    assert TR.state_sha256(c["model"]) != TR.state_sha256(a["model"])
    eh = TR.predict_episode(mid, a["model"], a["stats"], te, W.chunk_windows(te), R, Wrs, C)
    assert eh.shape == (te.n, 3) and np.isfinite(eh).all()


@pytest.mark.parametrize("mid", ["1b", "2b", "3b"])
def test_prediction_ignores_measurements_after_origin(mid, syn):
    tr, te, R, Wrs, C = syn
    out = TR.train_one(mid, tr, seed=0, steps=2)
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


def test_config_hash_tracks_steps():
    assert TR.config_hash(3000) != TR.config_hash(2999)
    assert TR.config_hash(3000) == TR.config_hash(3000)


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
