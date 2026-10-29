import copy

import numpy as np
import pytest
import torch

import q1_data as D
import q1_layered as LM
import q1_testing as T
import q1_windows as W


def _net(eps, seed=0, jitter=0.05):
    torch.manual_seed(seed)
    net = LM.make_servo(LM.joint_stats(eps))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * jitter)          # 打破零初始化才测得出依赖
    return net


def _pred(net, e, mode):
    ws = W.chunk_windows(e)
    traj = LM.open_loop_traj(net, e, max(w.t0 for w in ws)) if mode == "a" else None
    Q, U, M, F, _ = LM.window_inputs(net, e, ws, mode, traj)
    with torch.no_grad():
        return net.rollout(Q, U, M, F).numpy()


def test_zero_head_is_first_order_lag():
    eps = [T.synthetic_episode(s) for s in range(3)]
    torch.manual_seed(0)
    net = LM.make_servo(LM.joint_stats(eps))
    q = torch.zeros(1, 6, D.HIST + 1); u = torch.ones(1, 6, D.HIST + 1); m = torch.ones(1, D.HIST + 1)
    assert torch.allclose(net.step(q, u, m, torch.ones(1, 6)), torch.full((1, 6), LM.G_INIT))


@pytest.mark.parametrize("mode", ["a", "b"])
def test_no_commands_after_t1_and_no_measured_after_t0(mode):
    eps = [T.synthetic_episode(s) for s in range(3)]
    net = _net(eps)
    e = eps[2]
    w = W.chunk_windows(e)[2]
    base = _pred(net, e, mode)
    e1 = copy.copy(e); c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    assert np.array_equal(_pred(net, e1, mode)[2], base[2])
    e2 = copy.copy(e); q = e.q.copy(); q[w.t0 + 1:] += 0.05; e2.q = q
    assert np.array_equal(_pred(net, e2, mode)[2], base[2])


def test_1a_reads_no_measured_data_and_1b_does():
    eps = [T.synthetic_episode(s) for s in range(3)]
    net = _net(eps)
    e = eps[2]
    e2 = copy.copy(e); e2.q = e.q + 0.01
    assert np.array_equal(_pred(net, e2, "a"), _pred(net, e, "a"))
    assert not np.array_equal(_pred(net, e2, "b"), _pred(net, e, "b"))


def test_1b_first_chunk_history_is_fully_masked():
    e = T.synthetic_episode(0)
    q, u, m = LM.history_measured(e, 0)
    assert m.sum() == 0 and np.allclose(q, e.cmd[0, :6][:, None])
    q, u, m = LM.history_measured(e, 100)
    assert m.all() and np.allclose(q.T, e.q[80:101])


def test_open_loop_traj_starts_at_rest_on_first_command():
    eps = [T.synthetic_episode(s) for s in range(2)]
    net = _net(eps)
    tr = LM.open_loop_traj(net, eps[1], 120)
    assert tr.shape == (121, 6) and np.allclose(tr[0], eps[1].cmd[0, :6])


def test_param_count():
    eps = [T.synthetic_episode(s) for s in range(2)]
    torch.manual_seed(0)
    assert sum(p.numel() for p in LM.make_servo(LM.joint_stats(eps)).parameters()) == 24780
