import numpy as np
import torch

import q1_data as D
import q1_losses as LS
import q1_testing as T
import q1_windows as W


def test_torch_band_matches_numpy_P():
    e = T.synthetic_episode(0)
    r = np.random.default_rng(0).normal(size=(e.n, 3))
    b = LS._bp(torch.tensor(r - r.mean(0)), W.kernel()).numpy()
    assert np.abs(W.P(r, W.kernel()) - b).max() < 1e-12


def test_omega_loss_at_zero_is_relative_wall_energy():
    eps = [T.synthetic_episode(s) for s in range(3)]
    nm = LS.loss_norms(eps)
    K = W.kernel()
    e0 = [W.score(x.gap, 0 * x.gap, K) for x in eps]
    z = LS.omega_loss(eps[0], torch.zeros(eps[0].n, 3, dtype=torch.float64), nm).item()
    assert np.isclose(z, e0[0][0] / e0[0][2] / nm["c_w"], rtol=1e-12)
    assert np.isclose(np.mean([a / c for a, _, c in e0]) / nm["c_w"], 1.0)


def test_joint_loss_ignores_invalid_frames():
    e = T.synthetic_episode(0)
    e.q_valid = e.q_valid.copy(); e.q_valid[300] = False
    nm = LS.loss_norms([e])
    q = torch.tensor(e.cmd[D.ALIGN_L:D.ALIGN_L + e.n, :6])
    base = LS.joint_loss(e, q.clone(), nm).item()
    q2 = q.clone(); q2[300 - D.ALIGN_L] += 10.0
    assert LS.joint_loss(e, q2, nm).item() == base


def test_stored_value_at_invalid_frame_changes_nothing():
    """无效帧里存的实测值不得经带通进入关节损失或其归一化尺度。"""
    import copy
    a = T.synthetic_episode(0)
    a.q_valid = a.q_valid.copy(); a.q_valid[300] = False
    a.q = a.q.copy(); a.q[300] = 0.0
    b = copy.copy(a); b.q = a.q.copy(); b.q[300] = 5.0
    na, nb = LS.loss_norms([a]), LS.loss_norms([b])
    assert all(np.array_equal(na[k], nb[k]) for k in na)
    qh = torch.tensor(a.cmd[D.ALIGN_L:D.ALIGN_L + a.n, :6])
    assert LS.joint_loss(a, qh, na).item() == LS.joint_loss(b, qh, na).item()


def test_losses_are_differentiable():
    e = T.synthetic_episode(1)
    nm = LS.loss_norms([e])
    eh = torch.zeros(e.n, 3, dtype=torch.float64, requires_grad=True)
    qh = torch.tensor(e.cmd[D.ALIGN_L:D.ALIGN_L + e.n, :6], requires_grad=True)
    (LS.omega_loss(e, eh, nm) + LS.joint_loss(e, qh, nm)).backward()
    assert eh.grad.abs().sum() > 0 and qh.grad.abs().sum() > 0
