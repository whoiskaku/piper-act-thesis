#!/usr/bin/env python3
"""Step 4 verification — phase2_actor_critic.py pure-function correctness.

Hand-computed expected values vs code output. Catches math bugs that pass
shape/dtype checks but produce wrong gradients.

Run: python scripts/test_phase2_actor_critic.py
"""
from __future__ import annotations

import math
import torch

from lerobot.common.policies.act.phase2_actor_critic import (
    factorized_gaussian_log_prob,
    kl_gaussian_fixed_sigma,
    compute_gae,
    ppo_clip_loss,
    value_loss,
    standardize_advantages,
)


def aclose(a, b, tol=1e-4):
    return abs(float(a) - float(b)) < tol


def test_log_prob_at_mean():
    """log p at action=μ equals the Gaussian normalizer."""
    B, T, A = 2, 5, 3
    mu = torch.zeros(B, T, A)
    sigma = torch.ones(A) * 0.1
    lp = factorized_gaussian_log_prob(mu, mu, sigma)
    expected = -0.5 * T * A * (math.log(2 * math.pi) + 2 * math.log(0.1))
    assert aclose(lp[0].item(), expected, tol=1e-3)


def test_log_prob_decreases_with_deviation():
    mu = torch.zeros(1, 5, 3)
    sigma = torch.ones(3) * 0.1
    lp_at_mean = factorized_gaussian_log_prob(mu, mu, sigma)
    lp_far = factorized_gaussian_log_prob(mu + 0.1, mu, sigma)
    assert lp_far[0].item() < lp_at_mean[0].item()


def test_log_prob_factorization_explicit():
    """Hand-compute element-wise log p, sum, compare."""
    B, T, A = 1, 2, 3
    mu = torch.zeros(B, T, A)
    sigma = torch.tensor([0.1, 0.2, 0.5])
    action = torch.tensor([[[0.1, 0.0, 0.5], [0.0, 0.2, 0.0]]])
    expected_terms = []
    for a_val, s_val in zip([0.1, 0, 0.5, 0, 0.2, 0],
                             [0.1, 0.2, 0.5, 0.1, 0.2, 0.5]):
        t = -0.5 * (a_val ** 2 / s_val ** 2 + math.log(2 * math.pi * s_val ** 2))
        expected_terms.append(t)
    expected_total = sum(expected_terms)
    lp = factorized_gaussian_log_prob(action, mu, sigma)
    assert aclose(lp[0].item(), expected_total, tol=1e-3)


def test_kl_identity():
    mu = torch.randn(2, 5, 3)
    sigma = torch.ones(3) * 0.1
    kl = kl_gaussian_fixed_sigma(mu, mu.clone(), sigma)
    assert kl[0].item() == 0.0
    assert kl[1].item() == 0.0


def test_kl_quadratic_form():
    """KL = 0.5 × Σ ((μ_new - μ_old)/σ)²"""
    B, T, A = 1, 5, 3
    mu_new = torch.zeros(B, T, A) + 0.1
    mu_old = torch.zeros(B, T, A)
    sigma = torch.ones(A) * 0.1
    kl = kl_gaussian_fixed_sigma(mu_new, mu_old, sigma)
    expected = 0.5 * T * A * (0.1 / 0.1) ** 2
    assert aclose(kl[0].item(), expected, tol=1e-3)


def test_gae_handcomputed_basic():
    """T=3 hand-computed. γ=0.99, λ=0.95."""
    gamma, lam = 0.99, 0.95
    rewards = torch.tensor([0.1, 0.2, 0.3])
    values = torch.tensor([0.5, 0.4, 0.3, 0.0])
    valid = torch.ones(3, dtype=torch.bool)

    # δ_0 = 0.1 + 0.99·0.4 - 0.5 = -0.004
    # δ_1 = 0.2 + 0.99·0.3 - 0.4 = 0.097
    # δ_2 = 0.3 + 0.99·0.0 - 0.3 = 0.0
    # A_2 = 0
    # A_1 = 0.097 + 0.9405·0 = 0.097
    # A_0 = -0.004 + 0.9405·0.097 = 0.0872285
    expected_adv = [0.0872285, 0.097, 0.0]
    expected_ret = [0.5872285, 0.497, 0.3]

    adv, ret = compute_gae(rewards, values, valid, gamma, lam)
    for i in range(3):
        assert aclose(adv[i].item(), expected_adv[i])
        assert aclose(ret[i].item(), expected_ret[i])


def test_gae_invalid_step_does_not_pollute():
    """Outlier reward at invalid step should NOT propagate via GAE."""
    gamma, lam = 0.99, 0.95
    rewards_outlier = torch.tensor([0.1, 999.0, 0.3])
    values = torch.tensor([0.5, 0.4, 0.3, 0.0])
    valid = torch.tensor([True, False, True])

    # For invalid t=1: bootstrap reward = V(s_1) - γV(s_2) = 0.4 - 0.297 = 0.103
    # δ_1 = 0.103 + 0.297 - 0.4 = 0.0  (by design)
    # δ_0 = -0.004 unchanged
    # A_2 = 0, A_1 = 0, A_0 = -0.004 (no propagation from t=1)
    expected_adv = [-0.004, 0.0, 0.0]
    expected_ret = [-0.004 + 0.5, 0.4, 0.3]  # ret[1] falls back to V[1]

    adv, ret = compute_gae(rewards_outlier, values, valid, gamma, lam)
    for i in range(3):
        assert aclose(adv[i].item(), expected_adv[i])
        assert aclose(ret[i].item(), expected_ret[i])

    # Critically: A_0 with outlier ≈ A_0 without it (just the base δ_0)
    # If pollution occurred, A_0 would be ~939 (γλ × 999)
    assert abs(adv[0].item()) < 1.0


def test_ppo_clip_handcomputed():
    """PPO clip loss with manual numbers — verify clip + min + mean math."""
    log_prob_old = torch.tensor([0.0, 0.0, 0.0])
    log_prob_new = torch.tensor([0.05, 0.3, -0.5])
    advantages = torch.tensor([1.0, 1.0, -1.0])
    valid = torch.ones(3, dtype=torch.bool)
    eps = 0.2

    # ratio: [exp(0.05), exp(0.3), exp(-0.5)] = [1.0513, 1.3499, 0.6065]
    # clipped: [1.0513, 1.2, 0.8]   (1.3499 > 1.2 → clip; 0.6065 < 0.8 → clip)
    # surr1: ratio × A = [1.0513, 1.3499, -0.6065]
    # surr2: clipped × A = [1.0513, 1.2, -0.8]
    # raw_loss = -min(surr1, surr2) = [-1.0513, -1.2, 0.8]
    # mean = -0.48377
    ratios = [math.exp(0.05), math.exp(0.3), math.exp(-0.5)]
    surr1 = [ratios[i] * advantages[i].item() for i in range(3)]
    clipped = [max(0.8, min(1.2, r)) for r in ratios]
    surr2 = [clipped[i] * advantages[i].item() for i in range(3)]
    raw_per_step = [-min(s1, s2) for s1, s2 in zip(surr1, surr2)]
    expected_loss = sum(raw_per_step) / 3

    loss, m = ppo_clip_loss(log_prob_new, log_prob_old, advantages, valid, eps)
    assert aclose(loss.item(), expected_loss, tol=1e-4)
    assert aclose(m["ratio_clip_frac"], 2 / 3, tol=1e-3)


def test_ppo_clip_valid_mask_divisor():
    """When some steps are invalid, divisor is n_valid not n_total."""
    log_prob_old = torch.zeros(6)
    log_prob_new = torch.zeros(6)  # ratio=1 for all
    advantages = torch.ones(6)
    valid_partial = torch.tensor([True, True, False, False, True, True])
    loss, m = ppo_clip_loss(log_prob_new, log_prob_old, advantages,
                             valid_partial, 0.2)
    # All valid: ratio=1, surr=1×1=1, loss = -1
    # n_valid = 4, divisor = 4 → loss = -4/4 = -1
    assert aclose(loss.item(), -1.0)
    assert m["n_valid"] == 4


def test_value_loss_handcomputed():
    V_pred = torch.tensor([0.5, 0.6, 0.7])
    returns = torch.tensor([0.6, 0.5, 0.8])
    valid = torch.ones(3, dtype=torch.bool)
    # (V-r)² = [0.01, 0.01, 0.01], mean = 0.01
    loss = value_loss(V_pred, returns, valid)
    assert aclose(loss.item(), 0.01)


def test_value_loss_outlier_at_invalid_ignored():
    V_pred = torch.tensor([0.5, 0.6, 0.7])
    returns = torch.tensor([0.6, 100.0, 0.8])  # huge invalid
    valid = torch.tensor([True, False, True])
    # Only valid: ((0.5-0.6)² + (0.7-0.8)²) / 2 = 0.01
    loss = value_loss(V_pred, returns, valid)
    assert aclose(loss.item(), 0.01)


def test_standardize_advantages():
    """After standardization, valid-step mean ≈ 0, std ≈ 1; invalid → 0."""
    adv = torch.tensor([1., 2., 0., 3., -10., 4.])
    valid = torch.tensor([True, True, False, True, False, True])
    std_adv = standardize_advantages(adv, valid)
    valid_vals = std_adv[valid]
    assert aclose(valid_vals.mean().item(), 0.0, tol=1e-4)
    assert aclose(valid_vals.std(unbiased=False).item(), 1.0, tol=1e-3)
    assert std_adv[2].item() == 0.0
    assert std_adv[4].item() == 0.0


if __name__ == "__main__":
    tests = [
        test_log_prob_at_mean,
        test_log_prob_decreases_with_deviation,
        test_log_prob_factorization_explicit,
        test_kl_identity,
        test_kl_quadratic_form,
        test_gae_handcomputed_basic,
        test_gae_invalid_step_does_not_pollute,
        test_ppo_clip_handcomputed,
        test_ppo_clip_valid_mask_divisor,
        test_value_loss_handcomputed,
        test_value_loss_outlier_at_invalid_ignored,
        test_standardize_advantages,
    ]
    print(f"Running {len(tests)} numerical correctness tests...")
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            raise
    print(f"\n✓ ALL {len(tests)} TESTS PASS")
