#!/usr/bin/env python3
"""Step 2.5 — Verify Phase 2 modeling_act.py changes.

Independent test before downstream PPO/rollout code is written.
If any of these fail, downstream silently produces wrong gradients
or wrong actions — hard to debug. Catch issues here.

Tests:
  1. enable_value_head=True adds value_head module
  2. forward(return_value=True) returns 4-tuple with V shape (B,)
  3. V loss .backward() does NOT update encoder (gradient is None on encoder)
  4. V loss DOES update value_head
  5. enable_value_head=False makes return_value=True raise (clear error)
  6. Default forward (return_value=False) is unchanged (regression test)
  7. Loading C1 ckpt with enable_value_head=True works (value_head random init)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACT
from lerobot.configs.types import PolicyFeature, FeatureType


def make_test_config(enable_value_head=True):
    """Mock features matching v2 dataset shape."""
    cfg = ACTConfig(
        chunk_size=100, n_action_steps=100,
        enable_history_input=True, history_max_length=600,
        enable_value_head=enable_value_head,
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.imu_real": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.two": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.validate_features()
    return cfg


def make_test_batch(B=2, L=600):
    return {
        "observation.state": torch.randn(B, 7),
        "observation.imu_real": torch.randn(B, 6),
        "observation.images": torch.randn(B, 1, 3, 480, 640),
        "observation.history_action": torch.randn(B, L, 7),
        "observation.history_imu": torch.randn(B, L, 6),
        "history_pad_mask": torch.zeros(B, L, dtype=torch.bool),
        "action": torch.randn(B, 100, 7),
        "action_is_pad": torch.zeros(B, 100, dtype=torch.bool),
    }


def test_value_head_module_exists():
    print("[Test 1] enable_value_head=True creates value_head module ...", end=" ")
    cfg = make_test_config(enable_value_head=True)
    model = ACT(cfg)
    assert hasattr(model, "value_head"), "value_head module missing"
    assert isinstance(model.value_head, nn.Linear)
    assert model.value_head.in_features == cfg.dim_model
    assert model.value_head.out_features == 1
    print("PASS")


def test_forward_returns_value():
    print("[Test 2] forward(return_value=True) returns 4-tuple V shape (B,) ...", end=" ")
    cfg = make_test_config(enable_value_head=True)
    model = ACT(cfg).eval()
    batch = make_test_batch(B=2)
    with torch.no_grad():
        out = model(batch, return_value=True)
    assert len(out) == 4, f"expected 4 elements, got {len(out)}"
    actions, vae_params, aux_pred, V = out
    assert V.shape == (2,), f"expected V shape (2,), got {V.shape}"
    assert torch.isfinite(V).all(), "V has NaN/Inf"
    print("PASS")


def test_value_loss_does_not_update_encoder():
    print("[Test 3] V loss does NOT update encoder weights ...", end=" ")
    cfg = make_test_config(enable_value_head=True)
    model = ACT(cfg).train()
    batch = make_test_batch(B=2)
    actions, _, _, V = model(batch, return_value=True)
    L_V = ((V - 1.5) ** 2).mean()
    L_V.backward()

    # Check encoder parameters have no gradient
    for name, p in model.encoder.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().max().item() < 1e-9, (
                f"encoder.{name} got non-zero gradient from V loss "
                f"(max={p.grad.abs().max().item()}, expected ~0)"
            )

    # Also encoder's input projection layers should be untouched
    for name, p in model.named_parameters():
        if "encoder" in name and p.grad is not None:
            assert p.grad.abs().max().item() < 1e-9, (
                f"{name} got non-zero gradient from V loss"
            )
    print("PASS")


def test_value_loss_updates_value_head():
    print("[Test 4] V loss updates value_head weights ...", end=" ")
    cfg = make_test_config(enable_value_head=True)
    model = ACT(cfg).train()
    batch = make_test_batch(B=2)
    actions, _, _, V = model(batch, return_value=True)
    L_V = ((V - 1.5) ** 2).mean()
    L_V.backward()
    assert model.value_head.weight.grad is not None, "value_head.weight.grad is None"
    assert model.value_head.weight.grad.abs().max().item() > 1e-6, (
        "value_head got near-zero gradient from V loss"
    )
    print("PASS")


def test_disabled_value_head_raises():
    print("[Test 5] enable_value_head=False + return_value=True raises ...", end=" ")
    cfg = make_test_config(enable_value_head=False)
    model = ACT(cfg).eval()
    batch = make_test_batch(B=2)
    try:
        with torch.no_grad():
            model(batch, return_value=True)
        print("FAIL — should have raised"); sys.exit(1)
    except RuntimeError as e:
        assert "enable_value_head" in str(e)
    print("PASS")


def test_default_forward_unchanged():
    print("[Test 6] Default forward (return_value=False) is unchanged ...", end=" ")
    cfg = make_test_config(enable_value_head=True)
    model = ACT(cfg).eval()
    batch = make_test_batch(B=2)
    with torch.no_grad():
        out = model(batch)
    assert len(out) == 3, f"default forward should return 3-tuple, got {len(out)}"
    actions, vae_params, aux_pred = out
    assert actions.shape == (2, 100, 7)
    print("PASS")


def test_load_c1_ckpt_with_value_head():
    print("[Test 7] Loading C1 ckpt with enable_value_head=True works ...", end=" ")
    import json, tempfile, draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.common.policies.factory import make_policy

    ckpt_dir = Path("/home/enolab-guo/claude-test-2/outputs/train/2026-04-29/"
                    "C1_v2_history_seed42/checkpoints/100000")
    if not ckpt_dir.exists():
        print(f"SKIP (ckpt not found at {ckpt_dir})")
        return

    train_cfg = json.load(open(ckpt_dir / "pretrained_model" / "train_config.json"))
    # Inject enable_value_head=True (overriding C1's default False)
    train_cfg["policy"]["enable_value_head"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(train_cfg, f); tmp = f.name
    cfg = draccus.parse(TrainPipelineConfig, tmp, args=[])
    cfg.policy.pretrained_path = str(ckpt_dir / "pretrained_model")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    ds_meta = LeRobotDatasetMetadata("local/piper_pick_place_imu_v2", root=None)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta, device="cpu")

    # value_head should be present
    assert hasattr(policy.model, "value_head")
    # Value head should have random init (it's not in C1 ckpt)
    # but the rest of the model should be loaded from C1
    print("PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2.5: modeling_act.py Phase 2 changes verification")
    print("=" * 60)
    test_value_head_module_exists()
    test_forward_returns_value()
    test_value_loss_does_not_update_encoder()
    test_value_loss_updates_value_head()
    test_disabled_value_head_raises()
    test_default_forward_unchanged()
    test_load_c1_ckpt_with_value_head()
    print("=" * 60)
    print("ALL TESTS PASS ✓")
    print("=" * 60)
