"""ACT 推理的 CUDA graph（`config.cuda_graph`）：动作与逐算子推理逐位相同。

用随机初始化的小 ACT，不需要存档；只需要 CUDA。
"""
import pytest
import torch

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
IMG = (3, 64, 80)


def tiny_config(**kw) -> ACTConfig:
    return ACTConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (7,)),
                        "observation.images.two": PolicyFeature(FeatureType.VISUAL, IMG)},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=10, dim_model=64, n_heads=2, dim_feedforward=128, n_encoder_layers=1,
        n_decoder_layers=1, latent_dim=8, n_vae_encoder_layers=1, pretrained_backbone_weights=None, **kw)


def tiny_policy(te: bool) -> ACTPolicy:
    torch.manual_seed(0)
    cfg = tiny_config(n_action_steps=1 if te else 10, temporal_ensemble_coeff=0.01 if te else None)
    stats = {"observation.state": {"mean": torch.randn(7), "std": torch.rand(7) + 0.5},
             "observation.images.two": {"mean": torch.rand(3, 1, 1), "std": torch.rand(3, 1, 1) + 0.5},
             "action": {"mean": torch.randn(7), "std": torch.rand(7) + 0.5}}
    return ACTPolicy(cfg, dataset_stats=stats).to("cuda").eval()


@pytest.mark.parametrize("te", [True, False], ids=["TE", "chunk"])
def test_graph_actions_equal_eager_bitwise(te):
    policy = tiny_policy(te)
    g = torch.Generator(device="cuda").manual_seed(1)
    frames = [{"observation.state": torch.randn(1, 7, device="cuda", generator=g),
               "observation.images.two": torch.rand(1, *IMG, device="cuda", generator=g),
               "observation.imu_real": torch.randn(1, 6, device="cuda", generator=g)}  # 不是输入：不进 graph
              for _ in range(25)]

    def run(graph: bool) -> torch.Tensor:
        policy.config.cuda_graph = graph
        policy.reset()
        with torch.inference_mode():
            return torch.stack([policy.select_action(f) for f in frames])

    eager = run(False)
    graphed = run(True)
    first = policy._cuda_graph.graph
    assert torch.equal(eager, graphed)
    assert run(True).equal(eager) and policy._cuda_graph.graph is first      # 多出的键不会触发重新录制


def test_graph_refuses_history_inputs():
    with pytest.raises(NotImplementedError):
        tiny_config(cuda_graph=True, enable_history_input=True)
