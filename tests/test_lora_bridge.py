"""Tests for the LoRA bridge — converts adversarial adapter weights
into diffusers-compatible LoRA format for pipeline loading.

TDD: tests written first, implementation follows.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch

from backend.generator.adversarial_adapter import AdversarialAdapter
from backend.generator.lora_bridge import LoraBridge


@pytest.fixture
def adapter():
    """A trained adversarial adapter with non-trivial weights."""
    adapter = AdversarialAdapter(hidden_dim=64, rank=4, gate_init=1.0)
    # Simulate training by setting non-zero weights
    with torch.no_grad():
        adapter.A_visual.weight.fill_(0.1)
        adapter.B_visual.weight.fill_(0.2)
        adapter.A_language.weight.fill_(0.15)
        adapter.B_language.weight.fill_(0.25)
        adapter.gate_visual.fill_(2.0)  # sigmoid(2) ≈ 0.88
        adapter.gate_language.fill_(2.0)
    return adapter


@pytest.fixture
def bridge(adapter):
    return LoraBridge(adapter)


@pytest.fixture
def mock_unet():
    """Mock UNet with attention layers matching adapter hidden_dim."""
    unet = MagicMock()

    # Create fake attention modules with matching dimensions
    attn_module = MagicMock()
    to_k = MagicMock()
    to_k.in_features = 64
    to_k.out_features = 64
    to_q = MagicMock()
    to_q.in_features = 64
    to_q.out_features = 64
    to_v = MagicMock()
    to_v.in_features = 64
    to_v.out_features = 64
    to_out = MagicMock()
    to_out.in_features = 64
    to_out.out_features = 64

    attn_module.to_k = to_k
    attn_module.to_q = to_q
    attn_module.to_v = to_v
    to_out_container = MagicMock()
    to_out_container.__getitem__ = MagicMock(return_value=to_out)

    # named_modules returns (name, module) tuples
    unet.named_modules.return_value = [
        ("down_blocks.0.attentions.0.transformer_blocks.0.attn1", attn_module),
    ]

    return unet


@pytest.fixture
def mock_pipeline(mock_unet):
    pipeline = MagicMock()
    pipeline.unet = mock_unet

    # Mock text_encoder similarly
    text_attn = MagicMock()
    text_layer = MagicMock()
    text_layer.in_features = 64
    text_layer.out_features = 64
    text_attn.k_proj = text_layer
    text_attn.v_proj = text_layer
    text_attn.q_proj = text_layer
    text_attn.out_proj = text_layer

    pipeline.text_encoder = MagicMock()
    pipeline.text_encoder.named_modules.return_value = [
        ("text_model.encoder.layers.0.self_attn", text_attn),
    ]
    return pipeline


class TestLoraBridgeStateDict:
    def test_state_dict_has_lora_keys(self, bridge, mock_pipeline):
        """State dict must contain lora_down.weight and lora_up.weight keys."""
        state_dict = bridge.to_state_dict(mock_pipeline)
        assert len(state_dict) > 0
        for key in state_dict:
            assert "lora_down.weight" in key or "lora_up.weight" in key, (
                f"Key {key} doesn't follow LoRA naming convention"
            )

    def test_state_dict_keys_come_in_pairs(self, bridge, mock_pipeline):
        """Every lora_down must have a matching lora_up."""
        state_dict = bridge.to_state_dict(mock_pipeline)
        down_keys = {k for k in state_dict if "lora_down" in k}
        up_keys = {k for k in state_dict if "lora_up" in k}
        assert len(down_keys) == len(up_keys)
        for down_key in down_keys:
            up_key = down_key.replace("lora_down", "lora_up")
            assert up_key in up_keys

    def test_down_weight_shape_is_rank_by_hidden(self, bridge, mock_pipeline):
        """lora_down projects from hidden_dim to rank."""
        state_dict = bridge.to_state_dict(mock_pipeline)
        for key, tensor in state_dict.items():
            if "lora_down" in key:
                assert tensor.shape[0] == 4, f"down rank dim should be 4, got {tensor.shape[0]}"
                assert tensor.shape[1] == 64, f"down hidden dim should be 64, got {tensor.shape[1]}"

    def test_up_weight_shape_is_hidden_by_rank(self, bridge, mock_pipeline):
        """lora_up projects from rank back to hidden_dim."""
        state_dict = bridge.to_state_dict(mock_pipeline)
        for key, tensor in state_dict.items():
            if "lora_up" in key:
                assert tensor.shape[0] == 64, f"up hidden dim should be 64, got {tensor.shape[0]}"
                assert tensor.shape[1] == 4, f"up rank dim should be 4, got {tensor.shape[1]}"

    def test_visual_weights_scaled_by_gate(self, adapter, bridge, mock_pipeline):
        """Visual down weights include the sigmoid gate scaling."""
        state_dict = bridge.to_state_dict(mock_pipeline)
        gate_scale = torch.sigmoid(adapter.gate_visual).item()
        unet_down_keys = [k for k in state_dict if "lora_down" in k and "unet" in k.lower()]

        # At least one UNet key should exist
        assert len(unet_down_keys) > 0
        for key in unet_down_keys:
            weight = state_dict[key]
            # A_visual.weight is [rank, hidden], gate scales it
            expected = adapter.A_visual.weight.data * gate_scale
            assert torch.allclose(weight, expected, atol=1e-6)


class TestLoraBridgeDimensionHandling:
    def test_projects_mismatched_dimensions(self, adapter):
        """Layers with different dims get projected LoRA weights."""
        bridge = LoraBridge(adapter)

        unet = MagicMock()
        attn = MagicMock()
        layer = MagicMock()
        layer.in_features = 1280  # different from adapter's 64
        layer.out_features = 1280
        attn.to_k = layer
        attn.to_q = layer
        attn.to_v = layer

        unet.named_modules.return_value = [
            ("down_blocks.0.attentions.0.attn1", attn),
        ]

        pipeline = MagicMock()
        pipeline.unet = unet
        pipeline.text_encoder = MagicMock()
        pipeline.text_encoder.named_modules.return_value = []

        state_dict = bridge.to_state_dict(pipeline)
        unet_keys = [k for k in state_dict if "unet" in k.lower()]
        assert len(unet_keys) > 0
        # Projected down weight should be [rank, 1280]
        down_key = [k for k in unet_keys if "lora_down" in k][0]
        assert state_dict[down_key].shape[1] == 1280

    def test_all_layers_get_weights(self, adapter):
        """All attention layers get LoRA weights regardless of dim."""
        bridge = LoraBridge(adapter)

        unet = MagicMock()

        attn_64 = MagicMock()
        layer_64 = MagicMock()
        layer_64.in_features = 64
        attn_64.to_k = layer_64
        attn_64.to_q = layer_64
        attn_64.to_v = layer_64

        attn_256 = MagicMock()
        layer_256 = MagicMock()
        layer_256.in_features = 256
        attn_256.to_k = layer_256
        attn_256.to_q = layer_256
        attn_256.to_v = layer_256

        unet.named_modules.return_value = [
            ("block.0.attn1", attn_64),
            ("block.1.attn1", attn_256),
        ]

        pipeline = MagicMock()
        pipeline.unet = unet
        pipeline.text_encoder = MagicMock()
        pipeline.text_encoder.named_modules.return_value = []

        state_dict = bridge.to_state_dict(pipeline)
        keys = list(state_dict.keys())
        assert any("block.0" in k for k in keys)
        assert any("block.1" in k for k in keys)


class TestLoraBridgeSaveLoad:
    def test_save_produces_bytes(self, bridge, mock_pipeline):
        """save_safetensors returns non-empty bytes."""
        data = bridge.save_safetensors(mock_pipeline)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_save_is_valid_safetensors(self, bridge, mock_pipeline):
        """Saved bytes can be loaded by safetensors."""
        from safetensors.torch import load as st_load
        data = bridge.save_safetensors(mock_pipeline)
        loaded = st_load(data)
        assert len(loaded) > 0
        for key, tensor in loaded.items():
            assert isinstance(tensor, torch.Tensor)

    def test_load_into_pipeline_calls_load_lora(self, bridge, mock_pipeline, tmp_path):
        """load_into_pipeline uses diffusers' load_lora_weights, not peft."""
        bridge.load_into_pipeline(mock_pipeline, tmp_path=str(tmp_path))
        mock_pipeline.load_lora_weights.assert_called_once()

    def test_load_does_not_use_peft(self):
        """The bridge must never import or use peft — that's what broke before."""
        import inspect
        source = inspect.getsource(LoraBridge)
        assert "get_peft_model" not in source
        assert "PeftModel" not in source


class TestLoraBridgeGateBehavior:
    def test_zero_gate_produces_near_zero_weights(self):
        """With gate_init=0, sigmoid(0)=0.5, weights are halved."""
        adapter = AdversarialAdapter(hidden_dim=32, rank=2, gate_init=0.0)
        with torch.no_grad():
            adapter.A_visual.weight.fill_(1.0)
        bridge = LoraBridge(adapter)

        unet = MagicMock()
        attn = MagicMock()
        layer = MagicMock()
        layer.in_features = 32
        layer.out_features = 32
        attn.to_k = layer
        attn.to_q = layer
        attn.to_v = layer
        unet.named_modules.return_value = [("block.attn1", attn)]

        pipeline = MagicMock()
        pipeline.unet = unet
        pipeline.text_encoder = MagicMock()
        pipeline.text_encoder.named_modules.return_value = []

        state_dict = bridge.to_state_dict(pipeline)
        for key, tensor in state_dict.items():
            if "lora_down" in key and "unet" in key.lower():
                # sigmoid(0) = 0.5, so weights should be 0.5
                assert torch.allclose(tensor, torch.full_like(tensor, 0.5), atol=1e-6)

    def test_large_negative_gate_produces_near_zero_weights(self):
        """With very negative gate, sigmoid approaches 0, weights vanish."""
        adapter = AdversarialAdapter(hidden_dim=32, rank=2, gate_init=-10.0)
        with torch.no_grad():
            adapter.A_visual.weight.fill_(1.0)
        bridge = LoraBridge(adapter)

        unet = MagicMock()
        attn = MagicMock()
        layer = MagicMock()
        layer.in_features = 32
        layer.out_features = 32
        attn.to_k = layer
        attn.to_q = layer
        attn.to_v = layer
        unet.named_modules.return_value = [("block.attn1", attn)]

        pipeline = MagicMock()
        pipeline.unet = unet
        pipeline.text_encoder = MagicMock()
        pipeline.text_encoder.named_modules.return_value = []

        state_dict = bridge.to_state_dict(pipeline)
        for key, tensor in state_dict.items():
            if "lora_down" in key:
                assert tensor.abs().max() < 0.01
