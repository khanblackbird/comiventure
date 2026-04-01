"""LoRA bridge — converts adversarial adapter weights into diffusers-native
LoRA format for loading into the SDXL pipeline.

The adversarial adapter trains externally using feedback, latent comparison,
and review signals. Its matrices (A_visual/B_visual, A_language/B_language)
are already low-rank decompositions — exactly what LoRA expects.

This bridge:
1. Takes trained AdversarialAdapter weights
2. Maps them to diffusers LoRA state dict format (lora_down/lora_up)
3. Saves as safetensors
4. Loads via pipeline.load_lora_weights() — no peft, no UNet wrapping

The previous approach (peft's get_peft_model) wrapped the UNet in place,
which crashed with CPU offload because layers exist as meta tensors.
This approach never touches the UNet structure — just loads weights.
"""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional

from backend.generator.adversarial_adapter import AdversarialAdapter


UNET_ATTN_TARGETS = ("to_k", "to_q", "to_v")
TEXT_ENCODER_ATTN_TARGETS = ("k_proj", "q_proj", "v_proj")


class LoraBridge:
    """Converts adversarial adapter weights to LoRA format.

    The adapter's A_visual (down) and B_visual (up) map directly to
    LoRA's lora_down and lora_up. The sigmoid gate folds into the
    down projection as a scale factor.

    Only applies to layers whose dimensions match the adapter —
    mismatched layers are skipped safely.
    """

    def __init__(self, adapter: AdversarialAdapter) -> None:
        self.adapter = adapter

    def to_state_dict(self, pipeline) -> dict[str, torch.Tensor]:
        """Convert adapter weights to a LoRA state dict.

        Enumerates the pipeline's UNet and text encoder attention layers.
        For each layer with matching dimensions, creates lora_down/lora_up
        weight entries.
        """
        state_dict = {}

        visual_gate = torch.sigmoid(self.adapter.gate_visual).item()
        language_gate = torch.sigmoid(self.adapter.gate_language).item()

        # UNet attention layers — visual adapter
        self._collect_unet_keys(
            pipeline.unet,
            visual_gate,
            self.adapter.A_visual.weight.data,
            self.adapter.B_visual.weight.data,
            state_dict,
        )

        # Text encoder attention layers — language adapter
        self._collect_text_encoder_keys(
            pipeline.text_encoder,
            language_gate,
            self.adapter.A_language.weight.data,
            self.adapter.B_language.weight.data,
            state_dict,
        )

        return state_dict

    def _collect_unet_keys(
        self,
        unet,
        gate_scale: float,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        state_dict: dict,
    ) -> None:
        """Enumerate UNet attention layers and add matching LoRA keys."""
        hidden_dim = self.adapter.hidden_dim

        for module_name, module in unet.named_modules():
            for target in UNET_ATTN_TARGETS:
                layer = getattr(module, target, None)
                if layer is None:
                    continue
                if not hasattr(layer, "in_features"):
                    continue
                if layer.in_features != hidden_dim:
                    continue

                prefix = f"unet.{module_name}.{target}"
                # lora_down: [rank, hidden_dim] — A matrix scaled by gate
                state_dict[f"{prefix}.lora_down.weight"] = (
                    down_weight.clone() * gate_scale
                )
                # lora_up: [hidden_dim, rank] — B matrix
                state_dict[f"{prefix}.lora_up.weight"] = (
                    up_weight.clone()
                )

    def _collect_text_encoder_keys(
        self,
        text_encoder,
        gate_scale: float,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        state_dict: dict,
    ) -> None:
        """Enumerate text encoder attention layers and add matching LoRA keys."""
        hidden_dim = self.adapter.hidden_dim

        for module_name, module in text_encoder.named_modules():
            for target in TEXT_ENCODER_ATTN_TARGETS:
                layer = getattr(module, target, None)
                if layer is None:
                    continue
                if not hasattr(layer, "in_features"):
                    continue
                if layer.in_features != hidden_dim:
                    continue

                prefix = f"text_encoder.{module_name}.{target}"
                state_dict[f"{prefix}.lora_down.weight"] = (
                    down_weight.clone() * gate_scale
                )
                state_dict[f"{prefix}.lora_up.weight"] = (
                    up_weight.clone()
                )

    def save_safetensors(self, pipeline) -> bytes:
        """Convert to LoRA state dict and serialize as safetensors bytes."""
        from safetensors.torch import save as st_save
        state_dict = self.to_state_dict(pipeline)
        return st_save(state_dict)

    def load_into_pipeline(
        self,
        pipeline,
        tmp_path: Optional[str] = None,
    ) -> None:
        """Save adapter as safetensors and load into pipeline via
        diffusers' native load_lora_weights.

        No peft wrapping. No UNet mutation. Compatible with CPU offload.
        """
        import tempfile

        safetensors_bytes = self.save_safetensors(pipeline)

        if tmp_path is None:
            tmp_dir = tempfile.mkdtemp()
        else:
            tmp_dir = tmp_path

        adapter_path = Path(tmp_dir) / "adapter_model.safetensors"
        adapter_path.write_bytes(safetensors_bytes)

        pipeline.load_lora_weights(
            str(tmp_dir),
            local_files_only=True,
        )
