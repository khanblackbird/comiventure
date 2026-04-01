"""Story adapter — a trainable LoRA layer on top of the frozen base model.

Each story gets its own adapter weights trained on user feedback:
- Accepted images (thumbs up) = positive training signal
- Rejected images (thumbs down) = negative training signal
- Character reference banks contribute accepted images + captions

The adapter learns the story's visual style, character consistency,
and user preferences. Weights are small (~50-200MB) and stored in
the ContentStore by hash.

Architecture:
  Frozen SDXL base (knows how to draw)
    ↑
  LoRA adapter (learned from this story's feedback)
    ↑
  LLM prompt composer (translates hierarchy context into clean prompts)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from backend.models.content_store import ContentStore
from backend.models.story import Story
from backend.models.emitter import Emitter


@dataclass
class FeedbackEntry:
    """One piece of user feedback on a generated image."""
    content_hash: str           # the image in ContentStore
    prompt: str                 # the prompt that generated it
    accepted: bool              # thumbs up (True) or down (False)
    character_ids: list[str] = field(default_factory=list)
    panel_id: str = ""
    negative_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "content_hash": self.content_hash,
            "prompt": self.prompt,
            "accepted": self.accepted,
            "character_ids": self.character_ids,
            "panel_id": self.panel_id,
            "negative_prompt": self.negative_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackEntry:
        return cls(
            content_hash=data["content_hash"],
            prompt=data.get("prompt", ""),
            accepted=data.get("accepted", True),
            character_ids=data.get("character_ids", []),
            panel_id=data.get("panel_id", ""),
            negative_prompt=data.get("negative_prompt", ""),
        )


class StoryAdapter(Emitter):
    """Manages the LoRA adapter for a specific story.

    Collects feedback, trains when enough data accumulates,
    loads/saves adapter weights via ContentStore.
    """

    def __init__(self, story_id: str, content_store: ContentStore) -> None:
        super().__init__()
        self.story_id = story_id
        self.content_store = content_store
        self.feedback: list[FeedbackEntry] = []
        self.adapter_hash: Optional[str] = None  # trained weights in ContentStore
        self.is_training = False
        self.training_epochs = 5
        self.lora_rank = 4
        self.learning_rate = 1e-4
        self.min_training_samples = 5

    def add_feedback(self, content_hash: str, prompt: str, accepted: bool,
                     character_ids: list[str] = None, panel_id: str = "",
                     negative_prompt: str = "") -> FeedbackEntry:
        """Record user feedback on a generated image."""
        entry = FeedbackEntry(
            content_hash=content_hash,
            prompt=prompt,
            accepted=accepted,
            character_ids=character_ids or [],
            panel_id=panel_id,
            negative_prompt=negative_prompt,
        )
        self.feedback.append(entry)
        self.emit("feedback_added", entry)
        return entry

    def positive_samples(self) -> list[FeedbackEntry]:
        return [f for f in self.feedback if f.accepted]

    def negative_samples(self) -> list[FeedbackEntry]:
        return [f for f in self.feedback if not f.accepted]

    def can_train(self) -> bool:
        """Need enough positive samples to train."""
        return len(self.positive_samples()) >= self.min_training_samples

    async def train(self, base_pipeline) -> Optional[str]:
        """Train the LoRA adapter using collected feedback.

        Returns the content hash of the saved adapter weights,
        or None if training failed/not enough data.
        """
        if not self.can_train():
            return None

        if self.is_training:
            return None

        self.is_training = True
        self.emit("training_started", {"story_id": self.story_id})

        try:
            adapter_bytes = await asyncio.to_thread(
                self._train_lora,
                base_pipeline,
            )

            if adapter_bytes:
                self.adapter_hash = self.content_store.store(
                    adapter_bytes,
                    "application/octet-stream",
                    metadata={
                        "type": "lora_adapter",
                        "story_id": self.story_id,
                        "positive_samples": len(self.positive_samples()),
                        "negative_samples": len(self.negative_samples()),
                        "rank": self.lora_rank,
                        "epochs": self.training_epochs,
                    },
                )
                self.emit("training_complete", {
                    "story_id": self.story_id,
                    "adapter_hash": self.adapter_hash,
                })
                return self.adapter_hash

        except Exception as e:
            print(f"Adapter training failed: {e}")
            self.emit("training_failed", {"error": str(e)})
        finally:
            self.is_training = False

        return None

    def _train_lora(self, base_pipeline):
        """Synchronous LoRA training — runs in a thread.

        Training happens through the adversarial adapter. The trained
        weights are then converted to LoRA format via LoraBridge and
        loaded into the pipeline using diffusers' native load_lora_weights.

        This avoids peft's get_peft_model which corrupts the UNet with
        CPU offload (meta tensor crash).
        """
        if not hasattr(self, '_adversarial') or self._adversarial is None:
            print("LoRA training skipped — no adversarial adapter")
            return None

        from backend.generator.lora_bridge import LoraBridge
        bridge = LoraBridge(self._adversarial)
        bridge.load_into_pipeline(base_pipeline)
        print(f"LoRA weights loaded from adversarial adapter (rank={self._adversarial.rank})")
        return self._adversarial.save_weights()

    def _train_lora_DISABLED(self, base_pipeline):
        """Original LoRA training — kept for reference, do not call."""
        import torch
        from io import BytesIO
        from PIL import Image
        from peft import LoraConfig, get_peft_model

        positive = self.positive_samples()
        if not positive:
            return None

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.05,
        )

        # Get the UNet and add LoRA
        unet = base_pipeline.unet
        unet.requires_grad_(False)
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=self.learning_rate,
        )

        # Training loop
        unet.train()
        for epoch in range(self.training_epochs):
            total_loss = 0

            for sample in positive:
                image_bytes = self.content_store.retrieve(sample.content_hash)
                if not image_bytes:
                    continue

                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = image.resize((512, 512))

                # Encode the image through the VAE
                with torch.no_grad():
                    from diffusers.image_processor import VaeImageProcessor
                    processor = VaeImageProcessor()
                    pixel_values = processor.preprocess(image)
                    pixel_values = pixel_values.to(
                        device=base_pipeline.vae.device,
                        dtype=base_pipeline.vae.dtype,
                    )
                    latents = base_pipeline.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * base_pipeline.vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (1,), device=latents.device)
                noisy_latents = base_pipeline.scheduler.add_noise(latents, noise, timesteps)

                # Encode the prompt
                with torch.no_grad():
                    prompt_embeds = base_pipeline.encode_prompt(sample.prompt)
                    if isinstance(prompt_embeds, tuple):
                        prompt_embeds = prompt_embeds[0]

                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # MSE loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / max(len(positive), 1)
            print(f"  Epoch {epoch + 1}/{self.training_epochs}, loss: {avg_loss:.4f}")

        # Save LoRA weights
        buffer = BytesIO()
        unet.save_pretrained(buffer)
        adapter_bytes = buffer.getvalue()

        # Restore UNet to base state
        unet = unet.merge_and_unload()
        base_pipeline.unet = unet

        return adapter_bytes

    def load_adapter(self, base_pipeline) -> bool:
        """Load trained adapter weights into the pipeline.

        Uses LoraBridge to convert adversarial adapter weights to
        diffusers-native LoRA format. No peft wrapping.
        """
        if not self.adapter_hash:
            return False

        adapter_bytes = self.content_store.retrieve(self.adapter_hash)
        if not adapter_bytes:
            return False

        try:
            from backend.generator.adversarial_adapter import AdversarialAdapter
            from backend.generator.lora_bridge import LoraBridge

            adversarial = AdversarialAdapter.load_weights(adapter_bytes)
            bridge = LoraBridge(adversarial)
            bridge.load_into_pipeline(base_pipeline)

            self._adversarial = adversarial
            print(f"Loaded adapter for story {self.story_id}")
            return True
        except Exception as e:
            print(f"Failed to load adapter: {e}")
            return False

    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "adapter_hash": self.adapter_hash,
            "feedback": [f.to_dict() for f in self.feedback],
            "training_epochs": self.training_epochs,
            "lora_rank": self.lora_rank,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: dict, content_store: ContentStore) -> StoryAdapter:
        adapter = cls(data["story_id"], content_store)
        adapter.adapter_hash = data.get("adapter_hash")
        adapter.feedback = [FeedbackEntry.from_dict(f) for f in data.get("feedback", [])]
        adapter.training_epochs = data.get("training_epochs", 5)
        adapter.lora_rank = data.get("lora_rank", 4)
        adapter.learning_rate = data.get("learning_rate", 1e-4)
        return adapter
