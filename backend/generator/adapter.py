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

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

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
        """Record user feedback on a generated image.
        One vote per image — replaces any existing vote for this content_hash.
        """
        # Remove existing vote for this image
        self.feedback = [f for f in self.feedback if f.content_hash != content_hash]

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
            log.info("Loaded adapter for story %s", self.story_id)
            return True
        except Exception as e:
            log.warning("Failed to load adapter: %s", e)
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
