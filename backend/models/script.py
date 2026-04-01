from __future__ import annotations

from typing import Optional

from .emitter import Emitter


class Script(Emitter):
    """A character's individual contribution to a panel.

    The most granular data point — dialogue, actions, stage direction
    for one character in one panel. Like a screenplay entry.

    Emits 'script_updated' upward when content changes.
    Receives inherited context from Panel -> Page -> Chapter -> Character.
    """

    def __init__(
        self,
        script_id: str,
        character_id: str,
        dialogue: str = "",
        action: str = "",
        direction: str = "",
        emotion: str = "",
        pose: str = "",
        outfit: str = "",
    ) -> None:
        super().__init__()
        self.script_id = script_id
        self.character_id = character_id
        self.dialogue = dialogue
        self.action = action
        self.direction = direction
        self.emotion = emotion
        self.pose = pose      # "standing", "sitting", "crouching", "leaning"
        self.outfit = outfit  # overrides character default: "armor", "casual", "formal"
        self.source: str = "empty"  # 'empty', 'ai', 'manual'

    def update(
        self,
        dialogue: Optional[str] = None,
        action: Optional[str] = None,
        direction: Optional[str] = None,
        emotion: Optional[str] = None,
        pose: Optional[str] = None,
        outfit: Optional[str] = None,
        source: str = "manual",
    ) -> None:
        """Update script content and emit upward."""
        if dialogue is not None:
            self.dialogue = dialogue
        if action is not None:
            self.action = action
        if direction is not None:
            self.direction = direction
        if emotion is not None:
            self.emotion = emotion
        if pose is not None:
            self.pose = pose
        if outfit is not None:
            self.outfit = outfit
        self.source = source
        self.emit_up("script_updated", self)

    def _own_context(self) -> dict:
        return {
            "script": {
                "character_id": self.character_id,
                "dialogue": self.dialogue,
                "action": self.action,
                "direction": self.direction,
                "emotion": self.emotion,
                "pose": self.pose,
                "outfit": self.outfit,
            }
        }

    def to_prompt(self) -> str:
        """Compose this script into a text prompt for image generation.
        Dialogue is excluded — it gets overlaid as speech bubbles instead.
        """
        parts = []
        if self.pose:
            parts.append(self.pose)
        if self.action:
            parts.append(self.action)
        if self.emotion:
            parts.append(f"({self.emotion})")
        if self.outfit:
            parts.append(f"wearing {self.outfit}")
        if self.direction:
            parts.append(f"[{self.direction}]")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "script_id": self.script_id,
            "character_id": self.character_id,
            "dialogue": self.dialogue,
            "action": self.action,
            "direction": self.direction,
            "emotion": self.emotion,
            "pose": self.pose,
            "outfit": self.outfit,
            "source": self.source,
        }
