from __future__ import annotations

from typing import Optional

from .emitter import Emitter
from .script import Script
from .ids import make_id


class Panel(Emitter):
    """A single comic panel — a visual scene derived from its scripts.

    Holds content hashes, never pixel data. The actual images/video
    live in the ContentStore. What travels through emission is just
    a short hash string.

    Emits 'panel_updated' upward when content changes.
    Listens to 'script_updated' from child scripts.
    """

    def __init__(
        self,
        panel_id: str,
        image_hash: Optional[str] = None,
        video_hash: Optional[str] = None,
        is_animated: bool = False,
        narration: str = "",
        shot_type: str = "",
    ) -> None:
        super().__init__()
        self.panel_id = panel_id
        self.image_hash = image_hash
        self.video_hash = video_hash
        self.is_animated = is_animated
        self.narration = narration
        self.shot_type = shot_type  # "wide", "medium", "close-up", "extreme close-up", "over-shoulder", "bird's eye"
        self.negative_prompt = ""  # panel-specific negative
        self.scripts: dict[str, Script] = {}
        self.source: str = "empty"

    def create_script(
        self,
        character_id: str,
        script_id: str | None = None,
        dialogue: str = "",
        action: str = "",
        direction: str = "",
        emotion: str = "",
    ) -> Script:
        """Create a script within this panel. Auto-generates ID if not given."""
        if script_id is None:
            script_id = make_id("scr")
        script = Script(script_id, character_id, dialogue, action, direction, emotion)
        self.add_script(script)
        return script

    def ensure_scripts_for_characters(self, character_ids: list[str]) -> None:
        """Ensure this panel has a default script for each character.
        Called during cascade creation — every panel is born with scripts.
        """
        for character_id in character_ids:
            if character_id not in self.scripts:
                self.create_script(character_id)

    def inherit_from(self, other: Panel) -> None:
        """Copy defaults from another panel (e.g. the previous one).
        Panel inherits shot_type. Scripts inherit pose, outfit, direction
        (continuity fields). Only copies if the target field is empty.
        """
        # Panel-level inheritance
        if not self.shot_type and other.shot_type:
            self.shot_type = other.shot_type

        # Script-level inheritance
        for character_id, other_script in other.scripts.items():
            if character_id in self.scripts:
                my_script = self.scripts[character_id]
                # Only inherit if our script is empty
                if not my_script.to_prompt():
                    if other_script.emotion:
                        my_script.emotion = other_script.emotion
                    if other_script.pose:
                        my_script.pose = other_script.pose
                    if other_script.outfit:
                        my_script.outfit = other_script.outfit
                    if other_script.direction:
                        my_script.direction = other_script.direction

    def add_script(self, script: Script) -> None:
        """Add an existing script to this panel (wires parent)."""
        script.set_parent(self)
        script.on("script_updated", self._on_script_updated)
        self.scripts[script.character_id] = script

    def remove_script(self, character_id: str) -> None:
        """Remove a character's script from this panel.
        Refuses to remove the last script — would break the hierarchy.
        """
        if character_id not in self.scripts:
            return
        if len(self.scripts) <= 1:
            raise ValueError("Cannot remove the last script from a panel — hierarchy requires at least one")
        del self.scripts[character_id]
        self.emit_up("panel_updated", self)

    def get_script(self, character_id: str) -> Optional[Script]:
        return self.scripts.get(character_id)

    def update_image(self, image_hash: str, source: str = "ai") -> None:
        """Set the panel image hash and propagate upward.
        Only a hash travels through emission — not pixels.
        """
        self.image_hash = image_hash
        self.source = source
        self.emit_up("panel_updated", self)

    def update_video(self, video_hash: str, source: str = "ai") -> None:
        """Set the panel video hash and propagate upward."""
        self.video_hash = video_hash
        self.is_animated = True
        self.source = source
        self.emit_up("panel_updated", self)

    def to_prompt(self) -> str:
        """Panel-level prompt contribution: shot type, narration."""
        parts = []
        if self.shot_type:
            parts.append(f"{self.shot_type} shot")
        if self.narration:
            parts.append(self.narration)
        return ", ".join(parts)

    def update_narration(self, narration: str) -> None:
        self.narration = narration
        self.emit_up("panel_updated", self)

    def _on_script_updated(self, script: Script) -> None:
        """A child script changed — propagate upward."""
        self.emit_up("panel_updated", self)

    def collect_scripts_prompt(self) -> str:
        """Combine all scripts into a single prompt for image generation."""
        return "; ".join(
            script.to_prompt()
            for script in self.scripts.values()
            if script.to_prompt()
        )

    def _own_context(self) -> dict:
        return {
            "panel": {
                "panel_id": self.panel_id,
                "image_hash": self.image_hash,
                "narration": self.narration,
                "shot_type": self.shot_type,
                "negative_prompt": self.negative_prompt,
                "scripts": {
                    character_id: script.to_dict()
                    for character_id, script in self.scripts.items()
                },
            }
        }

    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id,
            "image_hash": self.image_hash,
            "video_hash": self.video_hash,
            "is_animated": self.is_animated,
            "narration": self.narration,
            "shot_type": self.shot_type,
            "negative_prompt": self.negative_prompt,
            "scripts": {
                character_id: script.to_dict()
                for character_id, script in self.scripts.items()
            },
            "source": self.source,
        }
