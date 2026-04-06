from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .emitter import Emitter
from .script import Script
from .ids import make_id


@dataclass
class TagObservation:
    """A tag observed in the gap between generation and review.

    Each observation maps a tag to the context that produced it.
    The collection of observations IS the explicit latent space —
    the learned mapping between intention and result.

    source_tags:  what we asked for (generation prompt tags)
    observed_tag: what emerged (found in review/edit, not in source)
    source:       how it was discovered (review, inpaint, analysis, manual)
    model_id:     which model produced this observation
    adapter_hash: adapter state when observed (or empty)
    context_hash: hash of the hierarchy context (chapter + page + panel settings)
    confidence:   how sure we are (review match_score, or 1.0 for edits)
    """
    observed_tag: str
    source_tags: list[str] = field(default_factory=list)
    source: str = ""
    model_id: str = ""
    adapter_hash: str = ""
    context_hash: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "observed_tag": self.observed_tag,
            "source_tags": self.source_tags,
            "source": self.source,
            "model_id": self.model_id,
            "adapter_hash": self.adapter_hash,
            "context_hash": self.context_hash,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TagObservation:
        return cls(
            observed_tag=data.get("observed_tag", ""),
            source_tags=data.get("source_tags", []),
            source=data.get("source", ""),
            model_id=data.get("model_id", ""),
            adapter_hash=data.get("adapter_hash", ""),
            context_hash=data.get("context_hash", ""),
            confidence=data.get("confidence", 1.0),
        )


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
        narration: str = "",
        shot_type: str = "",
    ) -> None:
        super().__init__()
        self.panel_id = panel_id
        self.image_hash = image_hash
        self.video_hash = video_hash
        self.narration = narration
        self.shot_type = shot_type  # "wide", "medium", "close-up", "extreme close-up", "over-shoulder", "bird's eye"
        self.negative_prompt = ""
        self.scripts: dict[str, Script] = {}
        self.tag_observations: list[TagObservation] = []
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

    @property
    def is_animated(self) -> bool:
        """Panel is animated if it has a video."""
        return self.video_hash is not None

    def update_video(self, video_hash: str, source: str = "ai") -> None:
        """Set the panel video hash and propagate upward."""
        self.video_hash = video_hash
        self.source = source
        self.emit_up("panel_updated", self)

    @property
    def discovered_tags(self) -> list[str]:
        """Flat list of observed tag strings — for prompt composition."""
        return [obs.observed_tag for obs in self.tag_observations]

    def observe_tags(
        self,
        tags: list[str],
        source: str,
        source_tags: list[str] = None,
        model_id: str = "",
        adapter_hash: str = "",
        confidence: float = 1.0,
    ) -> list[TagObservation]:
        """Record tags observed in the gap between generation and review.

        Each tag becomes a TagObservation — a hashmap connecting the
        observed tag to the context that produced it. Deduplicates
        against existing observations and script tags.

        Returns the new observations created.
        """
        existing = {obs.observed_tag for obs in self.tag_observations}
        script_tags = set()
        for script in self.scripts.values():
            for tag in script.to_prompt().split(", "):
                if tag:
                    script_tags.add(tag)

        # Build a context hash from the hierarchy
        context = self.get_context() if not self.is_orphan else {}
        import hashlib
        context_str = str(sorted(context.items())) if context else ""
        context_hash = hashlib.md5(context_str.encode()).hexdigest()[:12]

        new_observations = []
        for tag in tags:
            if tag and tag not in existing and tag not in script_tags:
                obs = TagObservation(
                    observed_tag=tag,
                    source_tags=source_tags or [],
                    source=source,
                    model_id=model_id,
                    adapter_hash=adapter_hash,
                    context_hash=context_hash,
                    confidence=confidence,
                )
                self.tag_observations.append(obs)
                existing.add(tag)
                new_observations.append(obs)

        if new_observations:
            self.emit_up("panel_updated", self)

        return new_observations

    def add_discovered_tags(self, tags: list[str]) -> None:
        """Convenience: observe tags without full context (backwards compat)."""
        self.observe_tags(tags, source="unknown")

    def get_tag_map(self) -> dict[str, TagObservation]:
        """Get observations as a hashmap: tag string → observation context."""
        return {obs.observed_tag: obs for obs in self.tag_observations}

    def get_observations_by_source(self, source: str) -> list[TagObservation]:
        """Filter observations by discovery source."""
        return [obs for obs in self.tag_observations if obs.source == source]

    def to_prompt(self) -> str:
        """Panel-level prompt tags: shot type + observed tags."""
        parts = []
        if self.shot_type:
            parts.append(self.shot_type)
        parts.extend(self.discovered_tags)
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
                "discovered_tags": self.discovered_tags,
                "tag_observations": [obs.to_dict() for obs in self.tag_observations],
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
            "discovered_tags": self.discovered_tags,
            "tag_observations": [obs.to_dict() for obs in self.tag_observations],
            "scripts": {
                character_id: script.to_dict()
                for character_id, script in self.scripts.items()
            },
            "source": self.source,
        }
