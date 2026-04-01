from __future__ import annotations

from typing import Optional

from .emitter import Emitter
from .chapter import Chapter
from .appearance import Appearance
from .profile import Profile


class Character(Emitter):
    """A character — the top-level owner in the hierarchy.

    Characters own chapters. The same chapter can belong to multiple
    characters, which is what makes the story relational — chapters
    are the pairwise (or groupwise) connections between characters.

    Appearance is structured — explicit visual properties plus reference
    images that train the adapter layer. Not a freeform string.

    Emits 'character_updated' when properties change.
    Listens to 'chapter_updated' from child chapters.
    Back-propagation from scripts enriches the character over time.
    """

    def __init__(
        self,
        character_id: str,
        name: str,
        description: str = "",
        personality_prompt: str = "",
        appearance_prompt: str = "",
        portrait_path: Optional[str] = None,
        is_temporary: bool = False,
        page_scope: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.character_id = character_id
        self.name = name
        self.description = description
        self.personality_prompt = personality_prompt
        self.appearance = Appearance()
        self.profile = Profile()
        self.portrait_path = portrait_path
        self.is_temporary = is_temporary
        self.page_scope = page_scope
        self.negative_prompt = ""  # per-character negative: "animal ears, tail" for humans
        self.chapters: list[Chapter] = []
        self.conversations: list[dict] = []  # saved conversation bank

        # Backwards compat
        if appearance_prompt:
            self.appearance.properties.art_style_notes = appearance_prompt

    @property
    def appearance_prompt(self) -> str:
        """Composed appearance prompt from structured properties."""
        return self.appearance.to_prompt()

    def to_prompt(self) -> str:
        """Character-level prompt contribution: appearance + physical traits.
        This is the STANDARD anchor used in every panel this character appears in.
        """
        parts = []
        if self.appearance_prompt:
            parts.append(self.appearance_prompt)
        # Add physical traits from profile if they add visual detail
        if self.profile and self.profile.physical:
            if self.profile.physical.distinguishing_marks:
                parts.append(self.profile.physical.distinguishing_marks)
        return ", ".join(parts)

    def add_chapter(self, chapter: Chapter) -> None:
        """Add a chapter to this character's arc and wire up emission.

        Note: set_parent overwrites any previous parent. In shared chapters,
        the last character to call add_chapter becomes the parent for context
        propagation. This is intentional — the prompt composer gets characters
        explicitly, not through the parent chain. The parent is only used for
        emission propagation (dirty flags).
        """
        chapter.set_parent(self)
        chapter.bind_character(self.character_id)
        chapter.on("chapter_updated", self._on_chapter_updated)
        self.chapters.append(chapter)

    def remove_chapter(self, chapter_id: str) -> None:
        """Remove a chapter from this character's arc."""
        self.chapters = [
            chapter for chapter in self.chapters
            if chapter.chapter_id != chapter_id
        ]

    def get_chapter(self, chapter_id: str) -> Chapter | None:
        for chapter in self.chapters:
            if chapter.chapter_id == chapter_id:
                return chapter
        return None

    def update(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        personality_prompt: Optional[str] = None,
        appearance_prompt: Optional[str] = None,
    ) -> None:
        """Update character properties and emit."""
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if personality_prompt is not None:
            self.personality_prompt = personality_prompt
        if appearance_prompt is not None:
            self.appearance.properties.art_style_notes = appearance_prompt
        self.emit("character_updated", self)

    def update_appearance(
        self,
        properties: Optional[dict] = None,
    ) -> None:
        """Update structured appearance properties and emit."""
        if properties:
            from .appearance import AppearanceProperties
            for key, value in properties.items():
                if hasattr(self.appearance.properties, key):
                    setattr(self.appearance.properties, key, value)
        self.emit("character_updated", self)

    def add_reference_image(self, content_hash: str, source: str = "upload", tags: list[str] | None = None):
        """Add a visual reference image for this character."""
        reference = self.appearance.add_reference(content_hash, source, tags)
        self.emit("character_updated", self)
        return reference

    def accept_reference(self, content_hash: str) -> None:
        """Accept a reference — positive training signal for adapter."""
        self.appearance.accept_reference(content_hash)
        self.emit("character_updated", self)

    def reject_reference(self, content_hash: str) -> None:
        """Reject a reference — negative training signal for adapter."""
        self.appearance.reject_reference(content_hash)
        self.emit("character_updated", self)

    def _on_chapter_updated(self, chapter: Chapter) -> None:
        """A child chapter changed — back-propagation point."""
        self.emit("character_updated", self)

    def _own_context(self) -> dict:
        return {
            "character": {
                "character_id": self.character_id,
                "name": self.name,
                "description": self.description,
                "personality_prompt": self.personality_prompt,
                "appearance_prompt": self.appearance_prompt,
                "negative_prompt": self.negative_prompt,
                "appearance": self.appearance.to_dict(),
                "profile": self.profile.to_dict(),
            }
        }

    def to_system_prompt(self) -> str:
        """Build the LLM system prompt for this character — includes full profile."""
        parts = [f"You are {self.name}."]
        if self.description:
            parts.append(self.description)
        if self.personality_prompt:
            parts.append(f"Your personality: {self.personality_prompt}")
        profile_context = self.profile.to_llm_context()
        if profile_context:
            parts.append(profile_context)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "character_id": self.character_id,
            "name": self.name,
            "description": self.description,
            "personality_prompt": self.personality_prompt,
            "appearance_prompt": self.appearance_prompt,
            "negative_prompt": self.negative_prompt,
            "appearance": self.appearance.to_dict(),
            "profile": self.profile.to_dict(),
            "portrait_path": self.portrait_path,
            "is_temporary": self.is_temporary,
            "page_scope": self.page_scope,
            "chapters": [chapter.chapter_id for chapter in self.chapters],
            "conversations": self.conversations,
        }
