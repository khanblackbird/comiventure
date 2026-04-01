from __future__ import annotations

from typing import Optional

from .emitter import Emitter
from .character import Character
from .chapter import Chapter
from .page import Page
from .panel import Panel
from .script import Script
from .ids import make_id


class Story(Emitter):
    """The top-level story container.

    Holds the character graph. Characters own chapters, chapters
    are shared between characters. The story itself is the collection
    of all characters and the relational web of chapters between them.

    Also maintains a flat registry of every object by ID for O(1)
    lookups — no traversal needed to find any node in the graph.
    """

    def __init__(
        self,
        story_id: str,
        title: str = "Untitled",
        synopsis: str = "",
        art_style: str = "",
        genre: str = "",
    ) -> None:
        super().__init__()
        self.story_id = story_id
        self.title = title
        self.synopsis = synopsis
        self.art_style = art_style      # "manga", "western comic", "watercolor"
        self.genre = genre              # "fantasy", "sci-fi", "slice of life"
        self.characters: dict[str, Character] = {}
        self.chapters: dict[str, Chapter] = {}

        # Flat registry — O(1) lookup for any object by ID
        self._registry: dict[str, Emitter] = {}

    def register(self, object_id: str, obj: Emitter) -> None:
        """Add any object to the flat registry."""
        self._registry[object_id] = obj

    def unregister(self, object_id: str) -> None:
        """Remove an object from the registry."""
        self._registry.pop(object_id, None)

    def lookup(self, object_id: str) -> Optional[Emitter]:
        """O(1) lookup for any object in the story by ID."""
        return self._registry.get(object_id)

    def lookup_as(self, object_id: str, expected_type: type) -> Optional[Emitter]:
        """O(1) typed lookup — returns None if not found or wrong type."""
        obj = self._registry.get(object_id)
        return obj if isinstance(obj, expected_type) else None

    def add_character(self, character: Character) -> None:
        """Add a character to the story.
        If chapters already exist, binds the character to the first non-solo chapter
        to maintain a complete chain. The bind cascades scripts into panels.
        """
        character.set_parent(self)
        character.on("character_updated", self._on_character_updated)
        self.characters[character.character_id] = character
        self.register(character.character_id, character)

        # Maintain chain: if non-solo chapters exist, bind to first one
        non_solo = [ch for ch in self.chapters.values() if not ch.is_solo]
        if non_solo:
            first_chapter = non_solo[0]
            if character.character_id not in first_chapter.character_ids:
                character.add_chapter(first_chapter)
                self._register_cascade(first_chapter)

    def _create_solo_chapter(self, character: Character) -> Chapter:
        """Create a solo 'Character Sheet' chapter for a character."""
        solo = Chapter(
            chapter_id=make_id("ch"),
            title=f"{character.name} — Character Sheet",
            is_solo=True,
        )
        return self.create_shared_chapter(solo, [character.character_id])

    def get_solo_chapter(self, character_id: str) -> Optional[Chapter]:
        """Find the solo chapter for a character, if it exists."""
        for chapter in self.chapters.values():
            if (chapter.is_solo
                    and len(chapter.character_ids) == 1
                    and chapter.character_ids[0] == character_id):
                return chapter
        return None

    def ensure_solo_chapter(self, character_id: str) -> Chapter:
        """Get or create the solo chapter for a character."""
        solo = self.get_solo_chapter(character_id)
        if solo:
            return solo
        character = self.get_character(character_id)
        if not character:
            raise ValueError(f"Character {character_id} not found")
        return self._create_solo_chapter(character)

    def remove_character(self, character_id: str) -> None:
        """Remove a character from the story.
        Cascades: removes their scripts from all panels, unbinds from all chapters.
        """
        if character_id not in self.characters:
            return

        # Remove scripts from all panels
        for chapter in self.chapters.values():
            if character_id in chapter.character_ids:
                chapter.unbind_character(character_id)
            for page in chapter.pages:
                for panel in page.panels:
                    if character_id in panel.scripts:
                        # Only remove if panel has other scripts (don't break chain)
                        if len(panel.scripts) > 1:
                            script = panel.scripts[character_id]
                            self.unregister(script.script_id)
                            del panel.scripts[character_id]

        # Remove character's chapters reference
        character = self.characters[character_id]
        character.chapters = []

        del self.characters[character_id]
        self.unregister(character_id)

    def get_character(self, character_id: str) -> Optional[Character]:
        return self.characters.get(character_id)

    def add_chapter(self, chapter: Chapter) -> None:
        """Register a chapter in the story's chapter index."""
        self.chapters[chapter.chapter_id] = chapter
        self.register(chapter.chapter_id, chapter)

    def get_chapter(self, chapter_id: str) -> Optional[Chapter]:
        return self.chapters.get(chapter_id)

    def create_shared_chapter(
        self,
        chapter: Chapter,
        character_ids: list[str],
    ) -> Chapter:
        """Create a chapter shared between multiple characters.
        This is how relationships are formed — the chapter is the
        pairwise connection.
        Auto-creates a page -> panel -> scripts cascade.
        """
        self.add_chapter(chapter)
        for character_id in character_ids:
            character = self.get_character(character_id)
            if character:
                character.add_chapter(chapter)

        # Cascade: chapter must have a page
        page = chapter.ensure_page()
        self._register_cascade(chapter)
        return chapter

    def create_chapter(
        self,
        title: str,
        character_ids: list[str],
        synopsis: str = "",
        chapter_id: str | None = None,
        default_location: str = "",
        default_time_of_day: str = "",
    ) -> Chapter:
        """Convenience: create and wire a chapter in one call."""
        if chapter_id is None:
            chapter_id = make_id("ch")
        chapter = Chapter(
            chapter_id, title, synopsis,
            default_location=default_location,
            default_time_of_day=default_time_of_day,
        )
        return self.create_shared_chapter(chapter, character_ids)

    def _register_cascade(self, chapter: Chapter) -> None:
        """Register all objects in a chapter's tree in the flat registry."""
        for page in chapter.pages:
            self.register_page(page)
            for panel in page.panels:
                self.register_panel(panel)
                for script in panel.scripts.values():
                    self.register_script(script)

    def get_characters_for_chapter(self, chapter_id: str) -> list[Character]:
        """Get all characters bound to a chapter."""
        chapter = self.get_chapter(chapter_id)
        if not chapter:
            return []
        return [
            self.characters[character_id]
            for character_id in chapter.character_ids
            if character_id in self.characters
        ]

    def register_page(self, page: Page) -> None:
        """Register a page in the flat registry."""
        self.register(page.page_id, page)

    def register_panel(self, panel: Panel) -> None:
        """Register a panel in the flat registry."""
        self.register(panel.panel_id, panel)

    def register_script(self, script: Script) -> None:
        """Register a script in the flat registry."""
        self.register(script.script_id, script)

    def validate(self) -> list[str]:
        """Walk the entire graph and return a list of integrity violations.
        Empty list means the graph is valid.

        Called:
        - After deserialization (load_story)
        - Before save (save_story)
        - On API startup
        - Periodically if needed
        """
        errors = []

        # Story must have at least one character
        if len(self.characters) == 0:
            errors.append("Story has no characters")

        # Story must have at least one chapter
        if len(self.chapters) == 0:
            errors.append("Story has no chapters")

        for character_id, character in self.characters.items():
            # Character must be parented to story
            if character.is_orphan:
                errors.append(f"Character '{character_id}' is orphaned")

            # Character must have at least one chapter
            if len(character.chapters) == 0:
                errors.append(f"Character '{character_id}' has no chapters")

        for chapter_id, chapter in self.chapters.items():
            # Chapter must be parented
            if chapter.is_orphan:
                errors.append(f"Chapter '{chapter_id}' is orphaned")

            # Chapter must have at least one character
            if len(chapter.character_ids) == 0:
                errors.append(f"Chapter '{chapter_id}' has no characters")

            # All chapter character_ids must reference existing characters
            for cid in chapter.character_ids:
                if cid not in self.characters:
                    errors.append(f"Chapter '{chapter_id}' references nonexistent character '{cid}'")

            # Chapter must have at least one page
            if len(chapter.pages) == 0:
                errors.append(f"Chapter '{chapter_id}' has no pages")

            for page in chapter.pages:
                # Page must be parented to this chapter
                if page.is_orphan:
                    errors.append(f"Page '{page.page_id}' is orphaned")

                # Page must be in registry
                if self.lookup(page.page_id) is not page:
                    errors.append(f"Page '{page.page_id}' not in registry")

                # Page must have at least one panel
                if len(page.panels) == 0:
                    errors.append(f"Page '{page.page_id}' has no panels")

                for panel in page.panels:
                    # Panel must be parented to this page
                    if panel.is_orphan:
                        errors.append(f"Panel '{panel.panel_id}' is orphaned")

                    # Panel must be in registry
                    if self.lookup(panel.panel_id) is not panel:
                        errors.append(f"Panel '{panel.panel_id}' not in registry")

                    # Panel must have at least one script
                    if len(panel.scripts) == 0:
                        errors.append(f"Panel '{panel.panel_id}' has no scripts")

                    for script_character_id, script in panel.scripts.items():
                        # Script must be parented to this panel
                        if script.is_orphan:
                            errors.append(f"Script '{script.script_id}' is orphaned")

                        # Script must be in registry
                        if self.lookup(script.script_id) is not script:
                            errors.append(f"Script '{script.script_id}' not in registry")

                        # Script's character_id must match its key
                        if script.character_id != script_character_id:
                            errors.append(
                                f"Script '{script.script_id}' character_id mismatch: "
                                f"key='{script_character_id}' vs field='{script.character_id}'"
                            )

                        # Script's character must exist in story
                        if script.character_id not in self.characters:
                            errors.append(
                                f"Script '{script.script_id}' references nonexistent "
                                f"character '{script.character_id}'"
                            )

                        # Script's character must be in the chapter
                        if script.character_id not in chapter.character_ids:
                            errors.append(
                                f"Script '{script.script_id}' character '{script.character_id}' "
                                f"not in chapter '{chapter_id}'"
                            )

        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise if any integrity violations found."""
        errors = self.validate()
        if errors:
            raise RuntimeError(
                f"Story graph integrity violation ({len(errors)} errors):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def _on_character_updated(self, character: Character) -> None:
        """A character changed — the story knows."""
        self.emit("story_updated", self)

    DEFAULT_ART_STYLE = "cinematic lighting, hyper-detailed textures"

    def to_prompt(self) -> str:
        """Story-level prompt contribution: art style.
        Falls back to DEFAULT_ART_STYLE if none set.
        """
        style = self.art_style or self.DEFAULT_ART_STYLE
        return style

    def _own_context(self) -> dict:
        return {
            "story": {
                "art_style": self.art_style,
                "genre": self.genre,
            }
        }

    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "synopsis": self.synopsis,
            "art_style": self.art_style,
            "genre": self.genre,
            "characters": {
                character_id: character.to_dict()
                for character_id, character in self.characters.items()
            },
            "chapters": {
                chapter_id: chapter.to_dict()
                for chapter_id, chapter in self.chapters.items()
            },
        }
