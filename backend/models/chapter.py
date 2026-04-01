from __future__ import annotations

from .emitter import Emitter
from .page import Page
from .ids import make_id


class Chapter(Emitter):
    """A chapter — a narrative thread shared between characters.

    The chapter IS the relationship between characters. The same
    chapter instance can be referenced by multiple characters,
    making the story relational.

    Emits 'chapter_updated' upward when pages/content change.
    Listens to 'page_updated' from child pages.
    """

    def __init__(
        self,
        chapter_id: str,
        title: str = "",
        synopsis: str = "",
        default_location: str = "",
        default_time_of_day: str = "",
        is_solo: bool = False,
    ) -> None:
        super().__init__()
        self.chapter_id = chapter_id
        self.title = title
        self.synopsis = synopsis
        self.default_location = default_location  # "enchanted forest", "space station"
        self.default_time_of_day = default_time_of_day  # "night", "dawn"
        self.negative_prompt = ""  # chapter-wide negative
        self.is_solo = is_solo  # True = character sheet chapter (1 character only)
        self.pages: list[Page] = []
        self.character_ids: list[str] = []

    def create_page(self, page_id: str | None = None, page_number: int = 0, layout_template: str = "auto") -> Page:
        """Create a page within this chapter.
        Inherits setting/mood/action from the previous page.
        Auto-creates a panel with scripts for each chapter character.
        """
        if page_id is None:
            page_id = make_id("page")
        if page_number == 0:
            page_number = len(self.pages) + 1

        # Inherit context from previous page, or chapter defaults
        setting = self.default_location
        mood = ""
        action_context = ""
        time_of_day = self.default_time_of_day
        weather = ""
        lighting = ""
        if self.pages:
            previous = self.pages[-1]
            setting = previous.setting or self.default_location
            mood = previous.mood
            action_context = previous.action_context
            time_of_day = previous.time_of_day or self.default_time_of_day
            weather = previous.weather
            lighting = previous.lighting

        page = Page(page_id, page_number, layout_template,
                    setting=setting, mood=mood, action_context=action_context,
                    time_of_day=time_of_day, weather=weather, lighting=lighting)
        self.add_page(page)
        page.ensure_panel(character_ids=self.character_ids)
        return page

    def ensure_page(self) -> Page:
        """Ensure this chapter has at least one page. Returns the first page."""
        if not self.pages:
            return self.create_page()
        return self.pages[0]

    def add_page(self, page: Page) -> None:
        """Add an existing page to this chapter (wires parent)."""
        page.set_parent(self)
        page.on("page_updated", self._on_page_updated)
        self.pages.append(page)
        self.emit_up("chapter_updated", self)

    def remove_page(self, page_id: str) -> None:
        """Remove a page by id. Refuses to remove the last page.
        Unregisters the page, its panels, and their scripts from the story registry.
        """
        if len(self.pages) <= 1:
            raise ValueError("Cannot remove the last page from a chapter — hierarchy requires at least one")

        page = self.get_page(page_id)
        if page:
            # Walk up to story for registry cleanup
            story = self._find_story()
            if story:
                for panel in page.panels:
                    for script in panel.scripts.values():
                        story.unregister(script.script_id)
                    story.unregister(panel.panel_id)
                story.unregister(page.page_id)

        self.pages = [p for p in self.pages if p.page_id != page_id]
        self.emit_up("chapter_updated", self)

    def _find_story(self):
        """Walk up the parent chain to find the Story root."""
        node = self
        while node._parent is not None:
            node = node._parent
        # Only return if it's actually a Story (has registry)
        return node if hasattr(node, '_registry') else None

    def get_page(self, page_id: str) -> Page | None:
        for page in self.pages:
            if page.page_id == page_id:
                return page
        return None

    def bind_character(self, character_id: str) -> None:
        """Bind a character to this chapter (creates the relationship).
        Cascades: adds a script for this character to every existing panel.
        """
        if character_id not in self.character_ids:
            self.character_ids.append(character_id)
            # Cascade into existing panels
            for page in self.pages:
                for panel in page.panels:
                    if character_id not in panel.scripts:
                        panel.create_script(character_id)
            self.emit_up("chapter_updated", self)

    def unbind_character(self, character_id: str) -> None:
        """Remove a character from this chapter."""
        if character_id in self.character_ids:
            self.character_ids.remove(character_id)
            self.emit_up("chapter_updated", self)

    def _on_page_updated(self, page: Page) -> None:
        """A child page changed — propagate upward."""
        self.emit_up("chapter_updated", self)

    def to_prompt(self) -> str:
        """Chapter-level prompt contribution: default location and time."""
        parts = []
        if self.default_location:
            parts.append(self.default_location)
        if self.default_time_of_day:
            parts.append(self.default_time_of_day)
        return ", ".join(parts)

    def _own_context(self) -> dict:
        return {
            "chapter": {
                "chapter_id": self.chapter_id,
                "title": self.title,
                "synopsis": self.synopsis,
                "character_ids": self.character_ids,
                "default_location": self.default_location,
                "default_time_of_day": self.default_time_of_day,
            }
        }

    def to_dict(self) -> dict:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "synopsis": self.synopsis,
            "character_ids": self.character_ids,
            "default_location": self.default_location,
            "default_time_of_day": self.default_time_of_day,
            "negative_prompt": self.negative_prompt,
            "is_solo": self.is_solo,
            "pages": [page.to_dict() for page in self.pages],
        }
