from __future__ import annotations

from backend.models.story import Story
from backend.models.chapter import Chapter
from backend.models.page import Page
from backend.models.panel import Panel
from backend.models.script import Script
from .character_ai import CharacterAI


class StoryEngine:
    """Orchestrates the story flow using the emission-based object hierarchy.

    Does not directly wire objects together — it creates them and lets
    the hierarchy handle context propagation and event emission.
    """

    def __init__(self, story: Story, character_ai: CharacterAI):
        self.story = story
        self.character_ai = character_ai

    async def handle_player_input(self, character_id: str, message: str) -> dict:
        """
        Process a player's message to a character.
        Returns a dict with the character's response and any new panel data.
        """
        character = self.story.get_character(character_id)
        if not character:
            return {"error": f"Character {character_id} not found"}

        # TODO: generate response via CharacterAI using character.get_context()
        # TODO: create script from response
        # TODO: propagation handles the rest
        raise NotImplementedError

    def create_chapter(self, chapter_id: str, title: str, character_ids: list[str]) -> Chapter:
        """Create a shared chapter between characters — forming their relationship."""
        chapter = Chapter(chapter_id=chapter_id, title=title)
        self.story.create_shared_chapter(chapter, character_ids)
        return chapter

    def create_page(self, chapter: Chapter, page_number: int) -> Page:
        """Add a new page to a chapter."""
        page = Page(
            page_id=f"page-{chapter.chapter_id}-{page_number}",
            page_number=page_number,
        )
        chapter.add_page(page)
        return page

    def create_panel(self, page: Page) -> Panel:
        """Create a new panel on a page."""
        panel = Panel(
            panel_id=f"panel-{page.page_id}-{page.panel_count()}",
        )
        page.add_panel(panel)
        return panel

    def create_script(self, panel: Panel, character_id: str) -> Script:
        """Create a script for a character in a panel."""
        script = Script(
            script_id=f"script-{panel.panel_id}-{character_id}",
            character_id=character_id,
        )
        panel.add_script(script)
        return script
