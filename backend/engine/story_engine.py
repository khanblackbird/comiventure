from __future__ import annotations

from typing import Optional

from backend.models.story import Story
from backend.models.scene import Scene
from backend.models.panel import Panel
from backend.models.page import Page
from .character_ai import CharacterAI


class StoryEngine:
    """Orchestrates the story flow — dialogue, scene transitions, panel creation."""

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

        character.add_dialogue("user", message)

        # TODO: generate response via CharacterAI
        # TODO: determine if scene should change
        # TODO: generate panel data for the new story beat
        raise NotImplementedError

    async def transition_scene(self, scene_id: str) -> Scene:
        """Move the story to a new scene."""
        scene = self.story.scenes.get(scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")
        self.story.current_scene_id = scene_id
        return scene

    def create_panel(self, scene_id: str, image_prompt: str) -> Panel:
        """Create a new panel and add it to the current page."""
        panel = Panel(
            panel_id=f"panel-{len(self.story.pages)}-{self._current_page_panel_count()}",
            scene_id=scene_id,
            image_prompt=image_prompt,
        )
        self._ensure_current_page().add_panel(panel)
        return panel

    def _ensure_current_page(self) -> Page:
        """Get the current page or create a new one."""
        if not self.story.pages or self.story.pages[-1].panel_count() >= 6:
            page = Page(
                page_id=f"page-{len(self.story.pages)}",
                page_number=len(self.story.pages) + 1,
            )
            self.story.add_page(page)
        return self.story.pages[-1]

    def _current_page_panel_count(self) -> int:
        if not self.story.pages:
            return 0
        return self.story.pages[-1].panel_count()
