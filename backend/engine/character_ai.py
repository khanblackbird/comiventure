from __future__ import annotations

from backend.models.character import Character


class CharacterAI:
    """Handles LLM-driven character dialogue and responses."""

    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3:8b"):
        self.ollama_host = ollama_host
        self.model = model

    async def generate_response(self, character: Character, player_message: str) -> str:
        """Generate a character's dialogue response to the player's message."""
        # TODO: call ollama API with character.to_system_prompt() + dialogue history
        raise NotImplementedError

    async def generate_scene_description(self, character: Character, context: str) -> str:
        """Generate a scene/image description based on the current story context."""
        # TODO: call ollama to produce an image prompt from narrative context
        raise NotImplementedError
