"""In-character chat — talk to characters about scenes using LLM.

Characters respond based on their full profile, the current scene context,
and surrounding panels. Used for:
- Generating dialogue for scripts
- Discussing the scene with the character
- Getting character reactions to panels
- Filling in missing script details
"""
from __future__ import annotations

import logging
import os
import httpx

log = logging.getLogger(__name__)

from backend.models.character import Character
from backend.models.panel import Panel
from backend.models.page import Page


class CharacterChat:

    def __init__(
        self,
        ollama_host: str = None,
        model: str = "llama3:8b",
    ) -> None:
        if ollama_host is None:
            ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
        self.ollama_host = ollama_host
        self.model = model

    async def chat(
        self,
        character: Character,
        message: str,
        panel: Panel = None,
        page: Page = None,
        history: list[dict] = None,
    ) -> str:
        """Send a message to a character and get their in-character response.

        The character knows:
        - Their full profile (biography, physical traits, personality, expressions)
        - The current scene context (page setting/mood, panel scripts)
        - Surrounding panel context
        - Conversation history
        """
        system_prompt = self._build_system_prompt(character, panel, page)

        messages = []
        for msg in (history or []):
            messages.append(msg)
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "system", "content": system_prompt}] + messages,
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "").strip()
        except Exception as e:
            log.warning("Character chat failed: %s", e)

        return f"*{character.name} doesn't respond*"

    async def react_to_panel(
        self,
        character: Character,
        panel: Panel,
        page: Page = None,
    ) -> str:
        """Get a character's reaction to the current panel — for dialogue generation."""
        context_parts = []
        if page:
            if page.setting:
                context_parts.append(f"Setting: {page.setting}")
            if page.mood:
                context_parts.append(f"The mood is {page.mood}")

        # What other characters are doing in this panel
        for char_id, script in panel.scripts.items():
            if char_id != character.character_id:
                parts = []
                if script.action:
                    parts.append(script.action)
                if script.emotion:
                    parts.append(f"feeling {script.emotion}")
                if parts:
                    context_parts.append(f"Another character is {', '.join(parts)}")

        scene_desc = ". ".join(context_parts) if context_parts else "a scene"
        message = (
            f"React to this scene in character. {scene_desc}. "
            "What do you do and say? Keep it brief — "
            "one or two lines of dialogue and a short action."
        )

        return await self.chat(character, message, panel, page)

    async def suggest_scripts(
        self,
        character: Character,
        panel: Panel,
        page: Page = None,
        previous_panel: Panel = None,
        next_panel: Panel = None,
    ) -> dict:
        """Suggest script fields (dialogue, action, emotion, direction) for a panel.

        Uses surrounding panels for continuity.
        """
        context_parts = []
        if page:
            if page.setting:
                context_parts.append(f"Setting: {page.setting}")
            if page.mood:
                context_parts.append(f"Mood: {page.mood}")
            if page.action_context:
                context_parts.append(f"Action: {page.action_context}")

        if previous_panel:
            prev_script = previous_panel.get_script(character.character_id)
            if prev_script:
                parts = []
                if prev_script.action:
                    parts.append(f"action: {prev_script.action}")
                if prev_script.dialogue:
                    parts.append(f"said: \"{prev_script.dialogue}\"")
                if parts:
                    context_parts.append(f"In the previous panel, you were {', '.join(parts)}")

        if next_panel:
            next_script = next_panel.get_script(character.character_id)
            if next_script:
                parts = []
                if next_script.action:
                    parts.append(f"action: {next_script.action}")
                if next_script.dialogue:
                    parts.append(f"saying: \"{next_script.dialogue}\"")
                if parts:
                    context_parts.append(f"In the next panel, you will be {', '.join(parts)}")

        scene = "\n".join(context_parts) if context_parts else ""

        message = (
            f"{scene}\n\n"
            "Suggest visual details for this panel. "
            "Do NOT suggest dialogue — only visual/physical details. "
            "Respond ONLY in this exact format:\n"
            "action: (what you physically do)\n"
            "emotion: (one word emotion)\n"
            "pose: (body position — standing, sitting, crouching, etc.)\n"
            "outfit: (what you are wearing, if different from default)\n"
            "direction: (camera/framing suggestion)"
        )

        response = await self.chat(character, message, panel, page)

        # Parse the response
        result = {
            "action": "", "emotion": "",
            "pose": "", "outfit": "", "direction": "",
        }
        for line in response.split("\n"):
            line = line.strip()
            for field in result:
                if line.lower().startswith(f"{field}:"):
                    result[field] = line[len(field) + 1:].strip()
                    break

        return result

    def _build_system_prompt(
        self,
        character: Character,
        panel: Panel = None,
        page: Page = None,
    ) -> str:
        parts = [character.to_system_prompt()]

        parts.append(
            "\nYou are roleplaying as this character. Stay in character. "
            "Respond with dialogue and actions. Use *asterisks* for actions."
        )

        if page:
            scene_parts = []
            if page.setting:
                scene_parts.append(f"Setting: {page.setting}")
            if page.mood:
                scene_parts.append(f"Mood: {page.mood}")
            if page.action_context:
                scene_parts.append(f"Current action: {page.action_context}")
            if scene_parts:
                parts.append(f"\nCurrent scene:\n" + "\n".join(scene_parts))

        if panel and panel.narration:
            parts.append(f"\nNarration: {panel.narration}")

        return "\n".join(parts)
