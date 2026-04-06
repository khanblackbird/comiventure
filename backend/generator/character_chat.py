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
        """Suggest ALL panel fields as Danbooru tags.

        Returns script fields (pose, action, emotion, outfit, direction)
        AND panel fields (shot_type, narration). All visual fields use
        Danbooru tag format for direct use in image generation.
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
                prompt = prev_script.to_prompt()
                if prompt:
                    context_parts.append(f"Previous panel tags: {prompt}")
            if previous_panel.shot_type:
                context_parts.append(f"Previous shot: {previous_panel.shot_type}")

        if next_panel:
            next_script = next_panel.get_script(character.character_id)
            if next_script:
                prompt = next_script.to_prompt()
                if prompt:
                    context_parts.append(f"Next panel tags: {prompt}")

        # Show current state so LLM fills gaps
        current = panel.get_script(character.character_id)
        current_parts = []
        if current:
            for f in ("pose", "action", "emotion", "outfit"):
                v = getattr(current, f, "")
                current_parts.append(f"{f}: {v}" if v else f"{f}: (empty)")
        current_parts.append(
            f"shot_type: {panel.shot_type}" if panel.shot_type else "shot_type: (empty)"
        )
        current_parts.append(
            f"narration: {panel.narration}" if panel.narration else "narration: (empty)"
        )

        scene = "\n".join(context_parts) if context_parts else ""
        current_state = "\n".join(current_parts)

        message = (
            f"{scene}\n\n"
            f"Current panel state:\n{current_state}\n\n"
            "Fill in ALL empty fields. Use Danbooru tag format "
            "(underscore, e.g. crossed_arms, looking_at_viewer).\n"
            "Respond ONLY in this exact format:\n"
            "pose: (standing, sitting, kneeling, running, "
            "crossed_arms, hand_on_hip, dynamic_pose)\n"
            "action: (fighting, reading, holding_sword, waving, "
            "casting_spell, dancing)\n"
            "emotion: (smile, angry, crying, surprised, serious, "
            "embarrassed, looking_at_viewer)\n"
            "outfit: (school_uniform, armor, dress, hoodie, "
            "kimono — Danbooru tag)\n"
            "direction: (portrait, upper_body, cowboy_shot, full_body, "
            "close-up, from_above, from_below)\n"
            "shot_type: (wide, medium, close-up, extreme_close-up, "
            "over-shoulder, birds_eye)\n"
            "narration: (one short sentence describing the scene)"
        )

        response = await self.chat(character, message, panel, page)

        from backend.generator.tag_vocabulary import (
            find_closest_tag, POSES, ACTIONS, EXPRESSIONS,
            CLOTHING, FRAMING,
        )

        result = {
            "pose": "", "action": "", "emotion": "",
            "outfit": "", "direction": "",
            "shot_type": "", "narration": "",
        }
        for line in response.split("\n"):
            line = line.strip()
            for field in result:
                if line.lower().startswith(f"{field}:"):
                    value = line[len(field) + 1:].strip().strip("()\"'")
                    if not value or value.lower() in ("empty", "n/a", "none"):
                        break
                    # Normalize visual fields through tag vocabulary
                    tag_sets = {
                        "pose": POSES, "action": ACTIONS,
                        "emotion": EXPRESSIONS, "outfit": CLOTHING,
                        "direction": FRAMING,
                    }
                    if field in tag_sets:
                        value = find_closest_tag(value, tag_sets[field])
                    result[field] = value
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
