"""LLM-based prompt composer — translates hierarchy context into
clean, unambiguous image generation prompts.

The hierarchy provides rich human-meaningful context:
- Character descriptions (personality, appearance, backstory)
- Page context (setting, mood, action type)
- Scripts (dialogue, action, emotion, direction)

The LLM understands all of this and composes a focused prompt
that the image model won't misinterpret.

"goofy expression" → "character with a wide silly grin"
Not → "the Disney character Goofy"
"""
from __future__ import annotations

from typing import Optional
import httpx

from backend.models.panel import Panel
from backend.models.character import Character


class PromptComposer:
    """Composes image generation prompts using an LLM.

    Falls back to direct composition if LLM is unavailable.
    """

    def __init__(
        self,
        ollama_host: str = None,
        model: str = "llama3:8b",
    ) -> None:
        import os
        if ollama_host is None:
            ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
        self.ollama_host = ollama_host
        self.model = model
        self._available = None

    async def compose(
        self,
        panel: Panel,
        characters: list[Character],
    ) -> str:
        """Compose a clean image generation prompt from hierarchy context.

        If LLM is available, uses it to translate context into a focused prompt.
        If not, falls back to direct composition.
        """
        # Always compute the direct version for logging
        direct = self._compose_direct(panel, characters)
        self.last_direct_prompt = direct

        if await self.is_available():
            llm_result = await self._compose_with_llm(panel, characters)
            self.last_llm_prompt = llm_result
            self.last_method = "llm"
            return llm_result

        self.last_llm_prompt = None
        self.last_method = "direct"
        return direct

    async def is_available(self) -> bool:
        """Check if the LLM is reachable."""
        if self._available is not None:
            return self._available
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_host}/api/tags", timeout=2.0)
                self._available = response.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def _active_characters(
        self, panel: Panel, characters: list[Character]
    ) -> list[Character]:
        """Filter to characters with non-empty scripts in this panel.

        The cascade creates blank scripts for every chapter character,
        but only characters with actual content should appear in prompts.
        """
        active = []
        for character in characters:
            script = panel.get_script(character.character_id)
            if script and script.to_prompt():
                active.append(character)
        return active

    async def _compose_with_llm(self, panel: Panel, characters: list[Character]) -> str:
        """Use the LLM to compose a clean prompt."""
        characters = self._active_characters(panel, characters)
        context = panel.get_context()
        page_context = context.get("page", {})

        # Build the LLM instruction — TAG FORMAT, not prose
        system = (
            "You are a tag-based prompt composer for Stable Diffusion XL. "
            "Convert the scene description into comma-separated tags. "
            "Rules:\n"
            "- Output ONLY comma-separated tags, NO sentences or prose\n"
            "- Short tags: 1-3 words each, separated by commas\n"
            "- ONLY include the characters listed — no others\n"
            "- Convert metaphors to visual tags "
            "(e.g. 'goofy expression' → 'silly grin', not 'Goofy')\n"
            "- Character count tag: '1girl', '2girls', '1boy 1girl', etc.\n"
            "- Include: species, hair, eyes, pose, action, emotion, "
            "outfit, setting, time, weather, lighting, mood\n"
            "- Danbooru/e621 tag style preferred\n"
            "- No articles (a, the), no verbs (is, are), no filler\n"
            "- Output ONLY the tags, nothing else"
        )

        # Build the context message using standard to_prompt() methods
        parts = []

        # Story — art style
        story_prompt = self._get_story_prompt(panel)
        if story_prompt:
            parts.append(f"Art style: {story_prompt}")

        # Page — setting, time, weather, lighting, mood
        page_prompt = self._get_page_prompt(panel)
        if page_prompt:
            parts.append(f"Scene: {page_prompt}")

        # Panel — shot type
        if panel.shot_type:
            parts.append(f"Shot type: {panel.shot_type}")

        # Characters — ALWAYS appearance_prompt as base, then script overrides
        parts.append(f"\nCharacters in this panel ({len(characters)}):")
        for character in characters:
            script = panel.get_script(character.character_id)
            char_desc = []
            char_desc.append(f"  Name: {character.name}")
            if character.description:
                char_desc.append(f"  Description: {character.description}")
            # Character.to_prompt() is the STANDARD base
            char_prompt = character.to_prompt()
            if char_prompt:
                char_desc.append(f"  Appearance: {char_prompt}")
            # Script.to_prompt() fields are per-panel overrides
            if script:
                if script.pose:
                    char_desc.append(f"  Pose: {script.pose}")
                if script.action:
                    char_desc.append(f"  Action: {script.action}")
                if script.emotion:
                    char_desc.append(f"  Emotion: {script.emotion}")
                if script.outfit:
                    char_desc.append(f"  Outfit: {script.outfit}")
                if script.direction:
                    char_desc.append(f"  Camera/Direction: {script.direction}")
            parts.append("\n".join(char_desc))

        # Panel narration
        if panel.narration:
            parts.append(f"\nScene narration: {panel.narration}")

        user_message = "\n".join(parts)
        self.last_llm_input = user_message

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": user_message,
                        "system": system,
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    result = response.json()
                    prompt = result.get("response", "").strip()
                    if prompt:
                        # Strip LLM preamble if present
                        for prefix in [
                            "Here is a concise image generation prompt:",
                            "Here is the image generation prompt:",
                            "Here's the prompt:",
                        ]:
                            if prompt.lower().startswith(prefix.lower()):
                                prompt = prompt[len(prefix):].strip()
                        # Strip surrounding quotes
                        if prompt.startswith('"') and prompt.endswith('"'):
                            prompt = prompt[1:-1].strip()
                        return prompt
        except Exception as e:
            print(f"LLM prompt composition failed: {e}")

        # Fall back to direct
        return self._compose_direct(panel, characters)

    def _compose_direct(self, panel: Panel, characters: list[Character]) -> str:
        """Direct composition — no LLM. Used as fallback.

        Each object type has a standard to_prompt() method.
        The composition order is always:
          1. Story (art style)
          2. Panel (shot type, narration)
          3. Characters (appearance + script per character)
          4. Page (setting, time, weather, lighting, mood, action)

        Character appearance is ALWAYS the anchor — it comes from
        the Character object, not from individual scripts or panels.
        Scripts add per-panel action/pose/outfit/emotion on top.
        """
        characters = self._active_characters(panel, characters)
        prompt_parts = []

        # 1. Story — art style (global, always first)
        story_prompt = self._get_story_prompt(panel)
        if story_prompt:
            prompt_parts.append(story_prompt)

        # 2. Panel — shot type, narration
        panel_prompt = panel.to_prompt()
        if panel_prompt:
            prompt_parts.append(panel_prompt)

        # 3. Characters — appearance tags + script tags
        #    Multi-character: use AND separator (SDXL regional prompting)
        character_descriptions = []
        for character in characters:
            parts = []
            char_prompt = character.to_prompt()
            if char_prompt:
                parts.append(char_prompt)

            script = panel.get_script(character.character_id)
            if script:
                script_prompt = script.to_prompt()
                if script_prompt:
                    parts.append(script_prompt)

            if parts:
                character_descriptions.append(", ".join(parts))

        if character_descriptions:
            if len(character_descriptions) == 1:
                prompt_parts.append(character_descriptions[0])
            else:
                prompt_parts.append(" AND ".join(character_descriptions))

        # 4. Page — setting, time, weather, lighting, mood, action
        page_prompt = self._get_page_prompt(panel)
        if page_prompt:
            prompt_parts.append(page_prompt)

        return ", ".join(part for part in prompt_parts if part)

    def _get_story_prompt(self, panel: Panel) -> str:
        """Walk up to the Story and call its to_prompt()."""
        node = panel
        while node._parent is not None:
            node = node._parent
        if hasattr(node, 'to_prompt'):
            return node.to_prompt()
        return ""

    def _get_page_prompt(self, panel: Panel) -> str:
        """Get the parent Page's to_prompt().
        Falls back to Chapter's to_prompt() for location/time defaults.
        """
        page_prompt = ""
        chapter_prompt = ""
        if panel._parent and hasattr(panel._parent, 'to_prompt'):
            page_prompt = panel._parent.to_prompt()
            # Chapter is page's parent
            page = panel._parent
            if page._parent and hasattr(page._parent, 'to_prompt'):
                chapter_prompt = page._parent.to_prompt()

        # Page prompt takes priority; chapter fills gaps
        if page_prompt:
            return page_prompt
        return chapter_prompt

    def compose_negative(
        self,
        panel: Panel,
        characters: list[Character],
    ) -> str:
        """Collect negative prompts from the entire hierarchy and combine them.

        Stacks: Story + Chapter + Page + Panel + per-character negatives.
        Per-character negatives from the Character object apply whenever
        that character appears. Script-level negatives are per-panel overrides.

        Order: global defaults, story, chapter, page, panel, character-level.
        """
        from backend.generator.panel_generator import DEFAULT_NEGATIVE

        parts = [DEFAULT_NEGATIVE]

        # Walk up from panel to collect hierarchy negatives
        node = panel
        hierarchy_negatives = []
        while node is not None:
            neg = getattr(node, 'negative_prompt', '')
            if neg:
                hierarchy_negatives.append(neg)
            node = node._parent if hasattr(node, '_parent') else None

        # Reverse so story is first, panel is last
        hierarchy_negatives.reverse()
        parts.extend(hierarchy_negatives)

        # Character-level negatives (from Character object — always applies)
        active = self._active_characters(panel, characters)
        for character in active:
            if character.negative_prompt:
                parts.append(character.negative_prompt)

            # Script-level negative (per-panel override)
            script = panel.get_script(character.character_id)
            if script and script.negative_prompt:
                parts.append(script.negative_prompt)

        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            for token in part.split(', '):
                token = token.strip()
                if token and token not in seen:
                    seen.add(token)
                    unique_parts.append(token)

        return ", ".join(unique_parts)
