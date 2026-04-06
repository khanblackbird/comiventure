"""Tag translator — uses Llama to convert natural language to Danbooru tags.

Users write naturally: "lying on the bed looking at her phone"
Models need tags: "lying, on_bed, holding_phone, looking_at_viewer"

This is what an LLM is for. Don't regex what you can reason about.

The translator:
1. Sends natural language to Llama with tag vocabulary examples
2. Gets back comma-separated Danbooru tags
3. Normalizes through tag_vocabulary for validation
4. Falls back to basic normalization if Llama is unavailable
"""
from __future__ import annotations

import logging
import os

import httpx

from backend.generator.tag_vocabulary import (
    normalize_tag, find_closest_tag,
    POSES, EXPRESSIONS, FRAMING, CLOTHING, ACCESSORIES, ACTIONS,
    ALL_TAGS,
)

log = logging.getLogger(__name__)

# Sample tags per field for the LLM prompt
FIELD_EXAMPLES = {
    "pose": "standing, sitting, kneeling, lying, running, crossed_arms, hand_on_hip",
    "action": "fighting, reading, holding_sword, waving, eating, sleeping, dancing",
    "emotion": "smile, angry, crying, surprised, serious, embarrassed, blush",
    "outfit": "school_uniform, dress, armor, hoodie, kimono, jacket, bikini, nude",
    "direction": "portrait, upper_body, cowboy_shot, full_body, close-up, from_above",
    "shot_type": "wide, medium, close-up, extreme_close-up, over-shoulder",
}


class TagTranslator:
    """Translates natural language field values to Danbooru tags via Llama."""

    def __init__(
        self,
        ollama_host: str = None,
        model: str = "llama3:8b",
    ) -> None:
        if ollama_host is None:
            ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
        self.ollama_host = ollama_host
        self.model = model
        self._available = None

    async def is_available(self) -> bool:
        """Check if Llama is reachable."""
        if self._available is not None:
            return self._available
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{self.ollama_host}/api/tags", timeout=2.0)
                self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def translate_field(self, field: str, value: str) -> list[str]:
        """Translate a single field value to Danbooru tags.

        "lying on the bed looking at her phone" → ["lying", "on_bed", "holding_phone"]
        "bored, content" → ["bored", "expressionless"]
        "overalls loosely fitted over a white shirt" → ["overalls", "shirt"]
        """
        if not value or normalize_tag(value) in ("", "none"):
            return []

        # Short values (1-2 words) — just normalize, no LLM needed
        if len(value.split()) <= 2:
            tag = normalize_tag(value)
            return [tag] if tag else []

        if not await self.is_available():
            return self._fallback(field, value)

        examples = FIELD_EXAMPLES.get(field, "")
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": (
                            f"Convert this to Danbooru image tags "
                            f"(comma-separated, underscore format).\n\n"
                            f"Field: {field}\n"
                            f"Input: {value}\n"
                            f"Example tags: {examples}\n\n"
                            f"Output ONLY the tags, nothing else."
                        ),
                        "stream": False,
                    },
                    timeout=10.0,
                )
                if r.status_code == 200:
                    text = r.json().get("response", "").strip()
                    tags = [normalize_tag(t) for t in text.split(",")]
                    return [t for t in tags if t]
        except Exception as e:
            log.warning("Tag translation failed: %s", e)

        return self._fallback(field, value)

    async def translate_script(self, script_dict: dict) -> dict:
        """Translate all script fields at once in a single LLM call.

        More efficient than per-field calls. Returns a dict with the
        same keys but Danbooru tag values.
        """
        fields_text = []
        for field in ("pose", "action", "emotion", "outfit", "direction"):
            value = script_dict.get(field, "")
            if value:
                fields_text.append(f"{field}: {value}")

        if not fields_text:
            return script_dict

        # Short/simple values — skip LLM
        all_short = all(len(v.split()) <= 2 for v in script_dict.values() if v)
        if all_short:
            return script_dict

        if not await self.is_available():
            return script_dict

        examples_block = "\n".join(
            f"  {f}: {FIELD_EXAMPLES[f]}" for f in FIELD_EXAMPLES
        )

        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": (
                            "Convert these scene descriptions to Danbooru "
                            "image tags (underscore format). "
                            "Output ONLY in the same field: value format.\n\n"
                            f"Input:\n" + "\n".join(fields_text) + "\n\n"
                            f"Example vocabulary:\n{examples_block}\n\n"
                            "Output:"
                        ),
                        "stream": False,
                    },
                    timeout=15.0,
                )
                if r.status_code == 200:
                    text = r.json().get("response", "").strip()
                    return self._parse_field_response(text, script_dict)
        except Exception as e:
            log.warning("Script translation failed: %s", e)

        return script_dict

    def _parse_field_response(self, text: str, original: dict) -> dict:
        """Parse LLM field:value response back into a dict."""
        result = dict(original)
        for line in text.split("\n"):
            line = line.strip()
            for field in ("pose", "action", "emotion", "outfit", "direction"):
                if line.lower().startswith(f"{field}:"):
                    value = line[len(field) + 1:].strip()
                    tags = [normalize_tag(t) for t in value.split(",")]
                    tags = [t for t in tags if t]
                    if tags:
                        result[field] = ", ".join(tags)
                    break
        return result

    def _fallback(self, field: str, value: str) -> list[str]:
        """Basic normalization when LLM is unavailable."""
        tag = normalize_tag(value)
        if not tag:
            return []
        tag_sets = {
            "pose": POSES, "action": ACTIONS,
            "emotion": EXPRESSIONS, "outfit": CLOTHING,
            "direction": FRAMING,
        }
        ts = tag_sets.get(field)
        if ts:
            return [find_closest_tag(tag, ts)]
        return [tag]
