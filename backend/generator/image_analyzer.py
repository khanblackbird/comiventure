"""Image analyzer — extracts structured descriptions from uploaded images.

Uses LLaVA to see the image, then Llama to parse the caption into
structured fields (appearance, art style, etc).

Use cases:
- Upload character reference → auto-fill appearance properties
- Upload art samples → detect art style for the story
- Bootstrap a story from existing artwork
"""
from __future__ import annotations

import os
import base64
import logging
import httpx
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class CharacterAnalysis:
    """Structured character description extracted from an image."""
    species: str = ""
    body_type: str = ""
    height: str = ""
    skin_tone: str = ""
    hair_style: str = ""
    hair_colour: str = ""
    eye_colour: str = ""
    facial_features: str = ""
    outfit: str = ""
    accessories: str = ""
    pose: str = ""
    expression: str = ""
    caption: str = ""       # full freeform description


@dataclass
class ArtStyleAnalysis:
    """Art style description extracted from an image."""
    art_style: str = ""     # "manga", "watercolor", "cel-shaded"
    colour_palette: str = ""
    line_style: str = ""    # "thick outlines", "no outlines", "sketchy"
    rendering: str = ""     # "flat colour", "detailed shading", "painterly"
    genre_hints: str = ""   # "fantasy", "sci-fi", "slice of life"
    caption: str = ""


@dataclass
class ImageAnalysis:
    """Combined analysis of an uploaded image."""
    character: CharacterAnalysis = field(default_factory=CharacterAnalysis)
    art_style: ArtStyleAnalysis = field(default_factory=ArtStyleAnalysis)
    raw_caption: str = ""


class ImageAnalyzer:
    """Analyzes uploaded images to extract structured descriptions."""

    def __init__(
        self,
        ollama_host: str = None,
        vision_model: str = "llava:7b",
        text_model: str = "llama3:8b",
    ) -> None:
        if ollama_host is None:
            ollama_host = os.environ.get(
                "OLLAMA_HOST", "http://ollama:11434"
            )
        self.ollama_host = ollama_host
        self.vision_model = vision_model
        self.text_model = text_model

    async def analyze(self, image_bytes: bytes) -> ImageAnalysis:
        """Full analysis: caption with LLaVA, parse with Llama."""
        caption = await self._caption(image_bytes)
        if not caption:
            return ImageAnalysis(raw_caption="(analysis failed)")

        character = await self._extract_character(caption)
        art_style = await self._extract_art_style(caption)

        return ImageAnalysis(
            character=character,
            art_style=art_style,
            raw_caption=caption,
        )

    async def _caption(self, image_bytes: bytes) -> str:
        """Detailed caption via LLaVA."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": (
                            "Describe this image in thorough detail. "
                            "Include:\n"
                            "1. Art style (manga, watercolor, "
                            "realistic, etc), line work, colouring\n"
                            "2. Character details: species/type, "
                            "body type, hair colour and style, "
                            "eye colour, skin tone, facial features, "
                            "outfit, accessories\n"
                            "3. Pose and expression\n"
                            "4. Background and setting\n"
                            "Be specific. Output only the description."
                        ),
                        "images": [image_b64],
                        "stream": False,
                    },
                    timeout=180.0,
                )
                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                else:
                    log.warning("Image analysis caption HTTP %s: %s", response.status_code, response.text[:200])
        except Exception as e:
            log.warning("Image analysis caption failed: %s: %s", type(e).__name__, e)

        return ""

    async def _extract_character(self, caption: str) -> CharacterAnalysis:
        """Parse a caption into structured character fields via Llama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.text_model,
                        "prompt": (
                            "Extract character details from this "
                            "image description. Respond ONLY in this "
                            "exact format (leave blank if not visible):"
                            "\n\n"
                            f"Description: {caption}\n\n"
                            "species: (human, cat, wolf, fox, etc)\n"
                            "body_type: (slim, muscular, curvy, etc)\n"
                            "height: (short, average, tall)\n"
                            "skin_tone: (pale, tan, dark, fur colour)\n"
                            "hair_style: (long, short, ponytail, etc)\n"
                            "hair_colour: (colour)\n"
                            "eye_colour: (colour)\n"
                            "facial_features: (notable features)\n"
                            "outfit: (what they are wearing)\n"
                            "accessories: (jewellery, weapons, etc)\n"
                            "pose: (body position)\n"
                            "expression: (facial expression)"
                        ),
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    text = response.json().get("response", "")
                    return self._parse_character(text, caption)
        except Exception as e:
            log.warning("Character extraction failed: %s", e)

        return CharacterAnalysis(caption=caption)

    async def _extract_art_style(self, caption: str) -> ArtStyleAnalysis:
        """Parse a caption into structured art style fields via Llama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.text_model,
                        "prompt": (
                            "Extract the art style from this image "
                            "description. Respond ONLY in this exact "
                            "format:\n\n"
                            f"Description: {caption}\n\n"
                            "art_style: (e.g. manga, anime, "
                            "watercolor, western comic, pixel art, "
                            "realistic, cel-shaded)\n"
                            "colour_palette: (e.g. vibrant, pastel, "
                            "muted, monochrome, warm, cool)\n"
                            "line_style: (e.g. thick outlines, thin "
                            "lines, no outlines, sketchy)\n"
                            "rendering: (e.g. flat colour, cell shaded,"
                            " detailed shading, painterly)\n"
                            "genre_hints: (e.g. fantasy, sci-fi, "
                            "slice of life, horror)"
                        ),
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    text = response.json().get("response", "")
                    return self._parse_art_style(text, caption)
        except Exception as e:
            log.warning("Art style extraction failed: %s", e)

        return ArtStyleAnalysis(caption=caption)

    def _parse_character(
        self, text: str, caption: str
    ) -> CharacterAnalysis:
        """Parse LLM response into CharacterAnalysis."""
        fields = {
            "species": "", "body_type": "", "height": "",
            "skin_tone": "", "hair_style": "", "hair_colour": "",
            "eye_colour": "", "facial_features": "", "outfit": "",
            "accessories": "", "pose": "", "expression": "",
        }
        for line in text.split("\n"):
            line = line.strip()
            for field_name in fields:
                if line.lower().startswith(f"{field_name}:"):
                    value = line.split(":", 1)[1].strip()
                    # Strip quotes, parentheses
                    value = value.strip("()\"'")
                    if value.lower() not in (
                        "", "n/a", "none",
                        "not visible", "not applicable",
                    ):
                        fields[field_name] = value
                    break

        return CharacterAnalysis(caption=caption, **fields)

    def _parse_art_style(
        self, text: str, caption: str
    ) -> ArtStyleAnalysis:
        """Parse LLM response into ArtStyleAnalysis."""
        fields = {
            "art_style": "", "colour_palette": "",
            "line_style": "", "rendering": "",
            "genre_hints": "",
        }
        for line in text.split("\n"):
            line = line.strip()
            for field_name in fields:
                if line.lower().startswith(f"{field_name}:"):
                    value = line.split(":", 1)[1].strip()
                    value = value.strip("()\"'")
                    if value.lower() not in ("", "n/a", "none"):
                        fields[field_name] = value
                    break

        return ArtStyleAnalysis(caption=caption, **fields)
