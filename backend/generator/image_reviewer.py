"""Image reviewer — closes the adversarial loop.

Prompt → Image → Caption (reverse) → Compare → Gap → Training signal

Uses LLaVA (vision-language model) via ollama to caption generated images,
then compares the caption to the original prompt to measure how well
the image matches the intention.

The gap between prompt and caption IS the adversarial signal:
- Small gap = model understood the prompt correctly
- Large gap = model misinterpreted something
- The specific differences tell us WHAT was wrong
"""
from __future__ import annotations

import os
import base64
import logging
import httpx
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of reviewing a generated image against its prompt."""
    original_prompt: str
    reverse_caption: str
    match_score: float          # 0.0 = completely wrong, 1.0 = perfect match
    differences: list[str]      # specific mismatches found
    suggestion: str             # how to improve the prompt


class ImageReviewer:
    """Reviews generated images by reverse-captioning and comparing to prompt."""

    def __init__(
        self,
        ollama_host: str = None,
        vision_model: str = "llava:7b",
        text_model: str = "llama3:8b",
    ) -> None:
        if ollama_host is None:
            ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
        self.ollama_host = ollama_host
        self.vision_model = vision_model
        self.text_model = text_model

    async def review(
        self,
        image_bytes: bytes,
        original_prompt: str,
    ) -> ReviewResult:
        """Full adversarial review: caption the image, compare to prompt."""
        caption = await self.caption_image(image_bytes)
        if not caption:
            return ReviewResult(
                original_prompt=original_prompt,
                reverse_caption="(captioning failed)",
                match_score=0.5,
                differences=[],
                suggestion="",
            )

        comparison = await self.compare_prompts(original_prompt, caption)
        return comparison

    async def caption_image(self, image_bytes: bytes) -> str:
        """Reverse an image back to a text description using LLaVA."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            async with httpx.AsyncClient() as client:
                # Check if vision model is available
                tags = await client.get(
                    f"{self.ollama_host}/api/tags", timeout=5.0
                )
                if tags.status_code == 200:
                    models = [
                        m["name"] for m in tags.json().get("models", [])
                    ]
                    if self.vision_model not in models:
                        log.warning(
                            "Vision model '%s' not found. "
                            "Available: %s. "
                            "Pull with: ollama pull %s",
                            self.vision_model, models, self.vision_model,
                        )
                        return ""

                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": (
                            "Describe this image in detail for an AI "
                            "image generator. Include: characters "
                            "(species, clothing, pose, expression), "
                            "setting, lighting, mood, and any notable "
                            "visual details. Be specific and concise. "
                            "Output only the description."
                        ),
                        "images": [image_b64],
                        "stream": False,
                    },
                    timeout=60.0,
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    log.warning(
                        "Captioning returned %s: %s",
                        response.status_code, response.text[:200],
                    )
        except Exception as e:
            log.warning("Image captioning failed: %s", e)

        return ""

    async def compare_prompts(
        self,
        original_prompt: str,
        reverse_caption: str,
    ) -> ReviewResult:
        """Compare the original prompt to the reverse caption using LLM."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.text_model,
                        "prompt": (
                            f"Compare these two descriptions of the same image.\n\n"
                            f"INTENDED (what was asked for):\n{original_prompt}\n\n"
                            f"ACTUAL (what the image shows):\n{reverse_caption}\n\n"
                            f"Respond in this exact format:\n"
                            f"SCORE: (0.0 to 1.0, how well they match)\n"
                            f"DIFFERENCES: (comma-separated list of mismatches)\n"
                            f"SUGGESTION: (one sentence on how to improve the prompt)\n"
                        ),
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("response", "")
                    return self._parse_comparison(
                        original_prompt, reverse_caption, text
                    )
        except Exception as e:
            log.warning("Prompt comparison failed: %s", e)

        return ReviewResult(
            original_prompt=original_prompt,
            reverse_caption=reverse_caption,
            match_score=0.5,
            differences=[],
            suggestion="",
        )

    def _parse_comparison(
        self,
        original: str,
        caption: str,
        llm_response: str,
    ) -> ReviewResult:
        score = 0.5
        differences = []
        suggestion = ""

        for line in llm_response.split("\n"):
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    score = float(score_text.split()[0])
                    score = max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("DIFFERENCES:"):
                diff_text = line.split(":", 1)[1].strip()
                differences = [
                    d.strip() for d in diff_text.split(",") if d.strip()
                ]
            elif line.upper().startswith("SUGGESTION:"):
                suggestion = line.split(":", 1)[1].strip()

        return ReviewResult(
            original_prompt=original,
            reverse_caption=caption,
            match_score=score,
            differences=differences,
            suggestion=suggestion,
        )
