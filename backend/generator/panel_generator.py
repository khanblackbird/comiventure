"""Panel image generation — uses PromptComposer to build clean prompts
from hierarchy context, then delegates to ImageGenerator.

Optionally uses IP-Adapter for character reference conditioning.
"""
from __future__ import annotations

from typing import Optional

from backend.models.panel import Panel
from backend.models.character import Character
from .image_generator import ImageGenerator
from .prompt_composer import PromptComposer
from .ip_adapter_bridge import IPAdapterBridge


DEFAULT_NEGATIVE = (
    "lowres, (worst_quality, bad_quality:1.2), bad_anatomy, sketch, "
    "jpeg_artefacts, signature, watermark, old, oldest, censored, "
    "bar_censor, simple_background"
)


class PanelGenerator:
    """Generates panel images from hierarchy context.

    Uses PromptComposer (LLM if available, direct fallback) to translate
    the rich hierarchy context into a clean image generation prompt.

    If an IP-Adapter bridge is provided, character reference images
    condition the generation for visual consistency.
    """

    def __init__(
        self,
        image_generator: ImageGenerator,
        prompt_composer: PromptComposer = None,
        ip_adapter_bridge: IPAdapterBridge = None,
    ):
        self.image_generator = image_generator
        self.prompt_composer = prompt_composer or PromptComposer()
        self.ip_adapter_bridge = ip_adapter_bridge

    async def generate_panel_image(
        self,
        panel: Panel,
        characters: list[Character],
        seed: Optional[int] = None,
    ) -> str:
        prompt = await self.compose_prompt(panel, characters)
        negative_prompt = self.compose_negative_prompt()

        # Collect IP-Adapter kwargs if bridge is available
        generate_kwargs = {}
        if self.ip_adapter_bridge and self.image_generator.pipeline:
            ip_kwargs = self.ip_adapter_bridge.prepare_generation_kwargs(
                characters, panel, self.image_generator.pipeline,
            )
            generate_kwargs.update(ip_kwargs)

        content_hash = await self.image_generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            **generate_kwargs,
        )

        panel.update_image(content_hash, source="ai")
        return content_hash

    async def compose_prompt(self, panel: Panel, characters: list[Character]) -> str:
        """Compose prompt via LLM or direct fallback."""
        return await self.prompt_composer.compose(panel, characters)

    def compose_prompt_direct(self, panel: Panel, characters: list[Character]) -> str:
        """Direct composition — for tests and when LLM is unavailable."""
        return self.prompt_composer._compose_direct(panel, characters)

    def compose_negative_prompt(self) -> str:
        """Default negative prompt for all generation."""
        return DEFAULT_NEGATIVE
