"""IP-Adapter bridge — conditions SDXL generation using character reference
images for visual consistency across panels.

Each character has a reference bank (Appearance.accepted_references) with
curated images stored in the ContentStore. IP-Adapter uses these as image
embeddings to guide generation — same character, consistent look.

IP-Adapter adds cross-attention conditioning from a CLIP image encoder.
It does NOT modify model weights — compatible with CPU offload.

Flow:
  Character reference bank → accepted images → PIL conversion
  → IP-Adapter image embeddings → cross-attention conditioning
  → visually consistent generation
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

from PIL import Image

from backend.models.content_store import ContentStore
from backend.models.character import Character
from backend.models.panel import Panel


IP_ADAPTER_MODEL = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
DEFAULT_SCALE = 0.6


class IPAdapterBridge:
    """Collects character reference images and conditions the pipeline.

    Only includes references from characters with active (non-empty)
    scripts in the panel — empty cascade scripts are excluded.
    """

    def __init__(
        self,
        content_store: ContentStore,
        model_name: str = IP_ADAPTER_MODEL,
        scale: float = DEFAULT_SCALE,
    ) -> None:
        self.content_store = content_store
        self.model_name = model_name
        self.scale = scale
        self._loaded = False

    def ensure_loaded(self, pipeline) -> None:
        """Load IP-Adapter weights into the pipeline. Only once."""
        if self._loaded:
            return

        pipeline.load_ip_adapter(
            self.model_name,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            local_files_only=True,
        )
        pipeline.set_ip_adapter_scale(self.scale)
        self._loaded = True

    def collect_reference_images(
        self,
        characters: list[Character],
        panel: Panel,
    ) -> list[Image.Image]:
        """Collect accepted reference images for active characters.

        Only characters with non-empty scripts in the panel are included.
        Only accepted (not rejected or unrated) references are used.
        """
        images = []

        for character in characters:
            # Skip characters with empty scripts
            script = panel.get_script(character.character_id)
            if not script or not script.to_prompt():
                continue

            for reference in character.appearance.accepted_references():
                image_bytes = self.content_store.retrieve(
                    reference.content_hash
                )
                if image_bytes:
                    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    images.append(pil_image)

        return images

    def collect_style_references(
        self,
        style_hashes: list[str],
    ) -> list[Image.Image]:
        """Collect story-level style reference images.
        These condition every generation toward the story's visual style.
        """
        images = []
        for content_hash in style_hashes:
            image_bytes = self.content_store.retrieve(content_hash)
            if image_bytes:
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                images.append(pil_image)
        return images

    def prepare_generation_kwargs(
        self,
        characters: list[Character],
        panel: Panel,
        pipeline,
        style_references: list[str] = None,
    ) -> dict:
        """Prepare IP-Adapter kwargs for the pipeline call.

        Combines character reference images + story style references.
        Returns a dict with ip_adapter_image if references exist,
        or an empty dict if there are no references (don't pollute kwargs).
        """
        images = self.collect_reference_images(characters, panel)

        # Add story-level style references
        if style_references:
            images.extend(self.collect_style_references(style_references))

        if not images:
            return {}

        self.ensure_loaded(pipeline)

        return {"ip_adapter_image": images}
