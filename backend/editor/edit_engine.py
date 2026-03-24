from __future__ import annotations

from backend.models.panel import Panel
from backend.generator.comfyui_bridge import ComfyUIBridge


class EditEngine:
    """Handles AI-driven edits to generated panel images and videos."""

    def __init__(self, comfyui_bridge: ComfyUIBridge):
        self.comfyui = comfyui_bridge

    async def edit_panel_region(self, panel: Panel, mask_data: bytes, prompt: str) -> str:
        """
        Edit a region of a panel's image using inpainting.
        mask_data is the PNG mask from the frontend canvas.
        prompt describes what to generate in the masked region.
        Returns the path to the edited image.
        """
        if not panel.image_path:
            raise ValueError(f"Panel {panel.panel_id} has no image to edit")

        # TODO: save mask_data to temp file, pass to comfyui inpainting
        raise NotImplementedError

    async def edit_video_region(self, panel: Panel, mask_data: bytes, prompt: str) -> str:
        """
        Edit a region across all frames of an animated panel.
        Returns the path to the edited video.
        """
        if not panel.video_path:
            raise ValueError(f"Panel {panel.panel_id} has no video to edit")

        # TODO: save mask, pass to video inpainting pipeline
        raise NotImplementedError
