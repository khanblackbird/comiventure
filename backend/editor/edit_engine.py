from __future__ import annotations

from backend.models.panel import Panel


class EditEngine:
    """Handles AI-driven edits to generated panel images and videos."""

    async def edit_panel_region(
        self, panel: Panel, mask_data: bytes, prompt: str,
    ) -> str:
        """Edit a region of a panel's image using inpainting.

        Returns the content hash of the edited image.
        """
        if not panel.image_hash:
            raise ValueError(f"Panel {panel.panel_id} has no image to edit")
        raise NotImplementedError

    async def edit_video_region(
        self, panel: Panel, mask_data: bytes, prompt: str,
    ) -> str:
        """Edit a region across all frames of an animated panel.

        Returns the content hash of the edited video.
        """
        if not panel.video_hash:
            raise ValueError(f"Panel {panel.panel_id} has no video to edit")
        raise NotImplementedError
