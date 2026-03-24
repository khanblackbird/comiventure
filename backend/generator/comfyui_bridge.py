from __future__ import annotations

from typing import Optional


class ComfyUIBridge:
    """Interface to ComfyUI for image and video generation."""

    def __init__(self, comfyui_host: str = "http://localhost:8188"):
        self.comfyui_host = comfyui_host

    async def generate_image(self, prompt: str, negative_prompt: str = "", width: int = 768, height: int = 512) -> str:
        """Generate an image from a text prompt. Returns the output file path."""
        # TODO: submit workflow to ComfyUI API, poll for result
        raise NotImplementedError

    async def generate_video(self, source_image_path: str, motion_prompt: str = "", frames: int = 16) -> str:
        """Generate a short video from a source image using AnimateDiff. Returns the output file path."""
        # TODO: submit AnimateDiff workflow to ComfyUI
        raise NotImplementedError

    async def inpaint_image(self, image_path: str, mask_path: str, prompt: str) -> str:
        """Inpaint a masked region of an image. Returns the output file path."""
        # TODO: submit inpainting workflow to ComfyUI
        raise NotImplementedError

    async def inpaint_video(self, video_path: str, mask_path: str, prompt: str) -> str:
        """Inpaint a masked region across video frames. Returns the output file path."""
        # TODO: submit video inpainting workflow
        raise NotImplementedError
