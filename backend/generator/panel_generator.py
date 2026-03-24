from __future__ import annotations

from backend.models.panel import Panel
from backend.models.scene import Scene
from backend.models.character import Character
from .comfyui_bridge import ComfyUIBridge


class PanelGenerator:
    """Generates panel images by composing scene and character prompts."""

    def __init__(self, comfyui_bridge: ComfyUIBridge):
        self.comfyui = comfyui_bridge

    async def generate_panel_image(self, panel: Panel, scene: Scene, characters: list[Character]) -> str:
        """
        Generate the image for a panel.
        Combines scene background prompt with character appearance prompts.
        Returns the file path of the generated image.
        """
        prompt_parts = [scene.to_image_prompt()]
        for character in characters:
            if character.appearance_prompt:
                prompt_parts.append(character.appearance_prompt)
        if panel.image_prompt:
            prompt_parts.append(panel.image_prompt)

        full_prompt = ", ".join(prompt_parts)
        image_path = await self.comfyui.generate_image(full_prompt)
        panel.image_path = image_path
        return image_path

    async def animate_panel(self, panel: Panel, motion_prompt: str = "") -> str:
        """Generate an animated version of a panel. Returns the video file path."""
        if not panel.image_path:
            raise ValueError(f"Panel {panel.panel_id} has no image to animate")
        video_path = await self.comfyui.generate_video(panel.image_path, motion_prompt)
        panel.video_path = video_path
        panel.is_animated = True
        return video_path
