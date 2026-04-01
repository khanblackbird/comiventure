from __future__ import annotations

from backend.models.page import Page
from backend.models.panel import Panel


class ComicComposer:
    """Arranges panels into comic page layouts with speech bubbles and narration."""

    LAYOUT_TEMPLATES = {
        "auto": None,
        "single": [[1.0]],
        "two_equal": [[0.5, 0.5]],
        "three_row": [[0.33, 0.33, 0.33]],
        "hero_top": [[1.0], [0.5, 0.5]],
        "two_over_one": [[0.5, 0.5], [1.0]],
        "grid_2x2": [[0.5, 0.5], [0.5, 0.5]],
        "asymmetric": [[0.6, 0.4], [0.4, 0.6]],
    }

    def compute_layout(self, page: Page) -> dict:
        """
        Compute the CSS grid layout for a page's panels.
        Returns a dict describing the grid structure for the frontend.
        """
        panel_count = page.panel_count()
        template = page.layout_template

        if template == "auto":
            template = self._auto_layout(panel_count)

        grid = self.LAYOUT_TEMPLATES.get(template)
        if not grid:
            grid = self._auto_layout_grid(panel_count)

        return {
            "page_id": page.page_id,
            "template": template,
            "grid": grid,
            "panels": [self._panel_layout(panel) for panel in page.panels],
        }

    def _panel_layout(self, panel: Panel) -> dict:
        """Build layout data for a single panel."""
        # Collect dialogue from scripts for speech bubbles
        dialogues = []
        for character_id, script in panel.scripts.items():
            if script.dialogue:
                dialogues.append({
                    "character_id": character_id,
                    "text": script.dialogue,
                })

        return {
            "panel_id": panel.panel_id,
            "image_hash": panel.image_hash,
            "video_hash": panel.video_hash,
            "is_animated": panel.is_animated,
            "dialogues": dialogues,
            "narration": panel.narration,
            "shot_type": panel.shot_type,
        }

    def _auto_layout(self, panel_count: int) -> str:
        """Pick a layout template based on panel count."""
        if panel_count <= 1:
            return "single"
        elif panel_count == 2:
            return "two_equal"
        elif panel_count == 3:
            return "hero_top"
        elif panel_count == 4:
            return "grid_2x2"
        else:
            return "asymmetric"

    def _auto_layout_grid(self, panel_count: int) -> list[list[float]]:
        """Generate a generic grid for any panel count."""
        columns = min(panel_count, 3)
        rows = (panel_count + columns - 1) // columns
        width = round(1.0 / columns, 2)
        return [[width] * min(columns, panel_count - row_index * columns) for row_index in range(rows)]
