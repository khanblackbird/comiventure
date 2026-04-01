from __future__ import annotations

from .emitter import Emitter
from .panel import Panel
from .ids import make_id


class Page(Emitter):
    """A comic page — an ordered collection of panels in a layout.

    Emits 'page_updated' upward when panels change.
    Listens to 'panel_updated' from child panels.
    """

    def __init__(
        self,
        page_id: str,
        page_number: int = 0,
        layout_template: str = "auto",
        setting: str = "",
        mood: str = "",
        action_context: str = "",
        time_of_day: str = "",
        weather: str = "",
        lighting: str = "",
    ) -> None:
        super().__init__()
        self.page_id = page_id
        self.page_number = page_number
        self.panels: list[Panel] = []
        self.layout_template = layout_template
        self.setting = setting          # "dimly lit tavern, wooden tables, firelight"
        self.mood = mood                # "tense", "romantic", "comedic"
        self.action_context = action_context  # "combat", "argument", "chase"
        self.time_of_day = time_of_day  # "dawn", "noon", "dusk", "night"
        self.weather = weather          # "rain", "snow", "clear", "fog", "storm"
        self.lighting = lighting        # "firelight", "neon", "moonlight", "harsh sun"

    def create_panel(self, panel_id: str | None = None, character_ids: list[str] | None = None, **kwargs) -> Panel:
        """Create a panel within this page.
        Auto-creates default scripts for each character.
        Inherits defaults from the previous panel if one exists.
        """
        if panel_id is None:
            panel_id = make_id("pan")
        panel = Panel(panel_id, **kwargs)
        self.add_panel(panel)
        if character_ids:
            panel.ensure_scripts_for_characters(character_ids)
        # Inherit from previous panel
        if len(self.panels) > 1:
            previous = self.panels[-2]
            panel.inherit_from(previous)
        return panel

    def ensure_panel(self, character_ids: list[str] | None = None) -> Panel:
        """Ensure this page has at least one panel. Returns the first panel."""
        if not self.panels:
            return self.create_panel(character_ids=character_ids)
        return self.panels[0]

    def add_panel(self, panel: Panel) -> None:
        """Add an existing panel to this page (wires parent)."""
        panel.set_parent(self)
        panel.on("panel_updated", self._on_panel_updated)
        self.panels.append(panel)
        self.emit_up("page_updated", self)

    def remove_panel(self, panel_id: str) -> None:
        """Remove a panel by id. Refuses to remove the last panel.
        Unregisters the panel and its scripts from the story registry.
        """
        if len(self.panels) <= 1:
            raise ValueError("Cannot remove the last panel from a page — hierarchy requires at least one")

        panel = self.get_panel(panel_id)
        if panel:
            # Walk up to story for registry cleanup
            story = self._find_story()
            if story:
                for script in panel.scripts.values():
                    story.unregister(script.script_id)
                story.unregister(panel.panel_id)

        self.panels = [p for p in self.panels if p.panel_id != panel_id]
        self.emit_up("page_updated", self)

    def _find_story(self):
        """Walk up the parent chain to find the Story root."""
        node = self
        while node._parent is not None:
            node = node._parent
        return node if hasattr(node, '_registry') else None

    def get_panel(self, panel_id: str) -> Panel | None:
        for panel in self.panels:
            if panel.panel_id == panel_id:
                return panel
        return None

    def panel_count(self) -> int:
        return len(self.panels)

    def to_prompt(self) -> str:
        """Page-level prompt contribution: setting, time, weather, lighting, mood."""
        parts = []
        if self.setting:
            parts.append(self.setting)
        if self.time_of_day:
            parts.append(self.time_of_day)
        if self.weather:
            parts.append(self.weather)
        if self.lighting:
            parts.append(f"{self.lighting} lighting")
        if self.mood:
            parts.append(f"{self.mood} atmosphere")
        if self.action_context:
            parts.append(self.action_context)
        return ", ".join(parts)

    def _on_panel_updated(self, panel: Panel) -> None:
        """A child panel changed — propagate upward."""
        self.emit_up("page_updated", self)

    def _own_context(self) -> dict:
        return {
            "page": {
                "page_id": self.page_id,
                "page_number": self.page_number,
                "panel_count": self.panel_count(),
                "setting": self.setting,
                "mood": self.mood,
                "action_context": self.action_context,
                "time_of_day": self.time_of_day,
                "weather": self.weather,
                "lighting": self.lighting,
            }
        }

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "page_number": self.page_number,
            "panels": [panel.to_dict() for panel in self.panels],
            "layout_template": self.layout_template,
            "setting": self.setting,
            "mood": self.mood,
            "action_context": self.action_context,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "lighting": self.lighting,
        }
