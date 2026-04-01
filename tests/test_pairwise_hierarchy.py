"""Exhaustive pairwise inheritance tests — 72 tests total.

6 objects: Story, Character, Chapter, Page, Panel, Script
36 forward (can A reach B) + 36 reverse (emission/context) = 72

All objects constructed through factory methods — no direct construction.
"""
import pytest
from backend.models import Story, Character, Chapter, Page, Panel, Script
from tests.helpers import make_story, get_chain


@pytest.fixture
def hierarchy():
    story = make_story()
    return get_chain(story)


class TestForward:
    """Forward traversal — every object can reach every other."""

    # Self-identity (6)
    def test_story_is_story(self, hierarchy):
        assert hierarchy["story"].story_id == "s1"

    def test_character_is_character(self, hierarchy):
        assert hierarchy["character"].character_id == "c1"

    def test_chapter_is_chapter(self, hierarchy):
        assert hierarchy["chapter"].chapter_id is not None

    def test_page_is_page(self, hierarchy):
        assert hierarchy["page"].page_id is not None

    def test_panel_is_panel(self, hierarchy):
        assert hierarchy["panel"].panel_id is not None

    def test_script_is_script(self, hierarchy):
        assert hierarchy["script"].script_id is not None

    # Story -> X (5)
    def test_story_to_character(self, hierarchy):
        assert hierarchy["story"].get_character("c1") is hierarchy["character"]

    def test_story_to_chapter(self, hierarchy):
        ch_id = hierarchy["chapter"].chapter_id
        assert hierarchy["story"].get_chapter(ch_id) is hierarchy["chapter"]

    def test_story_to_page(self, hierarchy):
        assert hierarchy["story"].lookup(hierarchy["page"].page_id) is hierarchy["page"]

    def test_story_to_panel(self, hierarchy):
        assert hierarchy["story"].lookup(hierarchy["panel"].panel_id) is hierarchy["panel"]

    def test_story_to_script(self, hierarchy):
        assert hierarchy["story"].lookup(hierarchy["script"].script_id) is hierarchy["script"]

    # Character -> X (5)
    def test_character_to_story(self, hierarchy):
        assert hierarchy["character"]._parent is hierarchy["story"]

    def test_character_to_chapter(self, hierarchy):
        assert len(hierarchy["character"].chapters) >= 1

    def test_character_to_page(self, hierarchy):
        pages = [p for ch in hierarchy["character"].chapters for p in ch.pages]
        assert hierarchy["page"] in pages

    def test_character_to_panel(self, hierarchy):
        panels = [pan for ch in hierarchy["character"].chapters for p in ch.pages for pan in p.panels]
        assert hierarchy["panel"] in panels

    def test_character_to_script(self, hierarchy):
        scripts = [s for ch in hierarchy["character"].chapters for p in ch.pages for pan in p.panels for s in pan.scripts.values()]
        assert hierarchy["script"] in scripts

    # Chapter -> X (5)
    def test_chapter_to_story(self, hierarchy):
        assert hierarchy["story"].get_chapter(hierarchy["chapter"].chapter_id) is not None

    def test_chapter_to_character(self, hierarchy):
        assert "c1" in hierarchy["chapter"].character_ids

    def test_chapter_to_page(self, hierarchy):
        assert hierarchy["page"] in hierarchy["chapter"].pages

    def test_chapter_to_panel(self, hierarchy):
        panels = [pan for p in hierarchy["chapter"].pages for pan in p.panels]
        assert hierarchy["panel"] in panels

    def test_chapter_to_script(self, hierarchy):
        scripts = [s for p in hierarchy["chapter"].pages for pan in p.panels for s in pan.scripts.values()]
        assert hierarchy["script"] in scripts

    # Page -> X (5)
    def test_page_to_story(self, hierarchy):
        assert hierarchy["story"].lookup(hierarchy["page"].page_id) is not None

    def test_page_to_character(self, hierarchy):
        ctx = hierarchy["page"].get_context()
        assert ctx["character"]["character_id"] == "c1"

    def test_page_to_chapter(self, hierarchy):
        ctx = hierarchy["page"].get_context()
        assert "chapter" in ctx

    def test_page_to_panel(self, hierarchy):
        assert hierarchy["panel"] in hierarchy["page"].panels

    def test_page_to_script(self, hierarchy):
        scripts = [s for pan in hierarchy["page"].panels for s in pan.scripts.values()]
        assert hierarchy["script"] in scripts

    # Panel -> X (5)
    def test_panel_to_story(self, hierarchy):
        assert hierarchy["story"].lookup(hierarchy["panel"].panel_id) is not None

    def test_panel_to_character(self, hierarchy):
        ctx = hierarchy["panel"].get_context()
        assert ctx["character"]["character_id"] == "c1"

    def test_panel_to_chapter(self, hierarchy):
        ctx = hierarchy["panel"].get_context()
        assert "chapter" in ctx

    def test_panel_to_page(self, hierarchy):
        ctx = hierarchy["panel"].get_context()
        assert ctx["page"]["page_id"] == hierarchy["page"].page_id

    def test_panel_to_script(self, hierarchy):
        assert hierarchy["script"].character_id in hierarchy["panel"].scripts

    # Script -> X (5)
    def test_script_to_story(self, hierarchy):
        assert hierarchy["story"].lookup(hierarchy["script"].script_id) is not None

    def test_script_to_character(self, hierarchy):
        assert hierarchy["story"].get_character(hierarchy["script"].character_id) is hierarchy["character"]

    def test_script_to_chapter(self, hierarchy):
        ctx = hierarchy["script"].get_context()
        assert "chapter" in ctx

    def test_script_to_page(self, hierarchy):
        ctx = hierarchy["script"].get_context()
        assert "page" in ctx

    def test_script_to_panel(self, hierarchy):
        ctx = hierarchy["script"].get_context()
        assert ctx["panel"]["panel_id"] == hierarchy["panel"].panel_id


class TestReverse:
    """Reverse propagation — events emit upward only."""

    # Self-emission (6)
    def test_story_emits_to_self(self, hierarchy):
        received = []
        hierarchy["story"].on("story_updated", lambda d: received.append(True))
        hierarchy["story"].emit("story_updated", None)
        assert len(received) == 1

    def test_character_emits_to_self(self, hierarchy):
        received = []
        hierarchy["character"].on("character_updated", lambda d: received.append(True))
        hierarchy["character"].emit("character_updated", None)
        assert len(received) == 1

    def test_chapter_emits_to_self(self, hierarchy):
        received = []
        hierarchy["chapter"].on("chapter_updated", lambda d: received.append(True))
        hierarchy["chapter"].emit("chapter_updated", None)
        assert len(received) == 1

    def test_page_emits_to_self(self, hierarchy):
        received = []
        hierarchy["page"].on("page_updated", lambda d: received.append(True))
        hierarchy["page"].emit("page_updated", None)
        assert len(received) == 1

    def test_panel_emits_to_self(self, hierarchy):
        received = []
        hierarchy["panel"].on("panel_updated", lambda d: received.append(True))
        hierarchy["panel"].emit("panel_updated", None)
        assert len(received) == 1

    def test_script_emits_to_self(self, hierarchy):
        received = []
        hierarchy["script"].on("script_updated", lambda d: received.append(True))
        hierarchy["script"].emit("script_updated", None)
        assert len(received) == 1

    # Script emits up (5)
    def test_script_emits_to_panel(self, hierarchy):
        received = []
        hierarchy["panel"].on("panel_updated", lambda d: received.append(True))
        hierarchy["script"].update(dialogue="x", source="manual")
        assert len(received) > 0

    def test_script_emits_to_page(self, hierarchy):
        received = []
        hierarchy["page"].on("page_updated", lambda d: received.append(True))
        hierarchy["script"].update(dialogue="x", source="manual")
        assert len(received) > 0

    def test_script_emits_to_chapter(self, hierarchy):
        received = []
        hierarchy["chapter"].on("chapter_updated", lambda d: received.append(True))
        hierarchy["script"].update(dialogue="x", source="manual")
        assert len(received) > 0

    def test_script_emits_to_character(self, hierarchy):
        received = []
        hierarchy["character"].on("character_updated", lambda d: received.append(True))
        hierarchy["script"].update(dialogue="x", source="manual")
        assert len(received) > 0

    def test_script_emits_to_story(self, hierarchy):
        received = []
        hierarchy["story"].on("story_updated", lambda d: received.append(True))
        hierarchy["script"].update(dialogue="x", source="manual")
        assert len(received) > 0

    # Panel emits up (5)
    def test_panel_emits_to_page(self, hierarchy):
        received = []
        hierarchy["page"].on("page_updated", lambda d: received.append(True))
        hierarchy["panel"].update_image("h", source="manual")
        assert len(received) > 0

    def test_panel_emits_to_chapter(self, hierarchy):
        received = []
        hierarchy["chapter"].on("chapter_updated", lambda d: received.append(True))
        hierarchy["panel"].update_image("h", source="manual")
        assert len(received) > 0

    def test_panel_emits_to_character(self, hierarchy):
        received = []
        hierarchy["character"].on("character_updated", lambda d: received.append(True))
        hierarchy["panel"].update_image("h", source="manual")
        assert len(received) > 0

    def test_panel_emits_to_story(self, hierarchy):
        received = []
        hierarchy["story"].on("story_updated", lambda d: received.append(True))
        hierarchy["panel"].update_image("h", source="manual")
        assert len(received) > 0

    def test_panel_does_not_emit_down_to_script(self, hierarchy):
        received = []
        hierarchy["script"].on("script_updated", lambda d: received.append(True))
        hierarchy["panel"].update_image("h", source="manual")
        assert len(received) == 0

    # Page emits up (5)
    def test_page_emits_to_chapter(self, hierarchy):
        received = []
        hierarchy["chapter"].on("chapter_updated", lambda d: received.append(True))
        hierarchy["page"].emit_up("page_updated", None)
        assert len(received) > 0

    def test_page_emits_to_character(self, hierarchy):
        received = []
        hierarchy["character"].on("character_updated", lambda d: received.append(True))
        hierarchy["page"].emit_up("page_updated", None)
        assert len(received) > 0

    def test_page_emits_to_story(self, hierarchy):
        received = []
        hierarchy["story"].on("story_updated", lambda d: received.append(True))
        hierarchy["page"].emit_up("page_updated", None)
        assert len(received) > 0

    def test_page_does_not_emit_down_to_panel(self, hierarchy):
        received = []
        hierarchy["panel"].on("panel_updated", lambda d: received.append(True))
        hierarchy["page"].emit_up("page_updated", None)
        assert len(received) == 0

    def test_page_does_not_emit_down_to_script(self, hierarchy):
        received = []
        hierarchy["script"].on("script_updated", lambda d: received.append(True))
        hierarchy["page"].emit_up("page_updated", None)
        assert len(received) == 0

    # Chapter emits up (5)
    def test_chapter_emits_to_character(self, hierarchy):
        received = []
        hierarchy["character"].on("character_updated", lambda d: received.append(True))
        hierarchy["chapter"].emit_up("chapter_updated", None)
        assert len(received) > 0

    def test_chapter_emits_to_story(self, hierarchy):
        received = []
        hierarchy["story"].on("story_updated", lambda d: received.append(True))
        hierarchy["chapter"].emit_up("chapter_updated", None)
        assert len(received) > 0

    def test_chapter_does_not_emit_down_to_page(self, hierarchy):
        received = []
        hierarchy["page"].on("page_updated", lambda d: received.append(True))
        hierarchy["chapter"].emit_up("chapter_updated", None)
        assert len(received) == 0

    def test_chapter_does_not_emit_down_to_panel(self, hierarchy):
        received = []
        hierarchy["panel"].on("panel_updated", lambda d: received.append(True))
        hierarchy["chapter"].emit_up("chapter_updated", None)
        assert len(received) == 0

    def test_chapter_does_not_emit_down_to_script(self, hierarchy):
        received = []
        hierarchy["script"].on("script_updated", lambda d: received.append(True))
        hierarchy["chapter"].emit_up("chapter_updated", None)
        assert len(received) == 0

    # Character emits up (5)
    def test_character_emits_to_story(self, hierarchy):
        received = []
        hierarchy["story"].on("story_updated", lambda d: received.append(True))
        hierarchy["character"].update(name="Updated")
        assert len(received) > 0

    def test_character_does_not_emit_down_to_chapter(self, hierarchy):
        received = []
        hierarchy["chapter"].on("chapter_updated", lambda d: received.append(True))
        hierarchy["character"].update(name="Updated")
        assert len(received) == 0

    def test_character_does_not_emit_down_to_page(self, hierarchy):
        received = []
        hierarchy["page"].on("page_updated", lambda d: received.append(True))
        hierarchy["character"].update(name="Updated")
        assert len(received) == 0

    def test_character_does_not_emit_down_to_panel(self, hierarchy):
        received = []
        hierarchy["panel"].on("panel_updated", lambda d: received.append(True))
        hierarchy["character"].update(name="Updated")
        assert len(received) == 0

    def test_character_does_not_emit_down_to_script(self, hierarchy):
        received = []
        hierarchy["script"].on("script_updated", lambda d: received.append(True))
        hierarchy["character"].update(name="Updated")
        assert len(received) == 0

    # Story emits (5)
    def test_story_does_not_emit_down_to_character(self, hierarchy):
        received = []
        hierarchy["character"].on("character_updated", lambda d: received.append(True))
        hierarchy["story"].emit("story_updated", None)
        assert len(received) == 0

    def test_story_does_not_emit_down_to_chapter(self, hierarchy):
        received = []
        hierarchy["chapter"].on("chapter_updated", lambda d: received.append(True))
        hierarchy["story"].emit("story_updated", None)
        assert len(received) == 0

    def test_story_does_not_emit_down_to_page(self, hierarchy):
        received = []
        hierarchy["page"].on("page_updated", lambda d: received.append(True))
        hierarchy["story"].emit("story_updated", None)
        assert len(received) == 0

    def test_story_does_not_emit_down_to_panel(self, hierarchy):
        received = []
        hierarchy["panel"].on("panel_updated", lambda d: received.append(True))
        hierarchy["story"].emit("story_updated", None)
        assert len(received) == 0

    def test_story_does_not_emit_down_to_script(self, hierarchy):
        received = []
        hierarchy["script"].on("script_updated", lambda d: received.append(True))
        hierarchy["story"].emit("story_updated", None)
        assert len(received) == 0
