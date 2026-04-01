"""Tests for graph integrity validation.

Since direct construction is now impossible, these tests verify:
1. validate() passes on properly constructed stories
2. validate() catches corruption after manual tampering
3. Removal guards prevent breaking the chain
"""
import pytest
from backend.models import Story, Character, Script
from tests.helpers import make_story, make_two_character_story, get_chain


class TestValidStory:
    def test_minimal_story_valid(self):
        story = make_story()
        assert story.validate() == []

    def test_validate_or_raise_passes(self):
        story = make_story()
        story.validate_or_raise()

    def test_multi_character_valid(self):
        story = make_two_character_story()
        assert story.validate() == []


class TestCorruptionDetection:
    """Simulate corruption by tampering with internals after construction."""

    def test_detects_removed_scripts(self):
        story = make_story()
        panel = get_chain(story)["panel"]
        panel.scripts = {}  # corrupt: remove all scripts
        errors = story.validate()
        assert any("no scripts" in e for e in errors)

    def test_detects_removed_panels(self):
        story = make_story()
        page = get_chain(story)["page"]
        page.panels = []  # corrupt: remove all panels
        errors = story.validate()
        assert any("no panels" in e for e in errors)

    def test_detects_removed_pages(self):
        story = make_story()
        chapter = get_chain(story)["chapter"]
        chapter.pages = []  # corrupt: remove all pages
        errors = story.validate()
        assert any("no pages" in e for e in errors)

    def test_detects_dangling_character_reference(self):
        story = make_story()
        chapter = get_chain(story)["chapter"]
        chapter.character_ids.append("ghost")
        errors = story.validate()
        assert any("nonexistent character" in e for e in errors)

    def test_detects_script_wrong_character(self):
        story = Story("s1", "Test")
        story.add_character(Character("c1", "Luna"))
        story.create_chapter("Ch1", ["c1"])
        # Manually inject a script for a nonexistent character
        panel = get_chain(story)["panel"]
        fake = Script("fake", "ghost")
        fake.set_parent(panel)
        panel.scripts["ghost"] = fake
        errors = story.validate()
        assert any("nonexistent" in e.lower() or "not in chapter" in e for e in errors)

    def test_detects_unregistered_page(self):
        story = make_story()
        page = get_chain(story)["page"]
        story.unregister(page.page_id)
        errors = story.validate()
        assert any("not in registry" in e for e in errors)

    def test_detects_unregistered_panel(self):
        story = make_story()
        panel = get_chain(story)["panel"]
        story.unregister(panel.panel_id)
        errors = story.validate()
        assert any("not in registry" in e for e in errors)

    def test_detects_unregistered_script(self):
        story = make_story()
        script = get_chain(story)["script"]
        story.unregister(script.script_id)
        errors = story.validate()
        assert any("not in registry" in e for e in errors)
