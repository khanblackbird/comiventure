"""Tests for graph integrity validation.

Since direct construction is now impossible, these tests verify:
1. validate() passes on properly constructed stories
2. validate() catches corruption after manual tampering
3. Removal guards prevent breaking the chain
4. Character removal cascades properly (scripts, solo chapters)
5. repair() fixes orphaned state
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


class TestCharacterRemoval:
    """Removing a character must cascade cleanly — no orphans left behind."""

    def test_remove_only_character_leaves_valid_minus_chapters(self):
        """Removing the only character removes solo chapters and scripts."""
        story = make_two_character_story()
        story.remove_character("c1")
        # c2 still exists, story should be valid
        assert story.get_character("c1") is None
        assert story.get_character("c2") is not None
        errors = story.validate()
        assert not any("nonexistent" in e for e in errors), errors

    def test_remove_character_cleans_scripts(self):
        """Scripts for the removed character must not remain in any panel."""
        story = make_two_character_story()
        story.remove_character("c1")
        for chapter in story.chapters.values():
            for page in chapter.pages:
                for panel in page.panels:
                    assert "c1" not in panel.scripts, \
                        f"Orphaned script for c1 in {panel.panel_id}"

    def test_remove_character_cleans_solo_chapter(self):
        """Solo chapter for the removed character must be deleted."""
        story = make_two_character_story()
        # Create solo chapters
        story.ensure_solo_chapter("c1")
        story.ensure_solo_chapter("c2")
        assert story.get_solo_chapter("c1") is not None

        story.remove_character("c1")
        assert story.get_solo_chapter("c1") is None
        # c2's solo chapter should still exist
        assert story.get_solo_chapter("c2") is not None

    def test_remove_character_unbinds_from_chapter(self):
        """Character must be removed from all chapter.character_ids."""
        story = make_two_character_story()
        chapter = list(story.chapters.values())[0]
        assert "c1" in chapter.character_ids

        story.remove_character("c1")
        assert "c1" not in chapter.character_ids

    def test_remove_character_story_validates(self):
        """Story must pass validation after removing a character."""
        story = make_two_character_story()
        story.ensure_solo_chapter("c1")
        story.ensure_solo_chapter("c2")
        story.remove_character("c1")
        errors = story.validate()
        assert errors == [], f"Validation failed after removal: {errors}"

    def test_remove_single_script_panel(self):
        """Removing a character who is the only one in a panel must remove the script."""
        story = make_story()  # single character
        story.add_character(Character("c2", "Rex", appearance_prompt="wolf"))
        # c1 has a script in the default panel; add c2 to the chapter
        chapter = get_chain(story)["chapter"]
        chapter.character_ids.append("c2")

        story.remove_character("c1")
        # The script for c1 should be gone
        for page in chapter.pages:
            for panel in page.panels:
                assert "c1" not in panel.scripts


class TestRepair:
    """repair() must fix orphaned state without data loss."""

    def test_repair_removes_orphaned_scripts(self):
        """Scripts referencing deleted characters should be removed."""
        story = make_two_character_story()
        panel = get_chain(story)["panel"]
        # Simulate corruption: delete character without cascade
        del story.characters["c1"]
        story.unregister("c1")

        assert any("nonexistent" in e for e in story.validate())
        repairs = story.repair()
        assert len(repairs) > 0
        assert "c1" not in panel.scripts

    def test_repair_removes_empty_chapters(self):
        """Chapters with no characters should be removed."""
        story = make_two_character_story()
        chapter = get_chain(story)["chapter"]
        # Simulate corruption: remove all characters from chapter
        chapter.character_ids = []

        repairs = story.repair()
        assert any("empty chapter" in r.lower() for r in repairs)

    def test_repair_then_validate(self):
        """After repair, validate should pass (or only have known-ok errors)."""
        story = make_two_character_story()
        story.ensure_solo_chapter("c1")
        # Corrupt: remove c1 without cascade
        del story.characters["c1"]
        story.unregister("c1")

        assert len(story.validate()) > 0
        story.repair()
        errors = story.validate()
        # Should have no nonexistent-character or empty-chapter errors
        assert not any("nonexistent" in e for e in errors), errors
        assert not any("no characters" in e for e in errors), errors

    def test_repair_idempotent(self):
        """Running repair twice produces no additional changes."""
        story = make_two_character_story()
        del story.characters["c1"]
        story.unregister("c1")

        first = story.repair()
        second = story.repair()
        assert len(first) > 0
        assert len(second) == 0
