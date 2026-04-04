"""Tests that every creation operation produces a complete chain.

The chain: Story - Character - Chapter - Page - Panel - Script

Every creation must result in validate() returning empty.
No operation should be able to produce a gap in the chain.

These tests call creation operations and then check validate().
If validate() returns errors, the coupling is broken.
"""
import pytest
from backend.models import Story, Character, Chapter, Page, Panel, Script
from tests.helpers import make_story, make_two_character_story, get_chain


class TestStoryCreation:
    """story.create_chapter() must produce a complete chain."""

    def test_create_chapter_validates_clean(self):
        story = Story("s1", "Test")
        story.add_character(Character("c1", "Luna"))
        story.create_chapter("Ch1", ["c1"])
        assert story.validate() == []

    def test_create_chapter_produces_page(self):
        story = make_story()
        chapter = list(story.chapters.values())[0]
        assert len(chapter.pages) >= 1

    def test_create_chapter_produces_panel(self):
        story = make_story()
        chapter = list(story.chapters.values())[0]
        assert len(chapter.pages[0].panels) >= 1

    def test_create_chapter_produces_script(self):
        story = make_story()
        chapter = list(story.chapters.values())[0]
        panel = chapter.pages[0].panels[0]
        assert len(panel.scripts) >= 1

    def test_create_chapter_script_references_character(self):
        story = make_story()
        chapter = list(story.chapters.values())[0]
        panel = chapter.pages[0].panels[0]
        assert "c1" in panel.scripts

    def test_multi_character_chapter_validates_clean(self):
        story = make_two_character_story()
        assert story.validate() == []

    def test_multi_character_scripts_for_all(self):
        story = make_two_character_story()
        panel = get_chain(story)["panel"]
        assert "c1" in panel.scripts
        assert "c2" in panel.scripts


class TestChapterCreation:
    """chapter.create_page() must produce page + panel + scripts."""

    def test_create_page_validates_clean(self):
        story = make_story()
        chapter = list(story.chapters.values())[0]
        chapter.create_page()
        story._register_cascade(chapter)
        assert story.validate() == []

    def test_create_page_has_panel(self):
        story = make_story()
        chapter = list(story.chapters.values())[0]
        page = chapter.create_page()
        assert len(page.panels) >= 1

    def test_create_page_has_scripts(self):
        story = make_story()
        chapter = list(story.chapters.values())[0]
        page = chapter.create_page()
        assert len(page.panels[0].scripts) >= 1


class TestPageCreation:
    """page.create_panel() must produce panel + scripts."""

    def test_create_panel_validates_clean(self):
        story = make_story()
        chain = get_chain(story)
        chain["page"].create_panel(character_ids=["c1"])
        # Re-register
        for panel in chain["page"].panels:
            story.register_panel(panel)
            for script in panel.scripts.values():
                story.register_script(script)
        assert story.validate() == []

    def test_create_panel_has_scripts(self):
        story = make_story()
        chain = get_chain(story)
        panel = chain["page"].create_panel(character_ids=["c1"])
        assert "c1" in panel.scripts


class TestPanelCreation:
    """panel.create_script() produces a script."""

    def test_create_script_validates_clean(self):
        story = make_story()
        chain = get_chain(story)
        # Panel already has a script from cascade, add another character
        story.add_character(Character("c2", "Rex"))
        chapter = chain["chapter"]
        chapter.bind_character("c2")
        chain["panel"].create_script("c2")
        story.register_script(chain["panel"].scripts["c2"])
        assert story.validate() == []


class TestCharacterWithoutChapter:
    """A character added to a story without a chapter should fail validation."""

    def test_character_without_chapter_fails_validate(self):
        story = Story("s1", "Test")
        story.add_character(Character("c1", "Luna"))
        errors = story.validate()
        assert len(errors) > 0
        assert any("no chapters" in e for e in errors)


class TestIncompleteChain:
    """Manually breaking the chain should fail validation."""

    def test_panel_without_scripts_fails(self):
        story = make_story()
        panel = get_chain(story)["panel"]
        panel.scripts = {}
        errors = story.validate()
        assert any("no scripts" in e for e in errors)

    def test_page_without_panels_fails(self):
        story = make_story()
        page = get_chain(story)["page"]
        page.panels = []
        errors = story.validate()
        assert any("no panels" in e for e in errors)

    def test_chapter_without_pages_fails(self):
        story = make_story()
        chapter = get_chain(story)["chapter"]
        chapter.pages = []
        errors = story.validate()
        assert any("no pages" in e for e in errors)


class TestRemovalGuards:
    """Cannot remove the last link in a chain."""

    def test_cannot_remove_last_script(self):
        story = make_story()
        panel = get_chain(story)["panel"]
        with pytest.raises(ValueError, match="last script"):
            panel.remove_script("c1")

    def test_cannot_remove_last_panel(self):
        story = make_story()
        page = get_chain(story)["page"]
        with pytest.raises(ValueError, match="last panel"):
            page.remove_panel(page.panels[0].panel_id)

    def test_cannot_remove_last_page(self):
        story = make_story()
        chapter = get_chain(story)["chapter"]
        with pytest.raises(ValueError, match="last page"):
            chapter.remove_page(chapter.pages[0].page_id)

    def test_can_remove_non_last_script(self):
        story = make_two_character_story()
        panel = get_chain(story)["panel"]
        assert len(panel.scripts) == 2
        panel.remove_script("c2")
        assert len(panel.scripts) == 1
        assert story.validate() == []


class TestCharacterRemoval:
    """Removing a character cascades — cleans up scripts from all panels."""

    def test_remove_character_cleans_scripts(self):
        story = make_two_character_story()
        panel = get_chain(story)["panel"]
        assert "c1" in panel.scripts
        assert "c2" in panel.scripts

        story.remove_character("c2")
        assert "c2" not in panel.scripts
        assert "c1" in panel.scripts
        assert story.validate() == []

    def test_remove_character_unbinds_from_chapters(self):
        story = make_two_character_story()
        chapter = list(story.chapters.values())[0]
        assert "c2" in chapter.character_ids

        story.remove_character("c2")
        assert "c2" not in chapter.character_ids

    def test_remove_last_character_cleans_script(self):
        story = make_story()
        panel = get_chain(story)["panel"]
        assert len(panel.scripts) == 1

        # Removing the only character — script must be removed too
        # (leaving orphaned scripts causes integrity violations on save)
        story.remove_character("c1")
        assert len(panel.scripts) == 0


class TestAllObjectsParented:
    """Every object in a valid story has a parent (except Story)."""

    def test_no_orphans(self):
        story = make_story()
        chain = get_chain(story)
        assert not chain["character"].is_orphan
        assert not chain["chapter"].is_orphan
        assert not chain["page"].is_orphan
        assert not chain["panel"].is_orphan
        assert not chain["script"].is_orphan

    def test_correct_parent_types(self):
        story = make_story()
        chain = get_chain(story)
        chain["character"].require_parent(Story)
        chain["chapter"].require_parent(Character)
        chain["page"].require_parent(Chapter)
        chain["panel"].require_parent(Page)
        chain["script"].require_parent(Panel)


class TestAllObjectsRegistered:
    """Every object in a valid story is in the registry."""

    def test_all_registered(self):
        story = make_story()
        chain = get_chain(story)
        assert story.lookup(chain["character"].character_id) is chain["character"]
        assert story.lookup(chain["chapter"].chapter_id) is chain["chapter"]
        assert story.lookup(chain["page"].page_id) is chain["page"]
        assert story.lookup(chain["panel"].panel_id) is chain["panel"]
        assert story.lookup(chain["script"].script_id) is chain["script"]
