"""Tests for the object hierarchy — context, emission, lookup, prompts, source tracking.

All objects constructed through factory methods only.
"""
import pytest
from backend.models import Story, Character, Chapter, Page, Panel, Script
from tests.helpers import make_story, make_two_character_story, get_chain


@pytest.fixture
def story():
    return make_story()

@pytest.fixture
def chain(story):
    return get_chain(story)

@pytest.fixture
def two_char_story():
    return make_two_character_story()


class TestTopDownContext:
    def test_script_inherits_full_context(self, chain):
        context = chain["script"].get_context()
        assert "character" in context
        assert "chapter" in context
        assert "page" in context
        assert "panel" in context
        assert "script" in context

    def test_script_sees_chapter_synopsis(self, chain):
        context = chain["script"].get_context()
        assert context["chapter"]["synopsis"] == "the beginning"

    def test_script_sees_page_number(self, chain):
        context = chain["script"].get_context()
        assert context["page"]["page_number"] == 1

    def test_panel_sees_chapter_but_not_script(self, chain):
        context = chain["panel"].get_context()
        assert "chapter" in context
        assert "page" in context

    def test_page_context_has_no_panel_data(self, chain):
        context = chain["page"].get_context()
        assert "chapter" in context
        assert "page" in context
        assert "panel" not in context


class TestBottomUpEmission:
    def test_script_update_reaches_story(self, chain):
        received = []
        chain["story"].on("story_updated", lambda d: received.append(True))
        chain["script"].update(dialogue="changed", source="manual")
        assert len(received) > 0

    def test_script_update_reaches_panel(self, chain):
        received = []
        chain["panel"].on("panel_updated", lambda d: received.append(True))
        chain["script"].update(dialogue="changed", source="manual")
        assert len(received) > 0

    def test_panel_image_update_reaches_story(self, chain):
        received = []
        chain["story"].on("story_updated", lambda d: received.append(True))
        chain["panel"].update_image("hash123", source="manual")
        assert len(received) > 0

    def test_narration_update_propagates(self, chain):
        received = []
        chain["story"].on("story_updated", lambda d: received.append(True))
        chain["panel"].update_narration("The wind howled.")
        assert len(received) > 0


class TestSharedChapters:
    def test_chapter_has_both_characters(self, two_char_story):
        chapter = list(two_char_story.chapters.values())[0]
        assert "c1" in chapter.character_ids
        assert "c2" in chapter.character_ids

    def test_characters_share_same_chapter_instance(self, two_char_story):
        luna = two_char_story.get_character("c1")
        rex = two_char_story.get_character("c2")
        luna_ch = luna.chapters[0]
        rex_ch = rex.chapters[0]
        assert luna_ch is rex_ch

    def test_get_characters_for_chapter(self, two_char_story):
        chapter = list(two_char_story.chapters.values())[0]
        characters = two_char_story.get_characters_for_chapter(chapter.chapter_id)
        names = {c.name for c in characters}
        assert names == {"Luna", "Rex"}

    def test_script_update_notifies_story(self, two_char_story):
        chain = get_chain(two_char_story)
        received = []
        two_char_story.on("story_updated", lambda d: received.append(True))
        chain["script"].update(dialogue="changed", source="manual")
        assert len(received) > 0


class TestRegistryLookup:
    def test_lookup_character(self, chain):
        assert chain["story"].lookup("c1") is chain["character"]

    def test_lookup_chapter(self, chain):
        assert chain["story"].lookup(chain["chapter"].chapter_id) is chain["chapter"]

    def test_lookup_page(self, chain):
        assert chain["story"].lookup(chain["page"].page_id) is chain["page"]

    def test_lookup_panel(self, chain):
        assert chain["story"].lookup(chain["panel"].panel_id) is chain["panel"]

    def test_lookup_script(self, chain):
        assert chain["story"].lookup(chain["script"].script_id) is chain["script"]

    def test_lookup_typed(self, chain):
        assert chain["story"].lookup_as("c1", Character) is not None
        assert chain["story"].lookup_as("c1", Panel) is None

    def test_lookup_nonexistent(self, chain):
        assert chain["story"].lookup("does-not-exist") is None


class TestPromptComposition:
    def test_panel_collects_all_scripts(self, two_char_story):
        chain = get_chain(two_char_story)
        panel = chain["panel"]
        # Panel should have scripts for both characters
        assert "c1" in panel.scripts
        assert "c2" in panel.scripts

    def test_script_to_prompt_format(self, chain):
        chain["script"].update(action="rushes forward", emotion="panicked", source="manual")
        prompt = chain["script"].to_prompt()
        assert "rushes forward" in prompt
        assert "(panicked)" in prompt

    def test_script_prompt_excludes_dialogue(self, chain):
        chain["script"].update(dialogue="Hello!", action="waves", source="manual")
        prompt = chain["script"].to_prompt()
        assert "Hello!" not in prompt
        assert "waves" in prompt

    def test_empty_script_has_empty_prompt(self, chain):
        # Default cascade-created script has empty fields
        assert chain["script"].to_prompt() == ""


class TestSourceTracking:
    def test_script_default_source(self, chain):
        assert chain["script"].source == "empty"

    def test_script_manual_source(self, chain):
        chain["script"].update(dialogue="hello", source="manual")
        assert chain["script"].source == "manual"

    def test_script_ai_source(self, chain):
        chain["script"].update(dialogue="generated", source="ai")
        assert chain["script"].source == "ai"

    def test_panel_default_source(self, chain):
        assert chain["panel"].source == "empty"

    def test_panel_ai_image(self, chain):
        chain["panel"].update_image("hash123", source="ai")
        assert chain["panel"].source == "ai"

    def test_panel_manual_image(self, chain):
        chain["panel"].update_image("hash123", source="manual")
        assert chain["panel"].source == "manual"


class TestCachePerformance:
    def test_repeated_get_context_is_cached(self, chain):
        ctx1 = chain["script"].get_context()
        ctx2 = chain["script"].get_context()
        assert ctx1 is ctx2

    def test_cache_invalidated_after_update(self, chain):
        ctx_before = chain["script"].get_context()
        chain["script"].update(dialogue="new", source="manual")
        ctx_after = chain["script"].get_context()
        assert ctx_before is not ctx_after
