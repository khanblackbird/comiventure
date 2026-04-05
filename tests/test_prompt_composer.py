"""Tests for the prompt composition pipeline.

Tests the direct prompt composition path (no LLM needed).
Verifies that each model's to_prompt() produces the expected output,
and that PromptComposer._compose_direct() assembles them correctly.
"""
import pytest

from backend.models import Story, Character
from backend.models.story import Story as StoryClass
from backend.generator.prompt_composer import PromptComposer
from backend.generator.panel_generator import DEFAULT_NEGATIVE
from tests.helpers import make_story, make_two_character_story, get_chain


# ---------------------------------------------------------------------------
# Story.to_prompt()
# ---------------------------------------------------------------------------

class TestStoryToPrompt:
    def test_returns_art_style_when_set(self):
        story = Story("s1", "Test", art_style="watercolor")
        assert story.to_prompt() == "watercolor"

    def test_returns_default_when_art_style_empty(self):
        story = Story("s1", "Test", art_style="")
        assert story.to_prompt() == Story.DEFAULT_ART_STYLE

    def test_default_art_style_matches_constant(self):
        assert Story.DEFAULT_ART_STYLE == "cinematic lighting, hyper-detailed textures"

    def test_default_art_style_is_set(self):
        assert len(Story.DEFAULT_ART_STYLE) > 0


# ---------------------------------------------------------------------------
# Page.to_prompt()
# ---------------------------------------------------------------------------

class TestPageToPrompt:
    def test_returns_all_fields(self):
        story = make_story()
        chain = get_chain(story)
        page = chain["page"]
        page.setting = "dark alley"
        page.time_of_day = "night"
        page.weather = "rain"
        page.lighting = "neon"
        page.mood = "tense"
        page.action_context = "chase scene"

        prompt = page.to_prompt()
        assert "dark alley" in prompt
        assert "night" in prompt
        assert "rain" in prompt
        assert "neon" in prompt
        assert "tense" in prompt
        assert "chase scene" in prompt

    def test_empty_page_returns_empty_string(self):
        story = make_story()
        chain = get_chain(story)
        page = chain["page"]
        page.setting = ""
        page.time_of_day = ""
        page.weather = ""
        page.lighting = ""
        page.mood = ""
        page.action_context = ""
        assert page.to_prompt() == ""

    def test_partial_fields(self):
        story = make_story()
        chain = get_chain(story)
        page = chain["page"]
        page.setting = "forest clearing"
        page.time_of_day = ""
        page.weather = ""
        page.lighting = "moonlight"
        page.mood = ""
        page.action_context = ""

        prompt = page.to_prompt()
        assert "forest clearing" in prompt
        assert "moonlight" in prompt
        # Should not contain empty separators
        assert ",," not in prompt


# ---------------------------------------------------------------------------
# Panel.to_prompt()
# ---------------------------------------------------------------------------

class TestPanelToPrompt:
    def test_returns_shot_type_without_narration(self):
        """Narration is text overlay, not image tags — excluded from to_prompt."""
        story = make_story()
        chain = get_chain(story)
        panel = chain["panel"]
        panel.shot_type = "close-up"
        panel.narration = "the hero arrives"

        prompt = panel.to_prompt()
        assert "close-up" in prompt
        assert "the hero arrives" not in prompt

    def test_empty_panel_returns_empty_string(self):
        story = make_story()
        chain = get_chain(story)
        panel = chain["panel"]
        panel.shot_type = ""
        panel.narration = ""
        assert panel.to_prompt() == ""

    def test_shot_type_only(self):
        story = make_story()
        chain = get_chain(story)
        panel = chain["panel"]
        panel.shot_type = "wide"
        panel.narration = ""
        assert panel.to_prompt() == "wide"


# ---------------------------------------------------------------------------
# Character.to_prompt()
# ---------------------------------------------------------------------------

class TestCharacterToPrompt:
    def test_returns_appearance_prompt(self):
        story = make_story(appearance_prompt="blue-haired girl")
        chain = get_chain(story)
        character = chain["character"]

        prompt = character.to_prompt()
        assert "blue-haired" in prompt or "blue" in prompt

    def test_includes_distinguishing_marks(self):
        story = make_story()
        chain = get_chain(story)
        character = chain["character"]
        character.profile.physical.distinguishing_marks = "scar on left cheek"

        prompt = character.to_prompt()
        assert "scar on left cheek" in prompt

    def test_empty_appearance_returns_empty(self):
        story = Story("s1", "Test")
        character = Character("c1", "Nobody")
        story.add_character(character)
        story.create_chapter("Ch", ["c1"])

        assert character.to_prompt() == ""


# ---------------------------------------------------------------------------
# Chapter.to_prompt()
# ---------------------------------------------------------------------------

class TestChapterToPrompt:
    def test_returns_location_and_time(self):
        story = make_story()
        chain = get_chain(story)
        chapter = chain["chapter"]
        chapter.default_location = "enchanted forest"
        chapter.default_time_of_day = "dusk"

        prompt = chapter.to_prompt()
        assert "enchanted forest" in prompt
        assert "dusk" in prompt

    def test_empty_chapter_returns_empty_string(self):
        story = make_story()
        chain = get_chain(story)
        chapter = chain["chapter"]
        chapter.default_location = ""
        chapter.default_time_of_day = ""
        assert chapter.to_prompt() == ""

    def test_location_only(self):
        story = make_story()
        chain = get_chain(story)
        chapter = chain["chapter"]
        chapter.default_location = "space station"
        chapter.default_time_of_day = ""
        assert chapter.to_prompt() == "space station"


# ---------------------------------------------------------------------------
# Script.to_prompt()
# ---------------------------------------------------------------------------

class TestScriptToPrompt:
    def test_returns_all_visual_fields(self):
        story = make_story()
        chain = get_chain(story)
        script = chain["script"]
        script.pose = "standing"
        script.action = "drawing a sword"
        script.emotion = "determined"
        script.outfit = "plate armor"
        script.direction = "facing camera"

        prompt = script.to_prompt()
        assert "standing" in prompt
        assert "holding_sword" in prompt or "sword_fighting" in prompt
        assert "determined" in prompt
        assert "armor" in prompt
        assert "facing_camera" in prompt

    def test_excludes_dialogue(self):
        story = make_story()
        chain = get_chain(story)
        script = chain["script"]
        script.dialogue = "Hello there!"
        script.action = "waving"

        prompt = script.to_prompt()
        assert "Hello there!" not in prompt
        assert "waving" in prompt

    def test_empty_script_returns_empty_string(self):
        story = make_story()
        chain = get_chain(story)
        script = chain["script"]
        script.pose = ""
        script.action = ""
        script.emotion = ""
        script.outfit = ""
        script.direction = ""
        assert script.to_prompt() == ""


# ---------------------------------------------------------------------------
# PromptComposer._compose_direct() — full composition chain
# ---------------------------------------------------------------------------

class TestComposeDirectFullChain:
    """Test the complete composition pipeline with a full hierarchy."""

    def setup_method(self):
        self.composer = PromptComposer()

    def test_full_hierarchy_composition(self):
        """Build Story->Character->Chapter->Page->Panel->Script and verify
        _compose_direct() output contains all parts in correct order."""
        story = make_story(appearance_prompt="blue-haired girl")
        story.art_style = "manga"
        chain = get_chain(story)

        page = chain["page"]
        page.setting = "moonlit garden"
        page.time_of_day = "night"
        page.mood = "romantic"

        panel = chain["panel"]
        panel.shot_type = "medium"
        panel.narration = "they meet at last"

        script = chain["script"]
        script.action = "reaching out"
        script.emotion = "hopeful"

        characters = [chain["character"]]
        prompt = self.composer._compose_direct(panel, characters)

        # Order: story first, then panel, then character+script, then page
        story_index = prompt.find("manga")
        panel_index = prompt.find("medium")
        character_index = prompt.find("blue-haired")
        page_index = prompt.find("moonlit garden")

        assert story_index != -1, "Story art style missing from prompt"
        assert panel_index != -1, "Panel shot type missing from prompt"
        assert character_index != -1, "Character appearance missing from prompt"
        assert page_index != -1, "Page setting missing from prompt"

        # Correct order: story < panel < character < page
        assert story_index < panel_index, "Story should come before panel"
        assert panel_index < character_index, "Panel should come before character"
        assert character_index < page_index, "Character should come before page"

    def test_empty_scripts_excluded(self):
        """Characters with blank scripts should not appear in the prompt."""
        story = make_two_character_story()
        chain = get_chain(story)
        panel = chain["panel"]

        # Get both characters
        characters = list(story.characters.values())

        # Leave all scripts blank (empty) — they should be filtered out
        for script in panel.scripts.values():
            script.pose = ""
            script.action = ""
            script.emotion = ""
            script.outfit = ""
            script.direction = ""

        prompt = self.composer._compose_direct(panel, characters)
        # With no active scripts, no character descriptions should appear
        assert "blue-haired" not in prompt
        assert "tall wolf" not in prompt

    def test_multi_character_and_separator(self):
        """Multiple characters should be joined with AND."""
        story = make_two_character_story()
        chain = get_chain(story)
        panel = chain["panel"]

        # Give both characters active scripts
        for character_id, script in panel.scripts.items():
            script.action = "looking around"

        characters = list(story.characters.values())
        prompt = self.composer._compose_direct(panel, characters)

        assert " AND " in prompt
        assert "blue-haired" in prompt
        assert "wolf anthro" in prompt

    def test_default_style_when_art_style_empty(self):
        """When story.art_style is empty, DEFAULT_ART_STYLE should appear."""
        story = make_story()
        story.art_style = ""  # explicitly empty
        chain = get_chain(story)

        script = chain["script"]
        script.action = "walking"

        panel = chain["panel"]
        characters = [chain["character"]]
        prompt = self.composer._compose_direct(panel, characters)

        assert "cinematic lighting, hyper-detailed textures" in prompt

    def test_custom_art_style_replaces_default(self):
        """When story.art_style is set, it replaces the default."""
        story = make_story()
        story.art_style = "watercolor pastel"
        chain = get_chain(story)

        script = chain["script"]
        script.action = "sitting"

        panel = chain["panel"]
        characters = [chain["character"]]
        prompt = self.composer._compose_direct(panel, characters)

        assert "watercolor pastel" in prompt
        assert "cinematic lighting" not in prompt

    def test_single_character_no_and_separator(self):
        """A single character should not have 'AND' or 'N characters'."""
        story = make_story()
        chain = get_chain(story)

        script = chain["script"]
        script.action = "running"

        panel = chain["panel"]
        characters = [chain["character"]]
        prompt = self.composer._compose_direct(panel, characters)

        assert " AND " not in prompt
        assert "characters" not in prompt.lower().split("cinematic")[0]  # avoid matching art style text

    def test_page_context_falls_back_to_chapter(self):
        """If page has no setting, chapter defaults should be used."""
        story = make_story()
        chain = get_chain(story)

        chapter = chain["chapter"]
        chapter.default_location = "ancient temple"
        chapter.default_time_of_day = "dawn"

        page = chain["page"]
        page.setting = ""
        page.time_of_day = ""
        page.weather = ""
        page.lighting = ""
        page.mood = ""
        page.action_context = ""

        script = chain["script"]
        script.action = "meditating"

        panel = chain["panel"]
        characters = [chain["character"]]
        prompt = self.composer._compose_direct(panel, characters)

        assert "ancient temple" in prompt
        assert "dawn" in prompt


# ---------------------------------------------------------------------------
# DEFAULT_NEGATIVE constant
# ---------------------------------------------------------------------------

class TestDefaultNegative:
    def test_default_negative_value(self):
        expected = (
            "lowres, (worst_quality, bad_quality:1.2), bad_anatomy, sketch, "
            "jpeg_artefacts, signature, watermark, old, oldest, censored, "
            "bar_censor, simple_background"
        )
        assert DEFAULT_NEGATIVE == expected

    def test_panel_generator_returns_default_negative(self):
        from unittest.mock import MagicMock
        from backend.generator.panel_generator import PanelGenerator

        mock_image_generator = MagicMock()
        generator = PanelGenerator(image_generator=mock_image_generator)
        assert generator.compose_negative_prompt() == DEFAULT_NEGATIVE
