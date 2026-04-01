"""Tests for ComicComposer — layout computation and panel layout data."""

import pytest

from backend.composer.comic_composer import ComicComposer
from backend.models.page import Page
from backend.models.panel import Panel
from backend.models.script import Script
from tests.helpers import make_story, get_chain


@pytest.fixture
def composer():
    return ComicComposer()


@pytest.fixture
def story_with_chain():
    """A story with one character, chapter, page, panel, script."""
    story = make_story()
    return get_chain(story)


# ---------- LAYOUT_TEMPLATES ----------


class TestLayoutTemplates:
    def test_known_templates_present(self, composer):
        expected_names = [
            "auto", "single", "two_equal", "three_row",
            "hero_top", "two_over_one", "grid_2x2", "asymmetric",
        ]
        for name in expected_names:
            assert name in composer.LAYOUT_TEMPLATES

    def test_auto_template_is_none(self, composer):
        assert composer.LAYOUT_TEMPLATES["auto"] is None

    def test_single_template(self, composer):
        assert composer.LAYOUT_TEMPLATES["single"] == [[1.0]]

    def test_two_equal_template(self, composer):
        assert composer.LAYOUT_TEMPLATES["two_equal"] == [[0.5, 0.5]]

    def test_grid_2x2_template(self, composer):
        assert composer.LAYOUT_TEMPLATES["grid_2x2"] == [[0.5, 0.5], [0.5, 0.5]]

    def test_hero_top_template(self, composer):
        assert composer.LAYOUT_TEMPLATES["hero_top"] == [[1.0], [0.5, 0.5]]

    def test_two_over_one_template(self, composer):
        assert composer.LAYOUT_TEMPLATES["two_over_one"] == [[0.5, 0.5], [1.0]]

    def test_asymmetric_template(self, composer):
        assert composer.LAYOUT_TEMPLATES["asymmetric"] == [[0.6, 0.4], [0.4, 0.6]]


# ---------- _auto_layout ----------


class TestAutoLayout:
    def test_zero_panels_returns_single(self, composer):
        assert composer._auto_layout(0) == "single"

    def test_one_panel_returns_single(self, composer):
        assert composer._auto_layout(1) == "single"

    def test_two_panels_returns_two_equal(self, composer):
        assert composer._auto_layout(2) == "two_equal"

    def test_three_panels_returns_hero_top(self, composer):
        assert composer._auto_layout(3) == "hero_top"

    def test_four_panels_returns_grid_2x2(self, composer):
        assert composer._auto_layout(4) == "grid_2x2"

    def test_five_panels_returns_asymmetric(self, composer):
        assert composer._auto_layout(5) == "asymmetric"

    def test_large_count_returns_asymmetric(self, composer):
        assert composer._auto_layout(10) == "asymmetric"


# ---------- _auto_layout_grid ----------


class TestAutoLayoutGrid:
    def test_one_panel_grid(self, composer):
        grid = composer._auto_layout_grid(1)
        assert grid == [[1.0]]

    def test_two_panel_grid(self, composer):
        grid = composer._auto_layout_grid(2)
        assert grid == [[0.5, 0.5]]

    def test_three_panel_grid(self, composer):
        grid = composer._auto_layout_grid(3)
        assert grid == [[0.33, 0.33, 0.33]]

    def test_four_panel_grid(self, composer):
        grid = composer._auto_layout_grid(4)
        # 3 columns: first row [0.33, 0.33, 0.33], second row [0.33]
        assert len(grid) == 2
        assert grid[0] == [0.33, 0.33, 0.33]
        assert grid[1] == [0.33]

    def test_five_panel_grid(self, composer):
        grid = composer._auto_layout_grid(5)
        assert len(grid) == 2
        assert grid[0] == [0.33, 0.33, 0.33]
        assert grid[1] == [0.33, 0.33]

    def test_six_panel_grid(self, composer):
        grid = composer._auto_layout_grid(6)
        assert len(grid) == 2
        assert grid[0] == [0.33, 0.33, 0.33]
        assert grid[1] == [0.33, 0.33, 0.33]

    def test_seven_panel_grid(self, composer):
        grid = composer._auto_layout_grid(7)
        assert len(grid) == 3
        assert grid[0] == [0.33, 0.33, 0.33]
        assert grid[1] == [0.33, 0.33, 0.33]
        assert grid[2] == [0.33]

    def test_grid_row_widths_sum_approximately_one(self, composer):
        """Each full row should sum to approximately 1.0."""
        for panel_count in range(1, 10):
            grid = composer._auto_layout_grid(panel_count)
            columns = min(panel_count, 3)
            for row in grid:
                if len(row) == columns:
                    total = sum(row)
                    assert abs(total - 1.0) < 0.05, (
                        f"Full row for {panel_count} panels sums to {total}"
                    )


# ---------- _panel_layout ----------


class TestPanelLayout:
    def test_returns_required_fields(self, composer, story_with_chain):
        panel = story_with_chain["panel"]
        result = composer._panel_layout(panel)

        required_fields = [
            "panel_id", "image_hash", "video_hash",
            "is_animated", "dialogues", "narration", "shot_type",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_panel_id_matches(self, composer, story_with_chain):
        panel = story_with_chain["panel"]
        result = composer._panel_layout(panel)
        assert result["panel_id"] == panel.panel_id

    def test_defaults_for_new_panel(self, composer, story_with_chain):
        panel = story_with_chain["panel"]
        result = composer._panel_layout(panel)
        assert result["image_hash"] is None
        assert result["video_hash"] is None
        assert result["is_animated"] is False

    def test_dialogues_empty_when_no_dialogue_set(self, composer, story_with_chain):
        panel = story_with_chain["panel"]
        result = composer._panel_layout(panel)
        assert result["dialogues"] == []

    def test_dialogues_collected_from_scripts(self, composer, story_with_chain):
        panel = story_with_chain["panel"]
        script = story_with_chain["script"]
        character = story_with_chain["character"]

        script.update(dialogue="Hello, world!")
        result = composer._panel_layout(panel)

        assert len(result["dialogues"]) == 1
        assert result["dialogues"][0]["character_id"] == character.character_id
        assert result["dialogues"][0]["text"] == "Hello, world!"

    def test_multiple_dialogues_from_multiple_scripts(self, composer):
        """Two characters with dialogue produce two dialogue entries."""
        from tests.helpers import make_two_character_story, get_chain
        story = make_two_character_story()
        chain = get_chain(story)
        panel = chain["panel"]

        # Both characters should have scripts in this panel
        scripts = list(panel.scripts.values())
        assert len(scripts) >= 2

        scripts[0].update(dialogue="First line")
        scripts[1].update(dialogue="Second line")

        result = composer._panel_layout(panel)
        assert len(result["dialogues"]) == 2
        texts = {dialogue["text"] for dialogue in result["dialogues"]}
        assert "First line" in texts
        assert "Second line" in texts

    def test_narration_passed_through(self, composer, story_with_chain):
        panel = story_with_chain["panel"]
        panel.narration = "Meanwhile, in the shadows..."
        result = composer._panel_layout(panel)
        assert result["narration"] == "Meanwhile, in the shadows..."

    def test_shot_type_passed_through(self, composer, story_with_chain):
        panel = story_with_chain["panel"]
        panel.shot_type = "close-up"
        result = composer._panel_layout(panel)
        assert result["shot_type"] == "close-up"


# ---------- compute_layout ----------


class TestComputeLayout:
    def test_returns_correct_structure(self, composer, story_with_chain):
        page = story_with_chain["page"]
        result = composer.compute_layout(page)

        assert "page_id" in result
        assert "template" in result
        assert "grid" in result
        assert "panels" in result

    def test_page_id_matches(self, composer, story_with_chain):
        page = story_with_chain["page"]
        result = composer.compute_layout(page)
        assert result["page_id"] == page.page_id

    def test_auto_layout_single_panel(self, composer, story_with_chain):
        page = story_with_chain["page"]
        # Default story has 1 panel
        result = composer.compute_layout(page)
        assert result["template"] == "single"
        assert result["grid"] == [[1.0]]

    def test_panels_list_matches_page_panels(self, composer, story_with_chain):
        page = story_with_chain["page"]
        result = composer.compute_layout(page)
        assert len(result["panels"]) == len(page.panels)

    def test_explicit_template_used(self, composer, story_with_chain):
        page = story_with_chain["page"]
        page.layout_template = "grid_2x2"
        result = composer.compute_layout(page)
        assert result["template"] == "grid_2x2"
        assert result["grid"] == [[0.5, 0.5], [0.5, 0.5]]

    def test_auto_selects_two_equal_for_two_panels(self, composer, story_with_chain):
        page = story_with_chain["page"]
        chapter = story_with_chain["chapter"]
        # Add a second panel
        page.create_panel(character_ids=chapter.character_ids)
        result = composer.compute_layout(page)
        assert result["template"] == "two_equal"

    def test_auto_selects_hero_top_for_three_panels(self, composer, story_with_chain):
        page = story_with_chain["page"]
        chapter = story_with_chain["chapter"]
        page.create_panel(character_ids=chapter.character_ids)
        page.create_panel(character_ids=chapter.character_ids)
        result = composer.compute_layout(page)
        assert result["template"] == "hero_top"

    def test_auto_selects_grid_2x2_for_four_panels(self, composer, story_with_chain):
        page = story_with_chain["page"]
        chapter = story_with_chain["chapter"]
        for _ in range(3):
            page.create_panel(character_ids=chapter.character_ids)
        result = composer.compute_layout(page)
        assert result["template"] == "grid_2x2"

    def test_unknown_template_falls_back_to_auto_grid(self, composer, story_with_chain):
        page = story_with_chain["page"]
        page.layout_template = "nonexistent_template"
        result = composer.compute_layout(page)
        # Falls back to _auto_layout_grid because LAYOUT_TEMPLATES.get returns None
        assert result["grid"] is not None
        assert isinstance(result["grid"], list)
