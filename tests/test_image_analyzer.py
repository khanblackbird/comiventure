"""Tests for ImageAnalyzer response parsing.

Tests the _parse_character() and _parse_art_style() methods directly,
without needing an ollama connection. Verifies correct handling of
various LLM response formats, missing fields, and excluded values.
"""
import pytest

from backend.generator.image_analyzer import (
    ImageAnalyzer,
    CharacterAnalysis,
    ArtStyleAnalysis,
    ImageAnalysis,
)


@pytest.fixture
def analyzer():
    """Create an ImageAnalyzer (no network calls needed for parsing tests)."""
    return ImageAnalyzer(ollama_host="http://localhost:11434")


# ---------------------------------------------------------------------------
# CharacterAnalysis dataclass defaults
# ---------------------------------------------------------------------------

class TestCharacterAnalysisDefaults:
    def test_all_fields_default_to_empty(self):
        analysis = CharacterAnalysis()
        assert analysis.species == ""
        assert analysis.body_type == ""
        assert analysis.height == ""
        assert analysis.skin_tone == ""
        assert analysis.hair_style == ""
        assert analysis.hair_colour == ""
        assert analysis.eye_colour == ""
        assert analysis.facial_features == ""
        assert analysis.outfit == ""
        assert analysis.accessories == ""
        assert analysis.pose == ""
        assert analysis.expression == ""
        assert analysis.caption == ""

    def test_caption_can_be_set(self):
        analysis = CharacterAnalysis(caption="a tall warrior")
        assert analysis.caption == "a tall warrior"


# ---------------------------------------------------------------------------
# ArtStyleAnalysis dataclass defaults
# ---------------------------------------------------------------------------

class TestArtStyleAnalysisDefaults:
    def test_all_fields_default_to_empty(self):
        analysis = ArtStyleAnalysis()
        assert analysis.art_style == ""
        assert analysis.colour_palette == ""
        assert analysis.line_style == ""
        assert analysis.rendering == ""
        assert analysis.genre_hints == ""
        assert analysis.caption == ""

    def test_caption_can_be_set(self):
        analysis = ArtStyleAnalysis(caption="vibrant anime style")
        assert analysis.caption == "vibrant anime style"


# ---------------------------------------------------------------------------
# ImageAnalysis dataclass defaults
# ---------------------------------------------------------------------------

class TestImageAnalysisDefaults:
    def test_nested_defaults(self):
        analysis = ImageAnalysis()
        assert isinstance(analysis.character, CharacterAnalysis)
        assert isinstance(analysis.art_style, ArtStyleAnalysis)
        assert analysis.raw_caption == ""


# ---------------------------------------------------------------------------
# _parse_character() — clean format
# ---------------------------------------------------------------------------

class TestParseCharacterCleanFormat:
    def test_parses_all_fields(self, analyzer):
        llm_response = (
            "species: human\n"
            "body_type: slim\n"
            "height: tall\n"
            "skin_tone: pale\n"
            "hair_style: long ponytail\n"
            "hair_colour: silver\n"
            "eye_colour: green\n"
            "facial_features: sharp jawline, high cheekbones\n"
            "outfit: leather armor with a cape\n"
            "accessories: silver pendant\n"
            "pose: standing with arms crossed\n"
            "expression: confident smirk"
        )
        result = analyzer._parse_character(llm_response, "test caption")

        assert result.species == "human"
        assert result.body_type == "slim"
        assert result.height == "tall"
        assert result.skin_tone == "pale"
        assert result.hair_style == "long ponytail"
        assert result.hair_colour == "silver"
        assert result.eye_colour == "green"
        assert result.facial_features == "sharp jawline, high cheekbones"
        assert result.outfit == "leather armor with a cape"
        assert result.accessories == "silver pendant"
        assert result.pose == "standing with arms crossed"
        assert result.expression == "confident smirk"
        assert result.caption == "test caption"

    def test_preserves_caption(self, analyzer):
        result = analyzer._parse_character("species: cat", "the original caption")
        assert result.caption == "the original caption"


# ---------------------------------------------------------------------------
# _parse_character() — messy LLM formats
# ---------------------------------------------------------------------------

class TestParseCharacterMessyFormats:
    def test_extra_whitespace(self, analyzer):
        llm_response = "  species:   wolf  \n  body_type:  muscular  "
        result = analyzer._parse_character(llm_response, "")
        assert result.species == "wolf"
        assert result.body_type == "muscular"

    def test_parenthesised_values(self, analyzer):
        llm_response = "species: (fox)\nheight: (short)"
        result = analyzer._parse_character(llm_response, "")
        assert result.species == "fox"
        assert result.height == "short"

    def test_quoted_values(self, analyzer):
        llm_response = 'species: "human"\noutfit: "red dress"'
        result = analyzer._parse_character(llm_response, "")
        assert result.species == "human"
        assert result.outfit == "red dress"

    def test_extra_text_before_fields(self, analyzer):
        """LLM sometimes includes preamble before the structured data."""
        llm_response = (
            "Here are the character details:\n\n"
            "species: elf\n"
            "body_type: slender\n"
            "height: average\n"
        )
        result = analyzer._parse_character(llm_response, "")
        assert result.species == "elf"
        assert result.body_type == "slender"
        assert result.height == "average"

    def test_colon_in_value(self, analyzer):
        """Value itself contains a colon."""
        llm_response = "outfit: armor: plate with chain mail underneath"
        result = analyzer._parse_character(llm_response, "")
        assert result.outfit == "armor: plate with chain mail underneath"

    def test_mixed_case_field_names(self, analyzer):
        llm_response = "Species: dragon\nBody_Type: large\nHAIR_COLOUR: red"
        result = analyzer._parse_character(llm_response, "")
        assert result.species == "dragon"
        assert result.body_type == "large"
        assert result.hair_colour == "red"


# ---------------------------------------------------------------------------
# _parse_character() — excluded values
# ---------------------------------------------------------------------------

class TestParseCharacterExcludedValues:
    def test_na_excluded(self, analyzer):
        llm_response = "species: n/a\nbody_type: muscular"
        result = analyzer._parse_character(llm_response, "")
        assert result.species == ""
        assert result.body_type == "muscular"

    def test_none_excluded(self, analyzer):
        llm_response = "species: none\nheight: tall"
        result = analyzer._parse_character(llm_response, "")
        assert result.species == ""
        assert result.height == "tall"

    def test_not_visible_excluded(self, analyzer):
        llm_response = "eye_colour: not visible\nhair_colour: blonde"
        result = analyzer._parse_character(llm_response, "")
        assert result.eye_colour == ""
        assert result.hair_colour == "blonde"

    def test_not_applicable_excluded(self, analyzer):
        llm_response = "accessories: not applicable\npose: sitting"
        result = analyzer._parse_character(llm_response, "")
        assert result.accessories == ""
        assert result.pose == "sitting"

    def test_empty_value_excluded(self, analyzer):
        llm_response = "species: \nbody_type: slim"
        result = analyzer._parse_character(llm_response, "")
        assert result.species == ""
        assert result.body_type == "slim"

    def test_blank_fields_remain_empty(self, analyzer):
        """Fields not mentioned at all stay as empty strings."""
        llm_response = "species: cat"
        result = analyzer._parse_character(llm_response, "")
        assert result.species == "cat"
        assert result.body_type == ""
        assert result.height == ""
        assert result.outfit == ""


# ---------------------------------------------------------------------------
# _parse_art_style() — clean format
# ---------------------------------------------------------------------------

class TestParseArtStyleCleanFormat:
    def test_parses_all_fields(self, analyzer):
        llm_response = (
            "art_style: manga\n"
            "colour_palette: vibrant\n"
            "line_style: thick outlines\n"
            "rendering: flat colour\n"
            "genre_hints: fantasy"
        )
        result = analyzer._parse_art_style(llm_response, "test caption")

        assert result.art_style == "manga"
        assert result.colour_palette == "vibrant"
        assert result.line_style == "thick outlines"
        assert result.rendering == "flat colour"
        assert result.genre_hints == "fantasy"
        assert result.caption == "test caption"


# ---------------------------------------------------------------------------
# _parse_art_style() — messy LLM formats
# ---------------------------------------------------------------------------

class TestParseArtStyleMessyFormats:
    def test_extra_whitespace(self, analyzer):
        llm_response = "  art_style:   anime  \n  rendering:  cel shaded  "
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == "anime"
        assert result.rendering == "cel shaded"

    def test_parenthesised_values(self, analyzer):
        llm_response = "art_style: (watercolor)\ngenre_hints: (sci-fi)"
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == "watercolor"
        assert result.genre_hints == "sci-fi"

    def test_quoted_values(self, analyzer):
        llm_response = 'art_style: "pixel art"\nline_style: "no outlines"'
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == "pixel art"
        assert result.line_style == "no outlines"

    def test_extra_preamble_text(self, analyzer):
        llm_response = (
            "Based on the description, here are the art style details:\n\n"
            "art_style: western comic\n"
            "colour_palette: muted\n"
        )
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == "western comic"
        assert result.colour_palette == "muted"

    def test_mixed_case_field_names(self, analyzer):
        llm_response = "Art_Style: realistic\nColour_Palette: warm"
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == "realistic"
        assert result.colour_palette == "warm"


# ---------------------------------------------------------------------------
# _parse_art_style() — excluded values
# ---------------------------------------------------------------------------

class TestParseArtStyleExcludedValues:
    def test_na_excluded(self, analyzer):
        llm_response = "art_style: manga\ngenre_hints: n/a"
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == "manga"
        assert result.genre_hints == ""

    def test_none_excluded(self, analyzer):
        llm_response = "art_style: none\ncolour_palette: vibrant"
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == ""
        assert result.colour_palette == "vibrant"

    def test_empty_value_excluded(self, analyzer):
        llm_response = "art_style: \nrendering: detailed shading"
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == ""
        assert result.rendering == "detailed shading"

    def test_missing_fields_stay_empty(self, analyzer):
        llm_response = "art_style: anime"
        result = analyzer._parse_art_style(llm_response, "")
        assert result.art_style == "anime"
        assert result.colour_palette == ""
        assert result.line_style == ""
        assert result.rendering == ""
        assert result.genre_hints == ""

    def test_completely_empty_response(self, analyzer):
        result = analyzer._parse_art_style("", "caption text")
        assert result.art_style == ""
        assert result.colour_palette == ""
        assert result.caption == "caption text"


# ---------------------------------------------------------------------------
# Edge cases: note that "not visible" is only excluded for character parsing
# ---------------------------------------------------------------------------

class TestArtStyleNotVisibleNotExcluded:
    """Art style parser only excludes '', 'n/a', and 'none' — not 'not visible'."""

    def test_not_visible_is_kept_in_art_style(self, analyzer):
        llm_response = "line_style: not visible"
        result = analyzer._parse_art_style(llm_response, "")
        # "not visible" is NOT in the art style exclusion list
        assert result.line_style == "not visible"
