"""Tests for tag_vocabulary — normalization, detection, and formatting.

Uses injected metadata dicts and fake safetensors files to test
auto-detection without needing real model weights.
"""
import json
import struct
import tempfile
from pathlib import Path

import pytest

from backend.generator.tag_vocabulary import (
    DANBOORU, PONY, E621, NATURAL,
    normalize_tag, normalize_tags, find_closest_tag,
    tags_for_appearance, tags_for_script,
    get_model_profile, get_quality_tags, get_prompt_format, is_tag_model,
    format_prompt, format_negative,
    read_safetensors_metadata, detect_format_from_metadata,
    detect_format_from_file, auto_profile_for_file,
    extract_tag_capabilities, get_field_suggestions,
    _classify_tags,
    POSES, EXPRESSIONS, FRAMING, HAIR_COLORS, HAIR_STYLES,
    EYE_COLORS, SPECIES, CLOTHING, ACCESSORIES, ACTIONS,
    MODEL_PROFILES, DEFAULT_PROFILE,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_safetensors(metadata: dict) -> Path:
    """Create a minimal safetensors file with the given metadata.

    The file has a valid header but no real tensor data.
    """
    header = {
        "__metadata__": {k: str(v) for k, v in metadata.items()},
        "dummy.weight": {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [0, 2],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    tensor_data = b"\x00\x00"  # 2 bytes of dummy data

    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.write(struct.pack("<Q", len(header_bytes)))
    tmp.write(header_bytes)
    tmp.write(tensor_data)
    tmp.close()
    return Path(tmp.name)


def _make_tag_freq(tags: dict[str, int]) -> str:
    """Build a ss_tag_frequency JSON string from a flat tag dict."""
    return json.dumps({"train_data": tags})


# ── Tag normalization ─────────────────────────────────────────────────

class TestNormalizeTag:
    def test_spaces_to_underscores(self):
        assert normalize_tag("long hair") == "long_hair"

    def test_lowercase(self):
        assert normalize_tag("Blue Eyes") == "blue_eyes"

    def test_strips_wearing(self):
        assert normalize_tag("wearing a school uniform") == "school_uniform"

    def test_strips_with(self):
        assert normalize_tag("with glasses") == "glasses"

    def test_strips_articles(self):
        assert normalize_tag("a red dress") == "red_dress"

    def test_removes_special_chars(self):
        assert normalize_tag("cat ears!?") == "cat_ears"

    def test_preserves_parentheses(self):
        assert normalize_tag("(emphasis:1.2)") == "(emphasis12)"

    def test_empty_string(self):
        assert normalize_tag("") == ""

    def test_already_normalized(self):
        assert normalize_tag("long_hair") == "long_hair"


class TestNormalizeTags:
    def test_comma_separated(self):
        result = normalize_tags("long hair, blue eyes, smile")
        assert result == ["long_hair", "blue_eyes", "smile"]

    def test_empty(self):
        assert normalize_tags("") == []

    def test_strips_empty_parts(self):
        result = normalize_tags("smile, , happy")
        assert result == ["smile", "happy"]


class TestFindClosestTag:
    def test_exact_match(self):
        assert find_closest_tag("standing", POSES) == "standing"

    def test_normalized_match(self):
        assert find_closest_tag("Standing", POSES) == "standing"

    def test_substring_match(self):
        result = find_closest_tag("crossed arms", POSES)
        assert result == "crossed_arms"

    def test_partial_word_match(self):
        result = find_closest_tag("ponytail", HAIR_STYLES)
        assert result == "ponytail"

    def test_no_match_returns_normalized(self):
        result = find_closest_tag("something_unknown", POSES)
        assert result == "something_unknown"


# ── Field-to-tag conversion ──────────────────────────────────────────

class TestTagsForAppearance:
    def test_basic_fields(self):
        tags = tags_for_appearance(
            species="human",
            hair_colour="blue",
            eye_colour="green",
        )
        assert "human" in tags
        assert "blue_hair" in tags
        assert "green_eyes" in tags

    def test_outfit_split(self):
        tags = tags_for_appearance(outfit="jacket, boots")
        assert "jacket" in tags
        assert "boots" in tags

    def test_empty_fields_excluded(self):
        tags = tags_for_appearance(species="", hair_colour="")
        assert tags == []


class TestTagsForScript:
    def test_all_fields(self):
        tags = tags_for_script(
            pose="standing",
            action="fighting",
            emotion="angry",
            outfit="armor",
            direction="close-up",
        )
        assert "standing" in tags
        assert "fighting" in tags
        assert "angry" in tags
        assert "armor" in tags
        assert "close-up" in tags

    def test_empty_fields_excluded(self):
        tags = tags_for_script(pose="", emotion="")
        assert tags == []

    def test_freeform_normalized(self):
        tags = tags_for_script(pose="sitting down", emotion="very happy")
        assert len(tags) == 2
        for tag in tags:
            assert " " not in tag


# ── Model profiles ────────────────────────────────────────────────────

class TestModelProfiles:
    def test_known_model_returns_profile(self):
        profile = get_model_profile("Lykon/AAM_XL_AnimeMix")
        assert profile["format"] == DANBOORU
        assert "masterpiece" in profile["positive"]

    def test_pony_model_returns_pony(self):
        profile = get_model_profile("CitronLegacy/ponyDiffusionV6XL_Diffusers")
        assert profile["format"] == PONY
        assert "score_9" in profile["positive"]

    def test_unknown_model_returns_default(self):
        profile = get_model_profile("unknown/model")
        assert profile == DEFAULT_PROFILE

    def test_get_quality_tags(self):
        tags = get_quality_tags("Lykon/AAM_XL_AnimeMix")
        assert "positive" in tags
        assert "negative" in tags

    def test_get_prompt_format(self):
        assert get_prompt_format("CitronLegacy/ponyDiffusionV6XL_Diffusers") == PONY

    def test_is_tag_model_true_for_danbooru(self):
        assert is_tag_model("Lykon/AAM_XL_AnimeMix") is True

    def test_is_tag_model_true_for_pony(self):
        assert is_tag_model("CitronLegacy/ponyDiffusionV6XL_Diffusers") is True


# ── Prompt formatting ─────────────────────────────────────────────────

class TestFormatPrompt:
    def test_danbooru_prepends_quality(self):
        result = format_prompt(["1girl", "blue_hair"], "Lykon/AAM_XL_AnimeMix")
        assert result.startswith("masterpiece")
        assert "1girl" in result
        assert "blue_hair" in result

    def test_pony_prepends_score_tags(self):
        result = format_prompt(["1girl"], "CitronLegacy/ponyDiffusionV6XL_Diffusers")
        assert result.startswith("score_9")
        assert "source_anime" in result
        assert "1girl" in result

    def test_unknown_model_uses_default(self):
        result = format_prompt(["1girl"], "unknown/model")
        assert "masterpiece" in result
        assert "1girl" in result


class TestFormatNegative:
    def test_includes_model_negatives(self):
        result = format_negative("Lykon/AAM_XL_AnimeMix")
        assert "watermark" in result

    def test_extra_tags_appended(self):
        result = format_negative("Lykon/AAM_XL_AnimeMix", extra=["ugly"])
        assert "ugly" in result

    def test_deduplicates(self):
        result = format_negative(
            "Lykon/AAM_XL_AnimeMix",
            extra=["watermark", "custom_tag"],
        )
        assert result.count("watermark") == 1
        assert "custom_tag" in result


# ── Tag classification ────────────────────────────────────────────────

class TestClassifyTags:
    def test_pony_detected(self):
        tags = {"score_9", "1girl", "blue_hair", "smile"}
        assert _classify_tags(tags) == PONY

    def test_e621_detected(self):
        tags = {"anthro", "canine", "rating_safe", "male", "fur"}
        assert _classify_tags(tags) == E621

    def test_danbooru_detected(self):
        tags = {"1girl", "solo", "blue_hair", "smile", "school_uniform"}
        assert _classify_tags(tags) == DANBOORU

    def test_natural_language_detected(self):
        tags = {
            "a beautiful girl standing in a sunlit meadow",
            "she is wearing a flowing white dress with flowers",
            "the background shows rolling green hills and blue sky",
        }
        assert _classify_tags(tags) == NATURAL

    def test_empty_returns_none(self):
        assert _classify_tags(set()) is None

    def test_unknown_tags_return_none(self):
        tags = {"xyz_custom_1", "abc_custom_2"}
        assert _classify_tags(tags) is None

    def test_pony_takes_priority_over_danbooru(self):
        """Pony models also have Danbooru tags but score_9 is the giveaway."""
        tags = {"score_9", "score_8_up", "1girl", "solo", "masterpiece"}
        assert _classify_tags(tags) == PONY

    def test_e621_needs_two_markers(self):
        """Single e621-ish tag isn't enough — could be coincidence."""
        tags = {"anthro", "smile", "blue_hair"}
        assert _classify_tags(tags) != E621


# ── Metadata detection ────────────────────────────────────────────────

class TestDetectFormatFromMetadata:
    def test_pony_from_tag_frequency(self):
        metadata = {
            "ss_tag_frequency": _make_tag_freq({
                "score_9": 100, "score_8_up": 80,
                "1girl": 50, "blue_hair": 30,
            }),
        }
        assert detect_format_from_metadata(metadata) == PONY

    def test_danbooru_from_tag_frequency(self):
        metadata = {
            "ss_tag_frequency": _make_tag_freq({
                "1girl": 100, "solo": 80,
                "blue_hair": 30, "smile": 20,
            }),
        }
        assert detect_format_from_metadata(metadata) == DANBOORU

    def test_e621_from_tag_frequency(self):
        metadata = {
            "ss_tag_frequency": _make_tag_freq({
                "anthro": 100, "canine": 50,
                "rating_safe": 80, "male": 30,
            }),
        }
        assert detect_format_from_metadata(metadata) == E621

    def test_natural_from_tag_frequency(self):
        metadata = {
            "ss_tag_frequency": _make_tag_freq({
                "a girl standing in a beautiful garden": 10,
                "she is wearing a long blue dress": 8,
                "the sun is setting behind the mountains": 5,
            }),
        }
        assert detect_format_from_metadata(metadata) == NATURAL

    def test_pony_from_base_model_name(self):
        metadata = {"ss_sd_model_name": "ponyDiffusionV6XL.safetensors"}
        assert detect_format_from_metadata(metadata) == PONY

    def test_danbooru_from_base_model_name(self):
        metadata = {"ss_sd_model_name": "animagine-xl-3.1.safetensors"}
        assert detect_format_from_metadata(metadata) == DANBOORU

    def test_e621_from_base_model_name(self):
        metadata = {"ss_sd_model_name": "nova-furry-xl.safetensors"}
        assert detect_format_from_metadata(metadata) == E621

    def test_danbooru_from_modelspec_title(self):
        metadata = {"modelspec.title": "My Custom Anime Model"}
        assert detect_format_from_metadata(metadata) == DANBOORU

    def test_pony_from_modelspec_title(self):
        metadata = {"modelspec.title": "Pony Realism V5"}
        assert detect_format_from_metadata(metadata) == PONY

    def test_empty_metadata_returns_none(self):
        assert detect_format_from_metadata({}) is None

    def test_irrelevant_metadata_returns_none(self):
        metadata = {"some_key": "some_value", "another": "thing"}
        assert detect_format_from_metadata(metadata) is None

    def test_malformed_tag_frequency_handled(self):
        metadata = {"ss_tag_frequency": "not valid json {{{"}
        assert detect_format_from_metadata(metadata) is None

    def test_tag_frequency_priority_over_model_name(self):
        """If tag_frequency says Pony but model name says anime, trust tags."""
        metadata = {
            "ss_tag_frequency": _make_tag_freq({
                "score_9": 100, "score_8_up": 80,
            }),
            "ss_sd_model_name": "animagine-xl.safetensors",
        }
        assert detect_format_from_metadata(metadata) == PONY


# ── Safetensors file reading ─────────────────────────────────────────

class TestReadSafetensorsMetadata:
    def test_reads_metadata_from_file(self):
        path = _make_safetensors({"format": "pt", "author": "test"})
        try:
            meta = read_safetensors_metadata(path)
            assert meta["format"] == "pt"
            assert meta["author"] == "test"
        finally:
            path.unlink()

    def test_empty_metadata(self):
        header = {
            "dummy.weight": {
                "dtype": "F16", "shape": [1], "data_offsets": [0, 2],
            },
        }
        header_bytes = json.dumps(header).encode("utf-8")
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        tmp.write(struct.pack("<Q", len(header_bytes)))
        tmp.write(header_bytes)
        tmp.write(b"\x00\x00")
        tmp.close()
        try:
            meta = read_safetensors_metadata(tmp.name)
            assert meta == {}
        finally:
            Path(tmp.name).unlink()

    def test_nonexistent_file_returns_empty(self):
        meta = read_safetensors_metadata("/tmp/does_not_exist.safetensors")
        assert meta == {}

    def test_corrupt_file_returns_empty(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        tmp.write(b"this is not a safetensors file")
        tmp.close()
        try:
            meta = read_safetensors_metadata(tmp.name)
            assert meta == {}
        finally:
            Path(tmp.name).unlink()


class TestDetectFormatFromFile:
    def test_pony_lora(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "score_9": 50, "score_8_up": 40, "1girl": 30,
            }),
        })
        try:
            assert detect_format_from_file(path) == PONY
        finally:
            path.unlink()

    def test_danbooru_lora(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "1girl": 100, "solo": 80, "looking_at_viewer": 60,
            }),
        })
        try:
            assert detect_format_from_file(path) == DANBOORU
        finally:
            path.unlink()

    def test_no_metadata_returns_none(self):
        header = {
            "dummy.weight": {
                "dtype": "F16", "shape": [1], "data_offsets": [0, 2],
            },
        }
        header_bytes = json.dumps(header).encode("utf-8")
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        tmp.write(struct.pack("<Q", len(header_bytes)))
        tmp.write(header_bytes)
        tmp.write(b"\x00\x00")
        tmp.close()
        try:
            assert detect_format_from_file(tmp.name) is None
        finally:
            Path(tmp.name).unlink()


class TestAutoProfileForFile:
    def test_pony_profile(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "score_9": 50, "source_anime": 40,
            }),
        })
        try:
            profile = auto_profile_for_file(path)
            assert profile["format"] == PONY
            assert "score_9" in profile["positive"]
            assert profile["detected_from"] == "metadata"
        finally:
            path.unlink()

    def test_danbooru_profile(self):
        path = _make_safetensors({
            "ss_sd_model_name": "animagine-xl-3.1.safetensors",
        })
        try:
            profile = auto_profile_for_file(path)
            assert profile["format"] == DANBOORU
            assert "masterpiece" in profile["positive"]
            assert profile["detected_from"] == "metadata"
        finally:
            path.unlink()

    def test_fallback_profile(self):
        path = _make_safetensors({"random_key": "random_value"})
        try:
            profile = auto_profile_for_file(path)
            assert profile["format"] == DANBOORU
            assert profile["detected_from"] == "fallback"
        finally:
            path.unlink()

    def test_natural_language_profile(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "a beautiful landscape with mountains and rivers": 10,
                "sunset over the ocean with dramatic clouds": 8,
                "a person walking through a field of flowers": 5,
            }),
        })
        try:
            profile = auto_profile_for_file(path)
            assert profile["format"] == NATURAL
            assert profile["positive"] == []
            assert profile["detected_from"] == "metadata"
        finally:
            path.unlink()

    def test_e621_profile(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "anthro": 100, "canine": 50,
                "rating_safe": 80, "feral": 20,
            }),
        })
        try:
            profile = auto_profile_for_file(path)
            assert profile["format"] == E621
            assert profile["detected_from"] == "metadata"
        finally:
            path.unlink()


# ── Tag capabilities extraction ───────────────────────────────────────

class TestExtractTagCapabilities:
    def test_classifies_tags_into_categories(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "standing": 100, "sitting": 80,
                "smile": 60, "angry": 40,
                "school_uniform": 50, "dress": 30,
                "upper_body": 20,
                "blue_eyes": 15,
                "long_hair": 10,
                "custom_tag": 5,
            }),
        })
        try:
            caps = extract_tag_capabilities(path)
            assert "pose" in caps
            assert "standing" in caps["pose"]["top_tags"]
            assert "expression" in caps
            assert "smile" in caps["expression"]["top_tags"]
            assert "clothing" in caps
            assert "school_uniform" in caps["clothing"]["top_tags"]
            assert "framing" in caps
            assert "upper_body" in caps["framing"]["top_tags"]
            assert "eye_color" in caps
            assert "hair_style" in caps
            assert "other" in caps
            assert "custom_tag" in caps["other"]["top_tags"]
        finally:
            path.unlink()

    def test_sorted_by_frequency(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "standing": 10, "sitting": 100, "kneeling": 50,
            }),
        })
        try:
            caps = extract_tag_capabilities(path)
            pose_tags = caps["pose"]["top_tags"]
            assert pose_tags[0] == "sitting"
            assert pose_tags[1] == "kneeling"
            assert pose_tags[2] == "standing"
        finally:
            path.unlink()

    def test_empty_without_tag_frequency(self):
        path = _make_safetensors({"random_key": "value"})
        try:
            caps = extract_tag_capabilities(path)
            assert caps == {}
        finally:
            path.unlink()

    def test_counts_per_category(self):
        path = _make_safetensors({
            "ss_tag_frequency": _make_tag_freq({
                "standing": 100, "sitting": 80, "kneeling": 60,
                "smile": 50,
            }),
        })
        try:
            caps = extract_tag_capabilities(path)
            assert caps["pose"]["count"] == 3
            assert caps["pose"]["total_frequency"] == 240
            assert caps["expression"]["count"] == 1
        finally:
            path.unlink()


class TestGetFieldSuggestions:
    def test_returns_canonical_for_unknown_model(self):
        suggestions = get_field_suggestions("unknown/model", "pose")
        assert len(suggestions) > 0
        assert all(tag in POSES for tag in suggestions)

    def test_returns_canonical_for_expression(self):
        suggestions = get_field_suggestions("unknown/model", "expression")
        assert len(suggestions) > 0
        assert all(tag in EXPRESSIONS for tag in suggestions)

    def test_unknown_field_returns_empty(self):
        suggestions = get_field_suggestions("unknown/model", "nonexistent_field")
        assert suggestions == []

    def test_limit_respected(self):
        suggestions = get_field_suggestions("unknown/model", "pose", limit=5)
        assert len(suggestions) <= 5

    def test_emotion_maps_to_expression(self):
        """Our field is 'emotion' but the category is 'expression'."""
        suggestions = get_field_suggestions("unknown/model", "emotion")
        assert len(suggestions) > 0

    def test_direction_maps_to_framing(self):
        suggestions = get_field_suggestions("unknown/model", "direction")
        assert len(suggestions) > 0
