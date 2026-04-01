"""Tests for backend.models.appearance — AppearanceProperties, ReferenceImage, Appearance."""

import pytest

from backend.models.appearance import AppearanceProperties, ReferenceImage, Appearance


# ---------------------------------------------------------------------------
# AppearanceProperties
# ---------------------------------------------------------------------------

class TestAppearancePropertiesToPrompt:
    def test_empty_properties_produce_empty_prompt(self):
        properties = AppearanceProperties()
        assert properties.to_prompt() == ""

    def test_single_field_species(self):
        properties = AppearanceProperties(species="elf")
        assert properties.to_prompt() == "elf"

    def test_body_type_appends_build(self):
        properties = AppearanceProperties(body_type="slim")
        assert properties.to_prompt() == "slim build"

    def test_hair_colour_only(self):
        properties = AppearanceProperties(hair_colour="red")
        assert properties.to_prompt() == "red hair"

    def test_hair_style_only(self):
        properties = AppearanceProperties(hair_style="braided")
        assert properties.to_prompt() == "braided hair"

    def test_hair_colour_and_style_combined(self):
        properties = AppearanceProperties(hair_colour="silver", hair_style="long")
        assert properties.to_prompt() == "silver long hair"

    def test_eye_colour_appends_eyes(self):
        properties = AppearanceProperties(eye_colour="green")
        assert properties.to_prompt() == "green eyes"

    def test_outfit_prepends_wearing(self):
        properties = AppearanceProperties(outfit="plate armor")
        assert properties.to_prompt() == "wearing plate armor"

    def test_accessories_prepends_with(self):
        properties = AppearanceProperties(accessories="golden ring")
        assert properties.to_prompt() == "with golden ring"

    def test_full_prompt_ordering(self):
        properties = AppearanceProperties(
            species="human",
            body_type="athletic",
            height="tall",
            skin_tone="dark",
            hair_colour="black",
            hair_style="curly",
            eye_colour="brown",
            facial_features="strong jaw",
            outfit="leather jacket",
            accessories="sunglasses",
            art_style_notes="manga style",
        )
        prompt = properties.to_prompt()
        parts = prompt.split(", ")
        assert parts[0] == "human"
        assert parts[1] == "athletic build"
        assert parts[2] == "tall"
        assert parts[3] == "dark"
        assert "black curly hair" in prompt
        assert "brown eyes" in prompt
        assert "strong jaw" in prompt
        assert "wearing leather jacket" in prompt
        assert "with sunglasses" in prompt
        assert parts[-1] == "manga style"


class TestAppearancePropertiesRoundtrip:
    def test_empty_roundtrip(self):
        original = AppearanceProperties()
        restored = AppearanceProperties.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_full_roundtrip(self):
        original = AppearanceProperties(
            species="orc",
            body_type="bulky",
            height="very tall",
            skin_tone="green",
            hair_style="mohawk",
            hair_colour="black",
            eye_colour="red",
            facial_features="tusks",
            outfit="war paint",
            accessories="bone necklace",
            art_style_notes="dark fantasy",
        )
        restored = AppearanceProperties.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_handles_missing_keys(self):
        properties = AppearanceProperties.from_dict({"species": "dwarf"})
        assert properties.species == "dwarf"
        assert properties.body_type == ""
        assert properties.eye_colour == ""


# ---------------------------------------------------------------------------
# ReferenceImage
# ---------------------------------------------------------------------------

class TestReferenceImageTrainingPrompt:
    def test_empty_labels_produce_empty_prompt(self):
        reference = ReferenceImage(content_hash="abc123")
        assert reference.to_training_prompt() == ""

    def test_caption_only(self):
        reference = ReferenceImage(content_hash="abc", caption="a knight in shining armor")
        assert reference.to_training_prompt() == "a knight in shining armor"

    def test_expression_appends_expression(self):
        reference = ReferenceImage(content_hash="abc", expression="happy")
        assert reference.to_training_prompt() == "happy expression"

    def test_angle_appends_view(self):
        reference = ReferenceImage(content_hash="abc", angle="side")
        assert reference.to_training_prompt() == "side view"

    def test_outfit_variant_prepends_wearing(self):
        reference = ReferenceImage(content_hash="abc", outfit_variant="formal")
        assert reference.to_training_prompt() == "wearing formal"

    def test_tags_are_appended(self):
        reference = ReferenceImage(content_hash="abc", tags=["detailed", "high quality"])
        assert reference.to_training_prompt() == "detailed, high quality"

    def test_all_label_fields(self):
        reference = ReferenceImage(
            content_hash="abc",
            caption="warrior elf",
            pose="standing",
            expression="angry",
            angle="front",
            scene="battlefield",
            outfit_variant="armor",
            tags=["epic"],
        )
        prompt = reference.to_training_prompt()
        assert "warrior elf" in prompt
        assert "standing" in prompt
        assert "angry expression" in prompt
        assert "front view" in prompt
        assert "battlefield" in prompt
        assert "wearing armor" in prompt
        assert "epic" in prompt


class TestReferenceImageRoundtrip:
    def test_minimal_roundtrip(self):
        original = ReferenceImage(content_hash="hash1")
        restored = ReferenceImage.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_full_roundtrip(self):
        original = ReferenceImage(
            content_hash="hash2",
            source="generated",
            accepted=True,
            caption="a cat",
            pose="sitting",
            expression="sleepy",
            angle="3/4",
            scene="indoor cafe",
            outfit_variant="casual",
            tags=["cute", "cozy"],
            notes="user liked this one",
        )
        restored = ReferenceImage.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_accepted_none_survives_roundtrip(self):
        original = ReferenceImage(content_hash="h", accepted=None)
        restored = ReferenceImage.from_dict(original.to_dict())
        assert restored.accepted is None

    def test_accepted_false_survives_roundtrip(self):
        original = ReferenceImage(content_hash="h", accepted=False)
        restored = ReferenceImage.from_dict(original.to_dict())
        assert restored.accepted is False


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------

class TestAppearanceReferenceManagement:
    def test_add_reference_returns_reference(self):
        appearance = Appearance()
        reference = appearance.add_reference("hash1", caption="a dog")
        assert isinstance(reference, ReferenceImage)
        assert reference.content_hash == "hash1"
        assert reference.caption == "a dog"

    def test_add_reference_appends_to_list(self):
        appearance = Appearance()
        appearance.add_reference("h1")
        appearance.add_reference("h2")
        assert len(appearance.references) == 2

    def test_get_reference_found(self):
        appearance = Appearance()
        appearance.add_reference("target", caption="found me")
        result = appearance.get_reference("target")
        assert result is not None
        assert result.caption == "found me"

    def test_get_reference_not_found(self):
        appearance = Appearance()
        assert appearance.get_reference("nonexistent") is None

    def test_remove_reference(self):
        appearance = Appearance()
        appearance.add_reference("keep")
        appearance.add_reference("remove_me")
        appearance.remove_reference("remove_me")
        assert len(appearance.references) == 1
        assert appearance.references[0].content_hash == "keep"

    def test_remove_reference_nonexistent_is_noop(self):
        appearance = Appearance()
        appearance.add_reference("h1")
        appearance.remove_reference("ghost")
        assert len(appearance.references) == 1


class TestAppearanceAcceptReject:
    def test_accept_reference(self):
        appearance = Appearance()
        appearance.add_reference("h1")
        appearance.accept_reference("h1")
        assert appearance.get_reference("h1").accepted is True

    def test_reject_reference(self):
        appearance = Appearance()
        appearance.add_reference("h1")
        appearance.reject_reference("h1")
        assert appearance.get_reference("h1").accepted is False

    def test_accept_nonexistent_is_noop(self):
        appearance = Appearance()
        appearance.accept_reference("ghost")  # should not raise

    def test_accepted_references_filter(self):
        appearance = Appearance()
        appearance.add_reference("h1")
        appearance.add_reference("h2")
        appearance.add_reference("h3")
        appearance.accept_reference("h1")
        appearance.reject_reference("h2")
        assert [ref.content_hash for ref in appearance.accepted_references()] == ["h1"]

    def test_rejected_references_filter(self):
        appearance = Appearance()
        appearance.add_reference("h1")
        appearance.add_reference("h2")
        appearance.accept_reference("h1")
        appearance.reject_reference("h2")
        assert [ref.content_hash for ref in appearance.rejected_references()] == ["h2"]

    def test_unrated_references_filter(self):
        appearance = Appearance()
        appearance.add_reference("h1")
        appearance.add_reference("h2")
        appearance.accept_reference("h1")
        assert [ref.content_hash for ref in appearance.unrated_references()] == ["h2"]


class TestAppearanceTrainingPairs:
    def test_training_pairs_only_accepted(self):
        appearance = Appearance()
        appearance.add_reference("h1", caption="good image", pose="standing")
        appearance.add_reference("h2", caption="bad image")
        appearance.add_reference("h3", caption="unrated image")
        appearance.accept_reference("h1")
        appearance.reject_reference("h2")
        pairs = appearance.training_pairs()
        assert len(pairs) == 1
        content_hash, caption = pairs[0]
        assert content_hash == "h1"
        assert "good image" in caption

    def test_training_pairs_skip_accepted_without_labels(self):
        appearance = Appearance()
        appearance.add_reference("h1")  # no labels at all
        appearance.accept_reference("h1")
        pairs = appearance.training_pairs()
        assert len(pairs) == 0


class TestAppearancePromptDelegation:
    def test_to_prompt_delegates_to_properties(self):
        appearance = Appearance()
        appearance.properties.species = "vampire"
        appearance.properties.eye_colour = "crimson"
        assert appearance.to_prompt() == appearance.properties.to_prompt()
        assert "vampire" in appearance.to_prompt()


class TestAppearanceRoundtrip:
    def test_full_roundtrip(self):
        appearance = Appearance()
        appearance.properties = AppearanceProperties(species="cat", eye_colour="gold")
        appearance.add_reference("h1", caption="portrait", source="upload")
        appearance.accept_reference("h1")
        appearance.adapter_hash = "adapter_weights_001"

        data = appearance.to_dict()
        restored = Appearance.from_dict(data)

        assert restored.properties.species == "cat"
        assert restored.properties.eye_colour == "gold"
        assert len(restored.references) == 1
        assert restored.references[0].accepted is True
        assert restored.adapter_hash == "adapter_weights_001"

    def test_empty_roundtrip(self):
        appearance = Appearance()
        restored = Appearance.from_dict(appearance.to_dict())
        assert restored.to_dict() == appearance.to_dict()
