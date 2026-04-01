"""Tests for backend.models.profile — PhysicalTraits, Outfit, Profile."""

import pytest

from backend.models.profile import PhysicalTraits, Outfit, Profile


# ---------------------------------------------------------------------------
# PhysicalTraits
# ---------------------------------------------------------------------------

class TestPhysicalTraits:
    def test_empty_roundtrip(self):
        original = PhysicalTraits()
        restored = PhysicalTraits.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_full_roundtrip(self):
        original = PhysicalTraits(
            body="lean muscular build, 5'8\"",
            face="sharp jawline, amber eyes",
            distinguishing_marks="gold tattoo on left rump",
            hair_fur="black fur with orange stripes",
            voice="low raspy voice",
        )
        restored = PhysicalTraits.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_handles_missing_keys(self):
        traits = PhysicalTraits.from_dict({"body": "tall"})
        assert traits.body == "tall"
        assert traits.face == ""
        assert traits.voice == ""


# ---------------------------------------------------------------------------
# Outfit
# ---------------------------------------------------------------------------

class TestOutfit:
    def test_roundtrip(self):
        original = Outfit(name="casual", description="torn jeans and a t-shirt", is_default=True)
        restored = Outfit.from_dict(original.to_dict())
        assert restored.name == "casual"
        assert restored.description == "torn jeans and a t-shirt"
        assert restored.is_default is True

    def test_default_is_false_by_default(self):
        outfit = Outfit(name="armor", description="plate mail")
        assert outfit.is_default is False

    def test_from_dict_missing_is_default(self):
        outfit = Outfit.from_dict({"name": "robe", "description": "silk robe"})
        assert outfit.is_default is False


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class TestProfileOutfits:
    def test_add_outfit_returns_outfit(self):
        profile = Profile()
        outfit = profile.add_outfit("casual", "jeans and hoodie")
        assert isinstance(outfit, Outfit)
        assert outfit.name == "casual"

    def test_default_outfit_returns_first_when_none_marked(self):
        profile = Profile()
        profile.add_outfit("casual", "jeans")
        profile.add_outfit("formal", "suit")
        assert profile.default_outfit().name == "casual"

    def test_default_outfit_returns_marked_default(self):
        profile = Profile()
        profile.add_outfit("casual", "jeans")
        profile.add_outfit("formal", "suit", is_default=True)
        assert profile.default_outfit().name == "formal"

    def test_add_outfit_with_default_clears_previous_default(self):
        profile = Profile()
        profile.add_outfit("casual", "jeans", is_default=True)
        profile.add_outfit("formal", "suit", is_default=True)
        assert profile.outfits[0].is_default is False
        assert profile.outfits[1].is_default is True

    def test_default_outfit_returns_none_when_empty(self):
        profile = Profile()
        assert profile.default_outfit() is None


class TestProfileExpressionsAndRelationships:
    def test_set_and_get_expression(self):
        profile = Profile()
        profile.set_expression("angry", "ears flatten, tail lashes")
        assert profile.get_expression("angry") == "ears flatten, tail lashes"

    def test_get_expression_missing_returns_empty(self):
        profile = Profile()
        assert profile.get_expression("sad") == ""

    def test_set_relationship(self):
        profile = Profile()
        profile.set_relationship("char_2", "rival turned lover")
        assert profile.relationships["char_2"] == "rival turned lover"


class TestProfileToLlmContext:
    def test_empty_profile_produces_empty_context(self):
        profile = Profile()
        assert profile.to_llm_context() == ""

    def test_biography_and_personality(self):
        profile = Profile()
        profile.biography = "Born in the northern wastes."
        profile.personality = "Stoic and reserved."
        context = profile.to_llm_context()
        assert "Biography: Born in the northern wastes." in context
        assert "Personality: Stoic and reserved." in context

    def test_physical_traits_in_context(self):
        profile = Profile()
        profile.physical.body = "stocky"
        profile.physical.face = "scarred"
        profile.physical.distinguishing_marks = "missing pinky"
        profile.physical.hair_fur = "grey mane"
        profile.physical.voice = "booming"
        context = profile.to_llm_context()
        assert "Body: stocky" in context
        assert "Face: scarred" in context
        assert "Distinguishing marks: missing pinky" in context
        assert "Hair/Fur: grey mane" in context
        assert "Voice: booming" in context

    def test_default_outfit_in_context(self):
        profile = Profile()
        profile.add_outfit("armor", "full plate with tabard", is_default=True)
        context = profile.to_llm_context()
        assert "Current outfit: full plate with tabard" in context

    def test_tendencies_in_context(self):
        profile = Profile()
        profile.tendencies = ["always smirks", "fidgets when nervous"]
        context = profile.to_llm_context()
        assert "Tendencies: always smirks, fidgets when nervous" in context

    def test_expressions_in_context(self):
        profile = Profile()
        profile.set_expression("angry", "bares teeth")
        context = profile.to_llm_context()
        assert "Expressions: angry: bares teeth" in context

    def test_notes_in_context(self):
        profile = Profile()
        profile.notes = "work in progress"
        context = profile.to_llm_context()
        assert "Notes: work in progress" in context


class TestProfileRoundtrip:
    def test_empty_roundtrip(self):
        original = Profile()
        restored = Profile.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_full_roundtrip(self):
        original = Profile()
        original.biography = "A wandering bard."
        original.personality = "Charismatic and reckless."
        original.physical = PhysicalTraits(
            body="wiry", face="sharp", distinguishing_marks="scar",
            hair_fur="red", voice="melodic",
        )
        original.add_outfit("travel", "cloak and boots", is_default=True)
        original.add_outfit("performance", "flashy vest")
        original.tendencies = ["hums constantly"]
        original.set_expression("happy", "wide grin, eyes sparkle")
        original.set_relationship("char_3", "best friend")
        original.notes = "Needs more backstory."

        restored = Profile.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()
        assert restored.biography == "A wandering bard."
        assert len(restored.outfits) == 2
        assert restored.default_outfit().name == "travel"
        assert restored.get_expression("happy") == "wide grin, eyes sparkle"
        assert restored.relationships["char_3"] == "best friend"
