"""Tests for IP-Adapter bridge — uses character reference images to
condition generation for visual consistency.

TDD: tests written first, implementation follows.
"""
import pytest
from io import BytesIO
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.models import ContentStore, Story, Character
from backend.generator.ip_adapter_bridge import IPAdapterBridge
from tests.helpers import make_two_character_story, get_chain


def _make_test_image_bytes():
    """Create minimal PNG bytes for testing."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def content_store(tmp_path):
    return ContentStore(str(tmp_path / "content"))


@pytest.fixture
def story_with_refs(content_store):
    """Story with two characters, one has accepted references."""
    story = make_two_character_story()
    chain = get_chain(story)

    # Give c1 some script content so it's active
    chain["panel"].scripts["c1"].update(action="waves", source="manual")
    chain["panel"].scripts["c2"].update(action="nods", source="manual")

    # Add accepted reference images for c1
    char_1 = story.get_character("c1")
    img_bytes = _make_test_image_bytes()
    hash_1 = content_store.store(img_bytes, "image/png")
    hash_2 = content_store.store(img_bytes + b"\x00", "image/png")  # different hash
    char_1.appearance.add_reference(hash_1, source="upload")
    char_1.appearance.add_reference(hash_2, source="upload")
    char_1.appearance.accept_reference(hash_1)
    char_1.appearance.accept_reference(hash_2)

    # c2 has no references
    return story, chain["panel"], content_store


@pytest.fixture
def bridge(content_store):
    return IPAdapterBridge(content_store)


class TestReferenceCollection:
    def test_collects_accepted_references(self, story_with_refs, bridge):
        """Accepted references are collected for active characters."""
        story, panel, _ = story_with_refs
        characters = [story.get_character("c1"), story.get_character("c2")]
        images = bridge.collect_reference_images(characters, panel)
        # c1 has 2 accepted refs, c2 has none
        assert len(images) == 2
        for img in images:
            assert isinstance(img, Image.Image)

    def test_skips_rejected_references(self, story_with_refs, bridge):
        """Rejected references must not appear in conditioning images."""
        story, panel, content_store = story_with_refs
        char_1 = story.get_character("c1")

        # Add a rejected reference
        rejected_bytes = _make_test_image_bytes()
        rejected_hash = content_store.store(rejected_bytes + b"\x01\x02", "image/png")
        char_1.appearance.add_reference(rejected_hash, source="upload")
        char_1.appearance.reject_reference(rejected_hash)

        characters = [char_1]
        images = bridge.collect_reference_images(characters, panel)
        # Should still be 2 (the 2 accepted), not 3
        assert len(images) == 2

    def test_skips_unrated_references(self, story_with_refs, bridge):
        """Unrated references (accepted=None) must not be included."""
        story, panel, content_store = story_with_refs
        char_1 = story.get_character("c1")

        # Add an unrated reference
        unrated_bytes = _make_test_image_bytes()
        unrated_hash = content_store.store(unrated_bytes + b"\x03\x04", "image/png")
        char_1.appearance.add_reference(unrated_hash, source="upload")
        # Don't accept or reject — stays None

        characters = [char_1]
        images = bridge.collect_reference_images(characters, panel)
        assert len(images) == 2  # still only the 2 accepted

    def test_skips_characters_without_refs(self, story_with_refs, bridge):
        """Characters with no references produce no images."""
        story, panel, _ = story_with_refs
        # c2 has no references
        characters = [story.get_character("c2")]
        images = bridge.collect_reference_images(characters, panel)
        assert len(images) == 0

    def test_returns_empty_for_no_characters(self, bridge):
        """Empty character list returns empty image list."""
        panel = MagicMock()
        images = bridge.collect_reference_images([], panel)
        assert images == []

    def test_skips_characters_with_empty_scripts(self, story_with_refs, bridge):
        """Characters with empty scripts (not active) are skipped."""
        story, panel, content_store = story_with_refs

        # Add a third character with refs but empty script
        char_3 = Character("c3", "Ghost", appearance_prompt="translucent figure")
        story.add_character(char_3)
        chapter = list(story.chapters.values())[0]
        chapter.bind_character("c3")

        # Give c3 references
        img_bytes = _make_test_image_bytes()
        ghost_hash = content_store.store(img_bytes + b"\x05", "image/png")
        char_3.appearance.add_reference(ghost_hash, source="upload")
        char_3.appearance.accept_reference(ghost_hash)

        # c3 has a cascade-created empty script — should be excluded
        all_characters = [
            story.get_character(cid)
            for cid in panel.scripts.keys()
        ]
        images = bridge.collect_reference_images(all_characters, panel)
        # Only c1's 2 refs (c2 has no refs, c3 has empty script)
        assert len(images) == 2


class TestPipelineLoading:
    def test_ensure_loaded_calls_load_ip_adapter(self, bridge):
        """Pipeline's load_ip_adapter is called on first use."""
        pipeline = MagicMock()
        bridge.ensure_loaded(pipeline)
        pipeline.load_ip_adapter.assert_called_once()

    def test_ensure_loaded_only_once(self, bridge):
        """Repeated calls to ensure_loaded don't reload."""
        pipeline = MagicMock()
        bridge.ensure_loaded(pipeline)
        bridge.ensure_loaded(pipeline)
        bridge.ensure_loaded(pipeline)
        assert pipeline.load_ip_adapter.call_count == 1

    def test_sets_ip_adapter_scale(self, bridge):
        """IP adapter scale is set after loading."""
        pipeline = MagicMock()
        bridge.ensure_loaded(pipeline)
        pipeline.set_ip_adapter_scale.assert_called_once()


class TestGenerationKwargs:
    def test_prepare_returns_ip_adapter_image(self, story_with_refs, bridge):
        """prepare_generation_kwargs returns ip_adapter_image key."""
        story, panel, _ = story_with_refs
        characters = [story.get_character("c1")]
        pipeline = MagicMock()

        kwargs = bridge.prepare_generation_kwargs(characters, panel, pipeline)
        assert "ip_adapter_image" in kwargs
        assert len(kwargs["ip_adapter_image"]) == 2

    def test_prepare_returns_empty_dict_when_no_refs(self, bridge):
        """No references means no ip_adapter_image key — don't pollute kwargs."""
        story = make_two_character_story()
        chain = get_chain(story)
        chain["panel"].scripts["c1"].update(action="waves", source="manual")
        characters = [story.get_character("c1")]
        pipeline = MagicMock()

        kwargs = bridge.prepare_generation_kwargs(characters, chain["panel"], pipeline)
        assert kwargs == {}

    def test_prepare_calls_ensure_loaded(self, story_with_refs, bridge):
        """Preparing kwargs triggers IP adapter loading if needed."""
        story, panel, _ = story_with_refs
        characters = [story.get_character("c1")]
        pipeline = MagicMock()

        bridge.prepare_generation_kwargs(characters, panel, pipeline)
        pipeline.load_ip_adapter.assert_called_once()


class TestLoadFailure:
    def test_ensure_loaded_handles_missing_model(self, bridge):
        """If IP-Adapter model isn't downloaded, ensure_loaded returns False
        instead of crashing. Generation must work without it."""
        pipeline = MagicMock()
        pipeline.load_ip_adapter.side_effect = OSError(
            "h94/IP-Adapter does not appear to have a file named ip-adapter_sdxl.bin"
        )
        result = bridge.ensure_loaded(pipeline)
        assert result is False
        assert bridge._load_failed is True

    def test_load_failure_does_not_retry(self, bridge):
        """After one failure, ensure_loaded should not retry."""
        pipeline = MagicMock()
        pipeline.load_ip_adapter.side_effect = OSError("not found")
        bridge.ensure_loaded(pipeline)
        bridge.ensure_loaded(pipeline)
        # Only tried once despite two calls
        assert pipeline.load_ip_adapter.call_count == 1

    def test_prepare_returns_empty_when_load_fails(self, story_with_refs, bridge):
        """If IP-Adapter can't load, prepare_generation_kwargs returns {}
        so generation proceeds without conditioning."""
        story, panel, _ = story_with_refs
        characters = [story.get_character("c1")]
        pipeline = MagicMock()
        pipeline.load_ip_adapter.side_effect = OSError("not found")

        kwargs = bridge.prepare_generation_kwargs(characters, panel, pipeline)
        assert kwargs == {}

    def test_generation_not_blocked_by_missing_ip_adapter(self, story_with_refs, bridge):
        """The full prepare flow must never raise — it returns {} on failure."""
        story, panel, _ = story_with_refs
        characters = [story.get_character("c1")]
        pipeline = MagicMock()
        pipeline.load_ip_adapter.side_effect = Exception("anything")

        # Must not raise
        kwargs = bridge.prepare_generation_kwargs(characters, panel, pipeline)
        assert isinstance(kwargs, dict)
