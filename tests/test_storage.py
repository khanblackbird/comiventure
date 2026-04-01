"""Tests for story save/load — .cvn file round-tripping.

Uses realistically-sized noise data. All objects constructed through factories.
"""
import os
import zipfile
import json
import pytest

from backend.models import Story, Character, ContentStore
from backend.models.storage import save_story, load_story
from tests.helpers import get_chain


def _generate_png_noise(width=1024, height=768):
    return b"\x89PNG\r\n\x1a\n" + os.urandom(width * height * 3)

def _generate_video_noise(frames=60):
    return b"\x00\x00\x00\x1cftypisom" + os.urandom(frames * 256 * 256 * 3)


@pytest.fixture
def source_store(tmp_path):
    return ContentStore(str(tmp_path / "source"))

@pytest.fixture
def dest_store(tmp_path):
    return ContentStore(str(tmp_path / "dest"))

@pytest.fixture
def image_data():
    return _generate_png_noise()

@pytest.fixture
def video_data():
    return _generate_video_noise()

@pytest.fixture
def story_with_assets(source_store, image_data, video_data):
    story = Story("s1", "The Adventure", synopsis="Two heroes meet")
    story.add_character(Character("luna", "Luna", appearance_prompt="blue-haired girl"))
    story.add_character(Character("rex", "Rex", appearance_prompt="tall wolf anthro"))
    story.create_chapter("The Meeting", ["luna", "rex"], synopsis="they meet at the crossroads")

    chain = get_chain(story)

    # Add image to panel
    image_hash = source_store.store(image_data, "image/png", {"width": 1024, "height": 768})
    chain["panel"].update_image(image_hash, source="ai")

    # Update scripts
    chain["panel"].scripts["luna"].update(
        dialogue="We need to leave NOW!", action="rushes forward",
        emotion="panicked", direction="close-up", source="ai",
    )
    chain["panel"].scripts["rex"].update(
        dialogue="You always say that.", action="leans on wall",
        emotion="amused", source="manual",
    )

    # Add second panel with video
    page = chain["page"]
    panel_two = page.create_panel(character_ids=["luna"])
    story.register_panel(panel_two)
    for s in panel_two.scripts.values():
        story.register_script(s)

    video_hash = source_store.store(video_data, "video/mp4")
    panel_two.update_video(video_hash, source="ai")
    panel_two.update_narration("The wind howled.")

    second_image = _generate_png_noise(512, 512)
    second_hash = source_store.store(second_image, "image/png")
    panel_two.update_image(second_hash, source="manual")

    panel_two.scripts["luna"].update(action="stares at horizon", emotion="melancholy", source="ai")

    return story


class TestSaveLoad:
    def test_save_creates_file(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        assert path.endswith(".cvn")

    def test_cvn_is_valid_zip(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        assert zipfile.is_zipfile(path)

    def test_cvn_contains_story_json(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        with zipfile.ZipFile(path) as z:
            assert "story.json" in z.namelist()
            data = json.loads(z.read("story.json"))
            assert data["title"] == "The Adventure"

    def test_cvn_contains_assets(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        with zipfile.ZipFile(path) as z:
            content = [n for n in z.namelist() if n.startswith("content/")]
            assert len(content) >= 2


class TestByteExactRoundTrip:
    def test_image_survives(self, story_with_assets, source_store, dest_store, image_data, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, dest_store)
        # Find panel with image
        for ch in loaded.chapters.values():
            for pg in ch.pages:
                for pan in pg.panels:
                    if pan.image_hash and not pan.video_hash:
                        retrieved = dest_store.retrieve(pan.image_hash)
                        assert retrieved == image_data
                        return
        pytest.fail("No panel with image found")

    def test_video_survives(self, story_with_assets, source_store, dest_store, video_data, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, dest_store)
        for ch in loaded.chapters.values():
            for pg in ch.pages:
                for pan in pg.panels:
                    if pan.video_hash:
                        retrieved = dest_store.retrieve(pan.video_hash)
                        assert retrieved == video_data
                        return
        pytest.fail("No panel with video found")


class TestHierarchyAfterLoad:
    def test_preserves_characters(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, source_store)
        assert loaded.get_character("luna") is not None
        assert loaded.get_character("rex") is not None

    def test_preserves_relationships(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, source_store)
        chapter = list(loaded.chapters.values())[0]
        assert "luna" in chapter.character_ids
        assert "rex" in chapter.character_ids

    def test_preserves_scripts(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, source_store)
        # Find luna's script with dialogue
        for ch in loaded.chapters.values():
            for pg in ch.pages:
                for pan in pg.panels:
                    s = pan.get_script("luna")
                    if s and s.dialogue == "We need to leave NOW!":
                        assert s.emotion == "panicked"
                        return
        pytest.fail("Luna's script not found")

    def test_emission_works(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, source_store)
        received = []
        loaded.on("story_updated", lambda d: received.append(True))
        chain = get_chain(loaded)
        chain["script"].update(dialogue="changed", source="manual")
        assert len(received) > 0

    def test_registry_works(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, source_store)
        chain = get_chain(loaded)
        assert loaded.lookup(chain["page"].page_id) is chain["page"]
        assert loaded.lookup(chain["panel"].panel_id) is chain["panel"]
        assert loaded.lookup(chain["script"].script_id) is chain["script"]

    def test_validates_after_load(self, story_with_assets, source_store, tmp_path):
        path = save_story(story_with_assets, source_store, str(tmp_path / "test"))
        loaded = load_story(path, source_store)
        assert loaded.validate() == []


class TestEdgeCases:
    def test_empty_story_rejected(self, source_store, tmp_path):
        story = Story("empty", "Empty")
        with pytest.raises(RuntimeError, match="integrity violation"):
            save_story(story, source_store, str(tmp_path / "empty"))

    def test_minimal_story_saves(self, source_store, tmp_path):
        story = Story("s1", "Minimal")
        story.add_character(Character("c1", "Solo"))
        story.create_chapter("Ch1", ["c1"])
        path = save_story(story, source_store, str(tmp_path / "min"))
        loaded = load_story(path, source_store)
        assert loaded.title == "Minimal"

    def test_extension_added(self, source_store, tmp_path):
        story = Story("s1", "T")
        story.add_character(Character("c1", "A"))
        story.create_chapter("C", ["c1"])
        path = save_story(story, source_store, str(tmp_path / "noext"))
        assert path.endswith(".cvn")
