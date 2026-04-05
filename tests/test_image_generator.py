"""Tests for image generation — prompt composition and generation flow.

All objects constructed through factories. Generation uses mocks.
"""
import pytest
from unittest.mock import MagicMock

from backend.models import ContentStore, Story, Character
from backend.generator.image_generator import ImageGenerator
from backend.generator.panel_generator import PanelGenerator
from tests.helpers import make_story, make_two_character_story, get_chain


@pytest.fixture
def content_store(tmp_path):
    return ContentStore(str(tmp_path / "content"))

@pytest.fixture
def story_with_scripts():
    story = make_two_character_story()
    chain = get_chain(story)
    chain["panel"].scripts["c1"].update(
        dialogue="Who are you?", action="grips staff",
        emotion="wary", direction="close-up", source="manual",
    )
    chain["panel"].scripts["c2"].update(
        dialogue="A friend.", action="raises hands",
        emotion="cautious", source="manual",
    )
    chain["panel"].update_narration("The wind carried whispers.")
    characters = [story.get_character("c1"), story.get_character("c2")]
    return story, characters, chain["panel"]


class TestPromptComposition:
    def test_prompt_has_no_hardcoded_style(self, story_with_scripts):
        _, characters, panel = story_with_scripts
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, characters)
        assert "comic book panel" not in prompt

    def test_prompt_includes_appearances(self, story_with_scripts):
        _, characters, panel = story_with_scripts
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, characters)
        assert "blue-haired girl" in prompt
        assert "wolf anthro" in prompt

    def test_prompt_includes_page_context(self, story_with_scripts):
        """Page setting/mood/action appear in prompt, not chapter synopsis."""
        story, characters, panel = story_with_scripts
        # Set page context
        context = panel.get_context()
        # The direct composer uses page context not chapter synopsis
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, characters)
        # Chapter synopsis is for the LLM, not direct prompt
        assert "they meet" not in prompt

    def test_prompt_includes_actions(self, story_with_scripts):
        _, characters, panel = story_with_scripts
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, characters)
        assert "grips_staff" in prompt
        assert "holding_hands" in prompt

    def test_prompt_excludes_narration(self, story_with_scripts):
        """Narration is text overlay, not image content — should not be in prompt."""
        _, characters, panel = story_with_scripts
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, characters)
        assert "whispers" not in prompt

    def test_prompt_includes_emotions(self, story_with_scripts):
        _, characters, panel = story_with_scripts
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, characters)
        assert "wary" in prompt
        assert "cautious" in prompt

    def test_prompt_excludes_dialogue(self, story_with_scripts):
        _, characters, panel = story_with_scripts
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, characters)
        assert "Who are you?" not in prompt

    def test_empty_scripts_produce_empty_prompt(self):
        """Characters with empty scripts are excluded — no appearance leak."""
        story = make_story()
        chain = get_chain(story)
        characters = [story.get_character("c1")]
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(chain["panel"], characters)
        # Empty script means character is not in the scene
        assert "blue-haired girl" not in prompt

    def test_active_scripts_include_appearance(self):
        """Characters with actual script content get their appearance in prompt."""
        story = make_story()
        chain = get_chain(story)
        chain["script"].update(action="waves", source="manual")
        characters = [story.get_character("c1")]
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(chain["panel"], characters)
        assert "blue-haired girl" in prompt

    def test_empty_script_characters_excluded_from_prompt(self):
        """Characters with blank scripts must NOT appear in generation prompts.

        The cascade creates scripts for all chapter characters in every panel,
        but only characters with actual content (action/emotion/direction)
        should be included in the prompt. Empty scripts are placeholders.
        """
        story = Story("s1", "Test Story")
        story.add_character(Character("c1", "Luna", appearance_prompt="blue-haired girl"))
        story.add_character(Character("c2", "Rex", appearance_prompt="tall wolf anthro"))
        story.add_character(Character("c3", "Nyx", appearance_prompt="black cat girl"))
        story.create_chapter("The Meeting", ["c1", "c2", "c3"], synopsis="they meet")

        chain = get_chain(story)
        panel = chain["panel"]

        # Only give c1 and c2 actual script content; c3 stays empty
        panel.scripts["c1"].update(action="grips staff", emotion="wary", source="manual")
        panel.scripts["c2"].update(action="raises hands", emotion="cautious", source="manual")
        # c3 has an empty cascade script — should NOT appear in prompt

        # Simulate what the API does: collect characters from panel.scripts.keys()
        all_characters = [
            story.get_character(character_id)
            for character_id in panel.scripts.keys()
        ]
        assert len(all_characters) == 3  # cascade created all 3

        # The prompt composer should filter out empty-script characters
        gen = PanelGenerator(MagicMock())
        prompt = gen.compose_prompt_direct(panel, all_characters)

        assert "blue-haired girl" in prompt  # c1 has content
        assert "wolf anthro" in prompt       # c2 has content
        assert "black cat" not in prompt     # c3 is empty — must be excluded
        assert "Nyx" not in prompt           # c3's name must not appear either
        assert "3 characters" not in prompt  # count should reflect actual, not all

    def test_negative_prompt(self):
        gen = PanelGenerator(MagicMock())
        negative = gen.compose_negative_prompt()
        assert "worst_quality" in negative
        assert "bad_anatomy" in negative
        assert "jpeg_artefacts" in negative

    def test_script_prompt_excludes_dialogue(self):
        story = make_story()
        chain = get_chain(story)
        chain["script"].update(dialogue="Hello", action="waves", emotion="happy", source="manual")
        prompt = chain["script"].to_prompt()
        assert "Hello" not in prompt
        assert "waves" in prompt


class TestLatentCapture:
    """Latent capture must not call encode_prompt directly — breaks CPU offload."""

    def test_run_inference_does_not_call_encode_prompt(self):
        """_run_inference must capture latents via callback, not manual encode_prompt."""
        import inspect
        source = inspect.getsource(ImageGenerator._run_inference)
        assert "encode_prompt" not in source, (
            "_run_inference must not call encode_prompt directly — "
            "it breaks with sequential_cpu_offload. "
            "Use callback_on_step_end to capture latents instead."
        )

    def test_generator_device_is_cpu(self):
        """torch.Generator must use 'cpu' — sequential_cpu_offload runs latents on CPU."""
        import inspect
        for method in (ImageGenerator._run_inference, ImageGenerator._run_inpaint):
            source = inspect.getsource(method)
            assert 'Generator(device="cpu")' in source or "Generator(device='cpu')" in source, (
                f"{method.__name__}: torch.Generator must be created on 'cpu', not self.device — "
                "sequential_cpu_offload runs prepare_latents on CPU"
            )

    def test_generator_has_latent_attributes(self, content_store):
        gen = ImageGenerator(content_store, device="cpu")
        assert hasattr(gen, '_last_visual_latent')
        assert hasattr(gen, '_last_language_latent')
        assert gen._last_visual_latent is None
        assert gen._last_language_latent is None


class TestGenerationFlow:
    def _make_mock_generator(self, content_store):
        from PIL import Image
        fake = Image.new("RGB", (768, 512), color=(100, 50, 200))
        gen = ImageGenerator(content_store, device="cpu")
        gen._loaded = True
        gen._run_inference = MagicMock(return_value=fake)
        return gen

    @pytest.mark.asyncio
    async def test_generate_stores_in_content_store(self, content_store):
        gen = self._make_mock_generator(content_store)
        content_hash = await gen.generate(prompt="test", width=768, height=512)
        assert content_store.exists(content_hash)
        assert content_store.retrieve(content_hash)[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_generate_emits_events(self, content_store):
        gen = self._make_mock_generator(content_store)
        events = []
        gen.on("generation_started", lambda d: events.append("started"))
        gen.on("generation_complete", lambda d: events.append("complete"))
        await gen.generate(prompt="test")
        assert "started" in events
        assert "complete" in events

    @pytest.mark.asyncio
    async def test_panel_generator_updates_hash(self, content_store):
        story = make_story()
        chain = get_chain(story)
        chain["script"].update(action="waves", source="manual")
        image_gen = self._make_mock_generator(content_store)
        panel_gen = PanelGenerator(image_gen)
        content_hash = await panel_gen.generate_panel_image(
            chain["panel"], [story.get_character("c1")]
        )
        assert chain["panel"].image_hash == content_hash
        assert chain["panel"].source == "ai"

    @pytest.mark.asyncio
    async def test_generation_propagates_up(self, content_store):
        story = make_story()
        chain = get_chain(story)
        chain["script"].update(action="waves", source="manual")
        image_gen = self._make_mock_generator(content_store)
        received = []
        story.on("story_updated", lambda d: received.append(True))
        panel_gen = PanelGenerator(image_gen)
        await panel_gen.generate_panel_image(chain["panel"], [story.get_character("c1")])
        assert len(received) > 0

    @pytest.mark.asyncio
    async def test_deterministic_seed(self, content_store):
        gen = self._make_mock_generator(content_store)
        h1 = await gen.generate(prompt="test", seed=42)
        h2 = await gen.generate(prompt="test", seed=42)
        assert h1 == h2
