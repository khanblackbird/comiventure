"""Tests for the full adversarial loop.

Prompt → Image → Caption (reverse) → Compare → Gap → Training signal

Each step must produce output that the next step can consume.
The loop must close — the review result must feed back into training.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from backend.app import app
from backend.api import routes
from backend.models import Story, Character, ContentStore
from backend.generator.image_reviewer import ImageReviewer, ReviewResult


@pytest.fixture(autouse=True)
def fresh_state(tmp_path):
    routes.story = Story("test", "Test")
    routes.content_store = ContentStore(str(tmp_path / "content"))
    routes.image_generator = None
    yield
    routes.story = None


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def story_with_image(tmp_path):
    """A story with a panel that has a generated image."""
    story = Story("test", "Test")
    content_store = ContentStore(str(tmp_path / "content"))

    story.add_character(Character("c1", "Peter",
                                  appearance_prompt="brown rabbit"))
    story.create_chapter("Ch1", ["c1"])

    # Store a fake image
    from PIL import Image
    from io import BytesIO
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_hash = content_store.store(
        buf.getvalue(), "image/png",
        metadata={"prompt": "brown rabbit in a garden"},
    )

    # Set the panel's image
    chapter = list(story.chapters.values())[0]
    panel = chapter.pages[0].panels[0]
    panel.update_image(image_hash, source="ai")

    routes.story = story
    routes.content_store = content_store

    return story, panel, image_hash


class TestImageReviewerUnit:
    """Test the reviewer in isolation with mocked ollama."""

    @pytest.mark.asyncio
    async def test_caption_returns_string(self):
        reviewer = ImageReviewer(ollama_host="http://fake:11434")
        with patch("httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=instance
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            # Mock tags check
            tags_response = MagicMock()
            tags_response.status_code = 200
            tags_response.json.return_value = {
                "models": [{"name": "llava:7b"}]
            }

            # Mock generation
            gen_response = MagicMock()
            gen_response.status_code = 200
            gen_response.json.return_value = {
                "response": "A brown rabbit sitting in a garden"
            }

            instance.get = AsyncMock(return_value=tags_response)
            instance.post = AsyncMock(return_value=gen_response)

            caption = await reviewer.caption_image(b"fake image bytes")
            assert isinstance(caption, str)
            assert len(caption) > 0
            assert "rabbit" in caption.lower()

    @pytest.mark.asyncio
    async def test_compare_returns_review_result(self):
        reviewer = ImageReviewer(ollama_host="http://fake:11434")
        with patch("httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=instance
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "response": (
                    "SCORE: 0.7\n"
                    "DIFFERENCES: wrong colour, missing jacket\n"
                    "SUGGESTION: specify blue jacket explicitly"
                )
            }
            instance.post = AsyncMock(return_value=response)

            result = await reviewer.compare_prompts(
                "brown rabbit in blue jacket",
                "brown rabbit in a garden",
            )

            assert isinstance(result, ReviewResult)
            assert 0.0 <= result.match_score <= 1.0
            assert result.match_score == 0.7
            assert len(result.differences) > 0
            assert "wrong colour" in result.differences
            assert len(result.suggestion) > 0

    @pytest.mark.asyncio
    async def test_full_review_returns_result(self):
        reviewer = ImageReviewer(ollama_host="http://fake:11434")
        with patch("httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=instance
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            # Tags check
            tags_response = MagicMock()
            tags_response.status_code = 200
            tags_response.json.return_value = {
                "models": [{"name": "llava:7b"}]
            }

            # Caption response
            caption_response = MagicMock()
            caption_response.status_code = 200
            caption_response.json.return_value = {
                "response": "A rabbit in a field"
            }

            # Compare response
            compare_response = MagicMock()
            compare_response.status_code = 200
            compare_response.json.return_value = {
                "response": (
                    "SCORE: 0.6\n"
                    "DIFFERENCES: field instead of garden\n"
                    "SUGGESTION: use garden not field"
                )
            }

            call_count = [0]

            async def mock_get(*args, **kwargs):
                return tags_response

            async def mock_post(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return caption_response
                return compare_response

            instance.get = mock_get
            instance.post = mock_post

            result = await reviewer.review(
                b"fake image", "rabbit in a garden"
            )

            assert result.match_score == 0.6
            assert result.reverse_caption == "A rabbit in a field"
            assert "field instead of garden" in result.differences

    @pytest.mark.asyncio
    async def test_missing_vision_model_returns_empty(self):
        reviewer = ImageReviewer(ollama_host="http://fake:11434")
        with patch("httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=instance
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            tags_response = MagicMock()
            tags_response.status_code = 200
            tags_response.json.return_value = {
                "models": [{"name": "llama3:8b"}]
            }
            instance.get = AsyncMock(return_value=tags_response)

            caption = await reviewer.caption_image(b"fake image")
            assert caption == ""


class TestReviewAPIEndpoint:
    """Test the review endpoint via the API."""

    def test_review_requires_panel_image(self, client, fresh_state):
        story = Story("test", "Test")
        story.add_character(Character("c1", "Peter"))
        story.create_chapter("Ch1", ["c1"])
        routes.story = story

        chapter = list(story.chapters.values())[0]
        panel_id = chapter.pages[0].panels[0].panel_id

        response = client.post(f"/api/review/{panel_id}")
        assert response.status_code == 400
        assert "no image" in response.json()["detail"].lower()

    def test_review_returns_result_structure(
        self, client, story_with_image
    ):
        story, panel, image_hash = story_with_image

        with patch(
            "backend.generator.image_reviewer.ImageReviewer.review"
        ) as mock_review:
            mock_review.return_value = ReviewResult(
                original_prompt="brown rabbit in a garden",
                reverse_caption="A rabbit sitting outdoors",
                match_score=0.75,
                differences=["outdoors vs garden"],
                suggestion="specify garden setting",
            )

            response = client.post(f"/api/review/{panel.panel_id}")
            assert response.status_code == 200

            data = response.json()
            assert "match_score" in data
            assert "reverse_caption" in data
            assert "differences" in data
            assert "suggestion" in data
            assert data["match_score"] == 0.75

    def test_review_nonexistent_panel_404(self, client):
        response = client.post("/api/review/fake-panel")
        assert response.status_code == 404


class TestUnifiedTrainer:
    """The unified trainer uses all three signals."""

    def test_trains_with_full_pairs(self):
        from backend.generator.adversarial_adapter import AdversarialAdapter
        from backend.generator.unified_trainer import (
            UnifiedTrainer, TrainingPair,
        )
        import torch

        adapter = AdversarialAdapter(hidden_dim=16, rank=2)
        trainer = UnifiedTrainer(adapter)

        for i in range(5):
            trainer.add_pair(TrainingPair(
                visual_latent=torch.randn(1, 16),
                language_latent=torch.randn(1, 16),
                accepted=(i % 2 == 0),
                prompt_used="a brown rabbit",
                reverse_caption="a rabbit in a field",
                object_context="Peter Rabbit, brown fur, blue jacket",
                match_score=0.6 + (i * 0.05),
            ))

        results = trainer.train(epochs=2)
        assert len(results) == 2
        assert results[0].visual_loss >= 0
        assert results[0].language_loss >= 0
        assert results[0].alignment > 0

    def test_trains_without_review_data(self):
        from backend.generator.adversarial_adapter import AdversarialAdapter
        from backend.generator.unified_trainer import UnifiedTrainer
        import torch

        adapter = AdversarialAdapter(hidden_dim=16, rank=2)
        trainer = UnifiedTrainer(adapter)

        for i in range(3):
            trainer.add_from_generation(
                visual_latent=torch.randn(1, 16),
                language_latent=torch.randn(1, 16),
                accepted=True,
            )

        results = trainer.train(epochs=1)
        assert len(results) == 1
        assert trainer.reviewed_pair_count() == 0

    def test_reviewed_pairs_counted(self):
        from backend.generator.adversarial_adapter import AdversarialAdapter
        from backend.generator.unified_trainer import UnifiedTrainer
        import torch

        adapter = AdversarialAdapter(hidden_dim=16, rank=2)
        trainer = UnifiedTrainer(adapter)

        trainer.add_from_generation(
            torch.randn(1, 16), torch.randn(1, 16),
            True, "prompt", "caption", "context", 0.8,
        )
        trainer.add_from_generation(
            torch.randn(1, 16), torch.randn(1, 16),
            False,
        )

        assert trainer.pair_count() == 2
        assert trainer.reviewed_pair_count() == 1


class TestReviewFeedsTraining:
    """The review result must be usable as training signal."""

    def test_low_score_means_mismatch(self):
        result = ReviewResult(
            original_prompt="blue rabbit",
            reverse_caption="red fox",
            match_score=0.1,
            differences=["wrong species", "wrong colour"],
            suggestion="be more specific about rabbit",
        )
        assert result.match_score < 0.5
        assert len(result.differences) >= 2

    def test_high_score_means_match(self):
        result = ReviewResult(
            original_prompt="brown rabbit in garden",
            reverse_caption="brown rabbit sitting in vegetable garden",
            match_score=0.9,
            differences=[],
            suggestion="",
        )
        assert result.match_score > 0.7
        assert len(result.differences) == 0

    def test_parse_comparison_handles_bad_format(self):
        reviewer = ImageReviewer()
        result = reviewer._parse_comparison(
            "prompt", "caption", "garbage output with no format"
        )
        assert result.match_score == 0.5
        assert result.differences == []

    def test_parse_comparison_extracts_score(self):
        reviewer = ImageReviewer()
        result = reviewer._parse_comparison(
            "prompt", "caption",
            "SCORE: 0.85\nDIFFERENCES: none\nSUGGESTION: looks good",
        )
        assert result.match_score == 0.85
        assert result.suggestion == "looks good"
