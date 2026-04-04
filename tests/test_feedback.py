"""Tests for feedback deduplication — one vote per image."""
import pytest
from unittest.mock import MagicMock
from backend.generator.adapter import StoryAdapter, FeedbackEntry


@pytest.fixture
def adapter():
    store = MagicMock()
    return StoryAdapter("test-story", store)


class TestOneVotePerImage:
    """Each image should have exactly one vote, not accumulate."""

    def test_single_vote_recorded(self, adapter):
        adapter.add_feedback("hash1", "prompt", True)
        assert len(adapter.feedback) == 1
        assert adapter.feedback[0].accepted is True

    def test_revote_replaces(self, adapter):
        adapter.add_feedback("hash1", "prompt", True)
        adapter.add_feedback("hash1", "prompt", False)
        assert len(adapter.feedback) == 1
        assert adapter.feedback[0].accepted is False

    def test_revote_back_to_positive(self, adapter):
        adapter.add_feedback("hash1", "prompt", True)
        adapter.add_feedback("hash1", "prompt", False)
        adapter.add_feedback("hash1", "prompt", True)
        assert len(adapter.feedback) == 1
        assert adapter.feedback[0].accepted is True

    def test_different_images_independent(self, adapter):
        adapter.add_feedback("hash1", "prompt1", True)
        adapter.add_feedback("hash2", "prompt2", False)
        assert len(adapter.feedback) == 2
        assert len(adapter.positive_samples()) == 1
        assert len(adapter.negative_samples()) == 1

    def test_revote_preserves_other_images(self, adapter):
        adapter.add_feedback("hash1", "prompt1", True)
        adapter.add_feedback("hash2", "prompt2", True)
        adapter.add_feedback("hash3", "prompt3", False)
        # Revote hash2
        adapter.add_feedback("hash2", "prompt2", False)
        assert len(adapter.feedback) == 3
        assert len(adapter.positive_samples()) == 1
        assert len(adapter.negative_samples()) == 2

    def test_counts_reflect_unique_images(self, adapter):
        """Voting 5 times on the same image should count as 1, not 5."""
        for _ in range(5):
            adapter.add_feedback("hash1", "prompt", True)
        assert len(adapter.positive_samples()) == 1
        assert len(adapter.negative_samples()) == 0
