"""Tests for the ContentStore — content-addressable storage for heavy data."""

import shutil
import pytest
from pathlib import Path

from backend.models.content_store import ContentStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary content store for each test."""
    return ContentStore(str(tmp_path / "content"))


class TestStore:
    def test_store_returns_hash(self, store):
        content_hash = store.store(b"test image data", "image/png")
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64  # SHA-256 hex

    def test_retrieve_returns_original_bytes(self, store):
        original = b"\x89PNG fake image data"
        content_hash = store.store(original, "image/png")
        retrieved = store.retrieve(content_hash)
        assert retrieved == original

    def test_retrieve_nonexistent_returns_none(self, store):
        assert store.retrieve("nonexistent_hash") is None

    def test_deduplication(self, store):
        data = b"same content"
        hash_first = store.store(data, "image/png")
        hash_second = store.store(data, "image/png")
        assert hash_first == hash_second

        # Only one file on disk
        files = list(Path(store.storage_dir).glob("*"))
        assert len(files) == 1

    def test_different_content_different_hashes(self, store):
        hash_a = store.store(b"image A", "image/png")
        hash_b = store.store(b"image B", "image/png")
        assert hash_a != hash_b

    def test_get_path_without_loading_bytes(self, store):
        content_hash = store.store(b"data", "image/png")
        path = store.get_path(content_hash)
        assert path is not None
        assert Path(path).exists()

    def test_get_path_nonexistent_returns_none(self, store):
        assert store.get_path("nonexistent") is None

    def test_exists(self, store):
        content_hash = store.store(b"data", "image/png")
        assert store.exists(content_hash)
        assert not store.exists("nonexistent")

    def test_delete(self, store):
        content_hash = store.store(b"data", "image/png")
        path = store.get_path(content_hash)
        store.delete(content_hash)
        assert not store.exists(content_hash)
        assert not Path(path).exists()

    def test_metadata_stored(self, store):
        content_hash = store.store(
            b"data", "image/png",
            metadata={"width": 1024, "height": 768},
        )
        meta = store.get_meta(content_hash)
        assert meta.metadata["width"] == 1024
        assert meta.size_bytes == 4
        assert meta.content_type == "image/png"

    def test_correct_file_extension(self, store):
        hash_png = store.store(b"png", "image/png")
        hash_mp4 = store.store(b"mp4", "video/mp4")
        assert store.get_path(hash_png).endswith(".png")
        assert store.get_path(hash_mp4).endswith(".mp4")
