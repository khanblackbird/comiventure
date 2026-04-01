from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


class ContentStore:
    """Content-addressable store for heavy data (images, video, audio).

    Objects in the hierarchy only hold content hashes — lightweight
    strings that travel through emission without cost. The actual
    bytes live here, addressed by SHA-256.

    This means:
    - Emission never carries pixel data, only hash references
    - Identical content is stored once (deduplication)
    - Lookups are O(1) by hash
    - The store is flat — no hierarchy, no traversal
    """

    def __init__(self, storage_dir: str = "data/content") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index: hash -> metadata
        self._index: dict[str, ContentMeta] = {}

    def store(self, data: bytes, content_type: str, metadata: Optional[dict] = None) -> str:
        """Store content bytes, return the content hash (the reference).
        This hash is what objects hold — never the bytes themselves.
        """
        content_hash = hashlib.sha256(data).hexdigest()

        if content_hash not in self._index:
            extension = _extension_for_type(content_type)
            file_path = self.storage_dir / f"{content_hash}{extension}"
            file_path.write_bytes(data)

            self._index[content_hash] = ContentMeta(
                content_hash=content_hash,
                content_type=content_type,
                file_path=str(file_path),
                size_bytes=len(data),
                metadata=metadata or {},
            )

        return content_hash

    def retrieve(self, content_hash: str) -> Optional[bytes]:
        """Load content bytes by hash. Only call when you actually
        need the pixels — not during propagation.
        """
        meta = self._index.get(content_hash)
        if not meta:
            return None
        file_path = Path(meta.file_path)
        if not file_path.exists():
            return None
        return file_path.read_bytes()

    def get_path(self, content_hash: str) -> Optional[str]:
        """Get the file path for a content hash. For serving to
        frontend without loading bytes into memory.
        """
        meta = self._index.get(content_hash)
        return meta.file_path if meta else None

    def get_meta(self, content_hash: str) -> Optional[ContentMeta]:
        """Get metadata for content without loading bytes."""
        return self._index.get(content_hash)

    def exists(self, content_hash: str) -> bool:
        return content_hash in self._index

    def delete(self, content_hash: str) -> None:
        """Remove content from store."""
        meta = self._index.pop(content_hash, None)
        if meta:
            file_path = Path(meta.file_path)
            if file_path.exists():
                file_path.unlink()


class ContentMeta:
    """Lightweight metadata about stored content.
    This is what gets indexed — never the bytes.
    """

    __slots__ = ("content_hash", "content_type", "file_path", "size_bytes", "metadata")

    def __init__(
        self,
        content_hash: str,
        content_type: str,
        file_path: str,
        size_bytes: int,
        metadata: dict,
    ) -> None:
        self.content_hash = content_hash
        self.content_type = content_type
        self.file_path = file_path
        self.size_bytes = size_bytes
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "content_hash": self.content_hash,
            "content_type": self.content_type,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


def _extension_for_type(content_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "audio/wav": ".wav",
        "text/plain": ".txt",
    }.get(content_type, ".bin")
