"""Latent reviewer — captures embeddings directly from ollama,
no text conversion between models.

Image → LLaVA embedding (4096)
Prompt → Llama embedding (4096)
Object context → Llama embedding (4096)

All three go into the interaction matrix as latents.
The text captions are only for the user's benefit.
"""
from __future__ import annotations

import os
import base64
import logging
import torch
import httpx
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class LatentReview:
    """Embeddings captured from the review loop."""
    image_embedding: torch.Tensor       # LLaVA's embedding of the image
    prompt_embedding: torch.Tensor      # Llama's embedding of the prompt
    context_embedding: torch.Tensor     # Llama's embedding of object graph
    caption_text: str                   # human-readable caption (for UI)
    match_score: float                  # text-based score (for UI)


class LatentReviewer:
    """Captures embeddings directly from ollama models."""

    def __init__(
        self,
        ollama_host: str = None,
        vision_model: str = "llava:7b",
        text_model: str = "llama3:8b",
    ) -> None:
        if ollama_host is None:
            ollama_host = os.environ.get(
                "OLLAMA_HOST", "http://ollama:11434"
            )
        self.ollama_host = ollama_host
        self.vision_model = vision_model
        self.text_model = text_model

    async def review(
        self,
        image_bytes: bytes,
        prompt: str,
        object_context: str,
    ) -> Optional[LatentReview]:
        """Full latent review — three embeddings + text for UI."""
        async with httpx.AsyncClient() as client:
            # Get all three embeddings
            image_emb = await self._image_embedding(client, image_bytes)
            prompt_emb = await self._text_embedding(client, prompt)
            context_emb = await self._text_embedding(client, object_context)

            if image_emb is None or prompt_emb is None:
                return None

            if context_emb is None:
                context_emb = torch.zeros_like(prompt_emb)

            # Also get text caption for the user (not for training)
            caption = await self._caption_image(client, image_bytes)

            # Compute text-based score for UI display
            cos_sim = torch.nn.functional.cosine_similarity(
                prompt_emb.unsqueeze(0),
                image_emb.unsqueeze(0),
            ).item()
            match_score = (cos_sim + 1.0) / 2.0  # map [-1,1] to [0,1]

            return LatentReview(
                image_embedding=image_emb,
                prompt_embedding=prompt_emb,
                context_embedding=context_emb,
                caption_text=caption,
                match_score=match_score,
            )

    async def _image_embedding(
        self, client: httpx.AsyncClient, image_bytes: bytes
    ) -> Optional[torch.Tensor]:
        """Get LLaVA's embedding of an image."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            response = await client.post(
                f"{self.ollama_host}/api/embeddings",
                json={
                    "model": self.vision_model,
                    "prompt": "describe this image",
                    "images": [image_b64],
                },
                timeout=60.0,
            )
            if response.status_code == 200:
                emb = response.json().get("embedding", [])
                if emb:
                    return torch.tensor(emb, dtype=torch.float32)
        except Exception as e:
            log.warning("Image embedding failed: %s", e)
        return None

    async def _text_embedding(
        self, client: httpx.AsyncClient, text: str
    ) -> Optional[torch.Tensor]:
        """Get Llama's embedding of text."""
        if not text:
            return None
        try:
            response = await client.post(
                f"{self.ollama_host}/api/embeddings",
                json={
                    "model": self.text_model,
                    "prompt": text,
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                emb = response.json().get("embedding", [])
                if emb:
                    return torch.tensor(emb, dtype=torch.float32)
        except Exception as e:
            log.warning("Text embedding failed: %s", e)
        return None

    async def _caption_image(
        self, client: httpx.AsyncClient, image_bytes: bytes
    ) -> str:
        """Get text caption for the user (not for training)."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": (
                        "Describe this image briefly for an AI "
                        "image generator. Be specific and concise."
                    ),
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=60.0,
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            log.warning("Caption failed: %s", e)
        return ""
