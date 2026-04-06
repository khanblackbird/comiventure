"""Unified trainer — closes all loops through the object graph.

The object graph is the ground truth. Everything else approximates it.

Three training signals, one pass:
1. Visual: captured image latent vs accepted reference latents
2. Language: prompt embedding vs object graph description embedding
3. Review: reverse caption vs object graph text

All flow through the adversarial adapter's interaction matrix.
All use captured latents — no extra inference.

The training step:
  - Takes: visual_latent, language_latent, review_caption, object_context
  - Computes: three loss terms
  - Updates: both adapter sides through the interaction matrix
  - Returns: combined loss + per-component losses for diagnostics
"""
from __future__ import annotations

import logging
import torch
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

from backend.generator.adversarial_adapter import (
    AdversarialAdapter,
    AdversarialTrainer,
)


@dataclass
class TrainingPair:
    """One sample from a generation or upload cycle.

    is_test=False: AI-generated image → train on this
    is_test=True:  user-uploaded image → validate against this (ground truth)
    """
    visual_latent: torch.Tensor
    language_latent: torch.Tensor
    accepted: bool
    prompt_used: str
    reverse_caption: str
    object_context: str
    match_score: float
    is_test: bool = False
    image_embedding: Optional[torch.Tensor] = None
    prompt_embedding: Optional[torch.Tensor] = None
    context_embedding: Optional[torch.Tensor] = None


@dataclass
class TrainingResult:
    """Diagnostic output from one training epoch."""
    visual_loss: float
    language_loss: float
    review_loss: float
    combined_loss: float
    alignment: float
    # Validation metrics (from test set — uploaded images)
    val_similarity: Optional[float] = None
    val_count: int = 0


class UnifiedTrainer:
    """Trains all adapter components through a single pass.

    The object graph context is the anchor — the training pushes
    all components toward producing output that matches the
    character descriptions, scene settings, and relationships
    defined by the user.
    """

    def __init__(
        self,
        adapter: AdversarialAdapter,
        learning_rate: float = 1e-4,
        visual_weight: float = 1.0,
        language_weight: float = 1.0,
        review_weight: float = 0.5,
    ) -> None:
        self.adapter = adapter
        self.optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=learning_rate
        )
        self.pairs: list[TrainingPair] = []
        self.visual_weight = visual_weight
        self.language_weight = language_weight
        self.review_weight = review_weight

    def add_pair(self, pair: TrainingPair) -> None:
        self.pairs.append(pair)

    def add_from_generation(
        self,
        visual_latent: torch.Tensor,
        language_latent: torch.Tensor,
        accepted: bool,
        prompt_used: str = "",
        reverse_caption: str = "",
        object_context: str = "",
        match_score: float = 0.5,
        is_test: bool = False,
    ) -> None:
        """Add a pair from a generation or upload cycle.

        is_test=False: AI-generated → used for training
        is_test=True:  user-uploaded → used for validation only
        """
        self.pairs.append(TrainingPair(
            visual_latent=visual_latent.detach().cpu(),
            language_latent=language_latent.detach().cpu(),
            accepted=accepted,
            prompt_used=prompt_used,
            reverse_caption=reverse_caption,
            object_context=object_context,
            match_score=match_score,
            is_test=is_test,
        ))

    @property
    def train_pairs(self) -> list[TrainingPair]:
        """AI-generated images — used for training."""
        return [p for p in self.pairs if not p.is_test]

    @property
    def test_pairs(self) -> list[TrainingPair]:
        """User-uploaded images — used for validation only."""
        return [p for p in self.pairs if p.is_test]

    def train(self, epochs: int = 5) -> list[TrainingResult]:
        """Train on generated images, validate against uploaded images.

        Training pairs (is_test=False): gradient updates
        Test pairs (is_test=True): validation loss only, no gradients

        Each epoch:
        1. Train: forward + backward on generated image pairs
        2. Validate: forward only on uploaded image pairs (no grad)
        3. Report both metrics
        """
        train_data = self.train_pairs
        test_data = self.test_pairs
        if not train_data:
            return []

        self.adapter.train()
        device = next(self.adapter.parameters()).device
        results = []

        for epoch in range(epochs):
            epoch_visual = 0.0
            epoch_language = 0.0
            epoch_review = 0.0

            # --- Training pass (generated images) ---
            for pair in train_data:
                vis_raw = pair.visual_latent.to(device).float()
                lang_raw = pair.language_latent.to(device).float()

                # Re-evaluate through current adapter state
                vis_adapted = self.adapter.visual_forward(
                    vis_raw, vis_raw
                )
                lang_adapted = self.adapter.language_forward(
                    lang_raw, lang_raw
                )

                # 1. Visual loss — adapted representations alignment
                vis_compressed = self.adapter.A_visual(vis_adapted)
                lang_compressed = self.adapter.A_language(lang_adapted)
                interaction = vis_compressed @ self.adapter.interaction

                similarity = torch.nn.functional.cosine_similarity(
                    interaction, lang_compressed, dim=-1
                )

                if pair.accepted:
                    visual_loss = (1.0 - similarity).mean()
                else:
                    visual_loss = similarity.mean()

                # 2. Language loss — if we have ollama embeddings,
                # use them directly. The LLaVA image embedding
                # should align with the Llama context embedding
                # through the same interaction matrix.
                if (pair.image_embedding is not None
                        and pair.context_embedding is not None):
                    img_emb = pair.image_embedding.to(device).float()
                    ctx_emb = pair.context_embedding.to(device).float()

                    # Project to adapter rank space
                    # Embeddings are 4096-dim, adapter is hidden_dim
                    # Truncate or pad to match
                    adim = self.adapter.hidden_dim
                    img_proj = img_emb[..., :adim]
                    ctx_proj = ctx_emb[..., :adim]

                    img_adapted = self.adapter.visual_forward(
                        img_proj, img_proj
                    )
                    ctx_adapted = self.adapter.language_forward(
                        ctx_proj, ctx_proj
                    )

                    review_sim = torch.nn.functional.cosine_similarity(
                        self.adapter.A_visual(img_adapted),
                        self.adapter.A_language(ctx_adapted),
                        dim=-1,
                    )

                    # Object context is ground truth — always push
                    # toward alignment
                    language_loss = (1.0 - review_sim).mean()
                else:
                    # Fallback: use match_score as static signal
                    review_accepted = pair.match_score > 0.5
                    if review_accepted:
                        language_loss = (1.0 - similarity).mean()
                    else:
                        language_loss = similarity.mean()

                # 3. Review loss — alignment regularisation
                review_loss = self.adapter.alignment_loss()

                total = (
                    self.visual_weight * visual_loss
                    + self.language_weight * language_loss
                    + self.review_weight * review_loss
                )

                epoch_visual += visual_loss.item()
                epoch_language += language_loss.item()
                epoch_review += review_loss.item()

                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

            train_count = max(len(train_data), 1)
            _, S, _ = self.adapter.compute_interaction()

            # --- Validation pass (uploaded images — no gradients) ---
            val_sim = 0.0
            val_count = len(test_data)
            if test_data:
                self.adapter.eval()
                with torch.no_grad():
                    for pair in test_data:
                        vis_raw = pair.visual_latent.to(device).float()
                        lang_raw = pair.language_latent.to(device).float()
                        vis_adapted = self.adapter.visual_forward(
                            vis_raw, vis_raw,
                        )
                        lang_adapted = self.adapter.language_forward(
                            lang_raw, lang_raw,
                        )
                        vis_c = self.adapter.A_visual(vis_adapted)
                        lang_c = self.adapter.A_language(lang_adapted)
                        inter = vis_c @ self.adapter.interaction
                        sim = torch.nn.functional.cosine_similarity(
                            inter, lang_c, dim=-1,
                        )
                        val_sim += sim.mean().item()
                self.adapter.train()

            result = TrainingResult(
                visual_loss=epoch_visual / train_count,
                language_loss=epoch_language / train_count,
                review_loss=epoch_review / train_count,
                combined_loss=(
                    epoch_visual + epoch_language + epoch_review
                ) / train_count,
                alignment=S.mean().item(),
                val_similarity=val_sim / max(val_count, 1) if val_count else None,
                val_count=val_count,
            )
            results.append(result)

            if (epoch == 0 or epoch == epochs - 1
                    or (epoch + 1) % max(1, epochs // 10) == 0):
                val_str = (
                    f" val={result.val_similarity:.4f}({val_count})"
                    if result.val_similarity is not None else ""
                )
                log.info(
                    "  Epoch %d/%d: vis=%.4f lang=%.4f review=%.4f"
                    " align=%.4f%s",
                    epoch + 1, epochs,
                    result.visual_loss, result.language_loss,
                    result.review_loss, result.alignment, val_str,
                )

        self.adapter.eval()
        return results

    def pair_count(self) -> int:
        return len(self.pairs)

    def train_pair_count(self) -> int:
        return len(self.train_pairs)

    def test_pair_count(self) -> int:
        return len(self.test_pairs)

    def reviewed_pair_count(self) -> int:
        """Pairs that have review data (not just accept/reject)."""
        return sum(
            1 for p in self.pairs
            if p.reverse_caption and p.object_context
        )
