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

    is_test:          False=generated (train), True=uploaded (validate)
    generation_round: which training round produced this latent.
                      Latents from round N were generated with LoRA_N-1.
                      Mixing rounds means mixing implicit distributions.
    """
    visual_latent: torch.Tensor
    language_latent: torch.Tensor
    accepted: bool
    prompt_used: str
    reverse_caption: str
    object_context: str
    match_score: float
    is_test: bool = False
    generation_round: int = 0
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
    val_mse: Optional[float] = None
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
        self.current_round: int = 0

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

        generation_round is set automatically — tracks which LoRA state
        was active when this latent was captured, so training can weight
        recent pairs higher (they come from the current implicit distribution).
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
            generation_round=self.current_round,
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

        pair_weights = self._recency_weights(train_data)

        for epoch in range(epochs):
            epoch_visual = 0.0
            epoch_language = 0.0
            epoch_review = 0.0

            # --- Training pass (generated images) ---
            for pair_idx, pair in enumerate(train_data):
                recency = pair_weights[pair_idx]
                vis_raw = pair.visual_latent.to(device).float()
                lang_raw = pair.language_latent.to(device).float()

                # Re-evaluate through current adapter state
                vis_adapted = self.adapter.visual_forward(
                    vis_raw, vis_raw
                )
                lang_adapted = self.adapter.language_forward(
                    lang_raw, lang_raw
                )

                # 1. Visual loss — MSE in rank-space (linear, magnitude matters)
                vis_compressed = self.adapter.A_visual(vis_adapted)
                lang_compressed = self.adapter.A_language(lang_adapted)
                interaction = vis_compressed @ self.adapter.interaction

                mse = torch.nn.functional.mse_loss(
                    interaction, lang_compressed,
                )

                margin = 2.0
                if pair.accepted:
                    visual_loss = mse
                else:
                    visual_loss = torch.clamp(margin - mse, min=0.0)

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

                    review_mse = torch.nn.functional.mse_loss(
                        self.adapter.A_visual(img_adapted),
                        self.adapter.A_language(ctx_adapted),
                    )

                    # Object context is ground truth — always minimize MSE
                    language_loss = review_mse
                else:
                    # Fallback: use match_score as static signal
                    if pair.match_score > 0.5:
                        language_loss = mse
                    else:
                        language_loss = torch.clamp(margin - mse, min=0.0)

                # 3. Review loss — alignment regularisation
                review_loss = self.adapter.alignment_loss()

                total = recency * (
                    self.visual_weight * visual_loss
                    + self.language_weight * language_loss
                    + self.review_weight * review_loss
                )

                epoch_visual += visual_loss.item() * recency
                epoch_language += language_loss.item() * recency
                epoch_review += review_loss.item() * recency

                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

            train_count = max(len(train_data), 1)
            _, S, _ = self.adapter.compute_interaction()

            # --- Validation pass (uploaded images — no gradients) ---
            val_mse = 0.0
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
                        mse_val = torch.nn.functional.mse_loss(
                            inter, lang_c,
                        )
                        val_mse += mse_val.item()
                self.adapter.train()

            result = TrainingResult(
                visual_loss=epoch_visual / train_count,
                language_loss=epoch_language / train_count,
                review_loss=epoch_review / train_count,
                combined_loss=(
                    epoch_visual + epoch_language + epoch_review
                ) / train_count,
                alignment=S.mean().item(),
                val_mse=val_mse / max(val_count, 1) if val_count else None,
                val_count=val_count,
            )
            results.append(result)

            if (epoch == 0 or epoch == epochs - 1
                    or (epoch + 1) % max(1, epochs // 10) == 0):
                val_str = (
                    f" val_mse={result.val_mse:.4f}({val_count})"
                    if result.val_mse is not None else ""
                )
                log.info(
                    "  Epoch %d/%d: vis=%.4f lang=%.4f review=%.4f"
                    " align=%.4f%s",
                    epoch + 1, epochs,
                    result.visual_loss, result.language_loss,
                    result.review_loss, result.alignment, val_str,
                )

        self.adapter.eval()
        self.current_round += 1
        return results

    def _recency_weights(
        self, pairs: list[TrainingPair], decay: float = 0.7,
    ) -> list[float]:
        """Exponential decay weights based on pair age.

        Pairs from the current LoRA round get weight 1.0. Older pairs
        decay exponentially because the implicit latent distribution
        shifted when LoRA weights were loaded.

        Args:
            pairs: Training pairs to weight.
            decay: Per-round decay factor. 0.7 means round N-1 pairs
                   get 70% weight, N-2 gets 49%, etc.

        Returns:
            List of weights, one per pair.
        """
        return [decay ** (self.current_round - p.generation_round) for p in pairs]

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
