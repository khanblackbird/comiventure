"""Adversarial adapter — two LoRA adapters connected directly through
their low-rank representations, no image/text conversion overhead.

UNet adapter (visual) ←→ Text encoder adapter (language)

The interaction happens in the compressed rank-space:
  A₁, B₁ = UNet adapter matrices
  A₂, B₂ = Text encoder adapter matrices
  Interaction = B₂·A₂·B₁·A₁ → SVD → U·Σ·V*

Σ (diagonal) = interaction strengths between visual and language adapters.
Low Σ values = misalignment (prompt says one thing, image does another).
Training pushes Σ toward alignment.

No pixels, no text, no disk — latents flow directly between adapters.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Optional

log = logging.getLogger(__name__)


class AdversarialAdapter(nn.Module):
    """Two LoRA adapters that train against each other through
    their shared compressed space.

    visual_rank and language_rank can differ but the interaction
    matrix bridges them.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 4,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim

        # Visual adapter (UNet side)
        self.A_visual = nn.Linear(hidden_dim, rank, bias=False)
        self.B_visual = nn.Linear(rank, hidden_dim, bias=False)

        # Language adapter (text encoder side)
        self.A_language = nn.Linear(hidden_dim, rank, bias=False)
        self.B_language = nn.Linear(rank, hidden_dim, bias=False)

        # Sigmoid gate — learned blend between frozen base and adapter
        self.gate_visual = nn.Parameter(torch.tensor(gate_init))
        self.gate_language = nn.Parameter(torch.tensor(gate_init))

        # Interaction: the direct connection between adapters
        # Instead of going through pixels/text, the rank-space
        # representations interact through this matrix
        self.interaction = nn.Parameter(torch.eye(rank))

        self._init_weights()

    def _init_weights(self):
        """Small init so adapters start near-zero (base model dominates)."""
        nn.init.kaiming_uniform_(self.A_visual.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B_visual.weight)
        nn.init.kaiming_uniform_(self.A_language.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B_language.weight)

    def visual_forward(self, x: torch.Tensor, frozen_output: torch.Tensor) -> torch.Tensor:
        """Apply visual adapter to UNet layer output."""
        adapter_out = self.B_visual(self.A_visual(x))
        gate = torch.sigmoid(self.gate_visual)
        return frozen_output + gate * adapter_out

    def language_forward(self, x: torch.Tensor, frozen_output: torch.Tensor) -> torch.Tensor:
        """Apply language adapter to text encoder layer output."""
        adapter_out = self.B_language(self.A_language(x))
        gate = torch.sigmoid(self.gate_language)
        return frozen_output + gate * adapter_out

    def compute_interaction(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the interaction matrix and decompose it.

        Returns (U, Σ, V) from SVD of the full interaction chain:
          B_language · A_language · interaction · B_visual · A_visual

        Σ (diagonal) contains the interaction strengths.
        Low values = misalignment between visual and language adapters.
        """
        # Chain the adapter matrices through the interaction
        # This is rank × rank, not hidden_dim × hidden_dim
        chain = (
            self.A_language.weight  # [rank, hidden]
            @ self.B_visual.weight  # [hidden, rank]
        )  # [rank, rank]

        # Apply the learned interaction
        M = chain * self.interaction  # element-wise with learned interaction

        # SVD decomposition — diagonal tells us alignment
        U, S, Vh = torch.linalg.svd(M)
        return U, S, Vh

    def alignment_loss(self) -> torch.Tensor:
        """Loss that pushes the adapters toward alignment.

        Maximises the singular values of the interaction matrix —
        meaning the visual and language adapters agree on what
        the compressed representations mean.
        """
        _, S, _ = self.compute_interaction()
        # Negative sum of singular values — maximise alignment
        return -S.sum()

    def misalignment_directions(self) -> torch.Tensor:
        """Returns the directions in rank-space where the adapters
        disagree most. These are the training targets.
        """
        U, S, Vh = self.compute_interaction()
        # Directions with smallest singular values = most misaligned
        min_idx = S.argmin()
        return Vh[min_idx]

    def training_step(
        self,
        visual_latent: torch.Tensor,
        language_latent: torch.Tensor,
        accepted: bool,
    ) -> torch.Tensor:
        """One training step using a visual/language latent pair.

        visual_latent: output from the UNet's attention layer
        language_latent: output from the text encoder
        accepted: user feedback — True = these should align, False = shouldn't

        The loss pushes accepted pairs toward alignment and
        rejected pairs toward misalignment in the compressed space.
        """
        # Project both into rank-space — cast to float32 for training
        v_compressed = self.A_visual(visual_latent.detach().float())
        l_compressed = self.A_language(language_latent.detach().float())

        # Interaction through the learned matrix
        v_through = v_compressed @ self.interaction  # [batch, rank]

        # Alignment = cosine similarity in rank-space
        similarity = nn.functional.cosine_similarity(v_through, l_compressed, dim=-1)

        if accepted:
            # Push toward alignment — similarity should be high
            loss = 1.0 - similarity.mean()
        else:
            # Push toward misalignment — similarity should be low
            loss = similarity.mean()

        # Add the global alignment regulariser
        loss = loss + 0.1 * self.alignment_loss()

        return loss

    def save_weights(self) -> bytes:
        """Serialize just the adapter weights — small, portable."""
        from io import BytesIO
        buffer = BytesIO()
        torch.save({
            'A_visual': self.A_visual.state_dict(),
            'B_visual': self.B_visual.state_dict(),
            'A_language': self.A_language.state_dict(),
            'B_language': self.B_language.state_dict(),
            'gate_visual': self.gate_visual.data,
            'gate_language': self.gate_language.data,
            'interaction': self.interaction.data,
            'rank': self.rank,
            'hidden_dim': self.hidden_dim,
        }, buffer)
        return buffer.getvalue()

    @classmethod
    def load_weights(cls, data: bytes) -> AdversarialAdapter:
        """Restore from serialized weights."""
        from io import BytesIO
        state = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
        adapter = cls(
            hidden_dim=state['hidden_dim'],
            rank=state['rank'],
        )
        adapter.A_visual.load_state_dict(state['A_visual'])
        adapter.B_visual.load_state_dict(state['B_visual'])
        adapter.A_language.load_state_dict(state['A_language'])
        adapter.B_language.load_state_dict(state['B_language'])
        adapter.gate_visual.data = state['gate_visual']
        adapter.gate_language.data = state['gate_language']
        adapter.interaction.data = state['interaction']
        return adapter


class AdversarialTrainer:
    """Trains the adversarial adapter using collected feedback pairs.

    Each pair is: (visual_latent, language_latent, accepted)
    Extracted from the pipeline during generation — no extra inference.
    """

    def __init__(
        self,
        adapter: AdversarialAdapter,
        learning_rate: float = 1e-4,
    ) -> None:
        self.adapter = adapter
        self.optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
        self.pairs: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def add_pair(
        self,
        visual_latent: torch.Tensor,
        language_latent: torch.Tensor,
        accepted: bool,
    ) -> None:
        """Store a training pair from a generation + feedback."""
        self.pairs.append((
            visual_latent.detach().cpu(),
            language_latent.detach().cpu(),
            accepted,
        ))

    def train(self, epochs: int = 5) -> float:
        """Train on accumulated pairs. Returns average loss."""
        if not self.pairs:
            return 0.0

        self.adapter.train()
        device = next(self.adapter.parameters()).device

        total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for v_latent, lang_latent, accepted in self.pairs:
                v = v_latent.to(device).float()
                lang = lang_latent.to(device).float()

                loss = self.adapter.training_step(v, lang, accepted)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg = epoch_loss / len(self.pairs)
            total_loss += avg
            log.debug("  Adversarial epoch %d/%d, loss: %.4f", epoch + 1, epochs, avg)

        self.adapter.eval()
        return total_loss / epochs

    def pair_count(self) -> int:
        return len(self.pairs)
