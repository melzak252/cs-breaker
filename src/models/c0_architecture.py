"""PyTorch C0 model architecture adapted for Counter-Strike 2.

Implements:
1. CS2PlayerHistoryAttention:
   - Projects player map tokens (map ID, side ID, continuous stats) to hidden_dim.
   - Computes attention weights over history sequence using masked softmax
     safely handling all-False empty history without NaN.
   - Outputs attended player representation (B, 2, 5, hidden_dim).
2. CS2TeamSetTransformer:
   - Set attention across 5 players using nn.TransformerEncoder (1 layer, 4 heads).
   - Permutation-invariant mean + std pooling: (B, d_model * 2).
3. CS2C0Model:
   - End-to-end model with shared attention and team set transformer.
   - Pairwise anti-symmetric scoring head:
     score_ab = head([team_a, team_b])
     score_ba = head([team_b, team_a])
     logit = 0.5 * (score_ab - score_ba)
     prob = sigmoid(logit)
   - Guaranteed P(A beats B) = 1 - P(B beats A) to machine precision.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.c0_tokens import CS2TokenConfig, CS2TokenEmbedding

__all__ = [
    "CS2PlayerHistoryAttention",
    "CS2TeamSetTransformer",
    "CS2C0Model",
]


class CS2PlayerHistoryAttention(nn.Module):
    """Player history attention pooling over map performance sequence.

    Embeds discrete map ID and side ID alongside continuous statistics into
    a hidden representation of dimension `hidden_dim`, then aggregates the 16
    map positions via masked attention weights. Empty history rows (all-False
    masks) are safely handled to return zero vectors without producing NaNs.
    """

    def __init__(
        self,
        token_dim: int | CS2TokenConfig = 18,
        hidden_dim: int = 32,
        config: CS2TokenConfig | None = None,
        token_config: CS2TokenConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(token_dim, CS2TokenConfig):
            resolved_config = token_dim
        else:
            resolved_config = token_config or config

        if resolved_config is not None:
            self.config = resolved_config
        else:
            self.config = CS2TokenConfig(
                hidden_dim=hidden_dim,
                player_stat_dim=max(0, token_dim - 2),
            )
        self.hidden_dim = self.config.hidden_dim
        self.token_dim = self.config.total_token_dim
        self.token_embedding = CS2TokenEmbedding(self.config)
        self.attn_score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        history: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            history: Tensor of shape (..., S, token_dim) containing map tokens.
                     Can be (B, 2, 5, 16, token_dim) or (B, 5, 16, token_dim).
            mask: Optional boolean tensor of shape (..., S) where True indicates
                  valid observed history and False indicates padding.
            return_weights: Whether to also return the attention weights.

        Returns:
            attended: Tensor of shape (..., hidden_dim).
            attn_weights (optional): Tensor of shape (..., S).
        """
        # Clean corrupted NaNs or infinities in raw history tensor
        clean_hist = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)

        # Discrete IDs clamped to valid embedding ranges
        m_ids = torch.clamp(clean_hist[..., 0].long(), 0, self.config.num_maps - 1)
        s_ids = torch.clamp(clean_hist[..., 1].long(), 0, self.config.num_sides - 1)
        st = clean_hist[..., 2:].float()

        # Project tokens to hidden_dim: (..., S, hidden_dim)
        embedded = self.token_embedding(map_ids=m_ids, side_ids=s_ids, stats=st)

        # Unnormalized attention scores: (..., S)
        scores = self.attn_score(embedded).squeeze(-1)

        if mask is not None:
            mask_bool = mask.bool()
            # Identify sequences with at least one valid history position
            has_valid = mask_bool.any(dim=-1, keepdim=True)

            # To avoid NaN from softmax on all -inf rows when a player has no history,
            # substitute an all-ones mask for the empty row during softmax calculation
            safe_mask = torch.where(has_valid, mask_bool, torch.ones_like(mask_bool))
            scores_masked = scores.masked_fill(~safe_mask, -1e9)
            attn_weights = torch.softmax(scores_masked, dim=-1)

            # Strictly zero out attention weights and embeddings for empty players
            attn_weights = torch.where(has_valid, attn_weights, torch.zeros_like(attn_weights))
            embedded = embedded * mask_bool.unsqueeze(-1).to(embedded.dtype)
        else:
            attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum over sequence dimension: (..., hidden_dim)
        attended = torch.sum(attn_weights.unsqueeze(-1) * embedded, dim=-2)

        if return_weights:
            return attended, attn_weights
        return attended


class CS2TeamSetTransformer(nn.Module):
    """Set Transformer encoder for 5-player Counter-Strike teams.

    Applies multi-head self-attention without positional encodings across the 5
    players to model team synergies and interactions in a permutation-equivariant
    manner. Permutation invariance is then guaranteed by pooling across the player
    set using concatenated mean and standard deviation.
    """

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (..., 5, d_model) representing player embeddings.
               Supports batched (B, 5, d_model), unbatched (5, d_model), or
               multi-dimensional batches (*batch, 5, d_model).

        Returns:
            team_vector: Tensor of shape (..., d_model * 2).
        """
        orig_shape = x.shape[:-2]
        # Flatten leading batch dimensions to 2D for TransformerEncoder
        x_flat = x.view(-1, 5, self.d_model)

        # Set attention (permutation equivariant across the 5 players)
        enc_out = self.encoder(x_flat)

        # Permutation-invariant pooling: mean and standard deviation
        mean_pool = enc_out.mean(dim=1)
        std_pool = enc_out.std(dim=1, unbiased=False)
        team_vec_flat = torch.cat([mean_pool, std_pool], dim=-1)

        if len(orig_shape) == 0:
            return team_vec_flat.squeeze(0)
        return team_vec_flat.view(*orig_shape, self.d_model * 2)


class CS2C0Model(nn.Module):
    """Complete Counter-Strike 2 C0 neural model.

    Processes both teams using shared player history attention and team set
    transformers. Employs a pairwise anti-symmetric scoring head guaranteeing:
        logit(A, B) = -logit(B, A)
        P(A beats B) = 1 - P(B beats A)
    to machine precision.
    """

    def __init__(
        self,
        token_dim: int | CS2TokenConfig = 18,
        hidden_dim: int = 32,
        nhead: int = 4,
        transformer_layers: int = 1,
        head_hidden_dim: int = 64,
        token_config: CS2TokenConfig | None = None,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if isinstance(token_dim, CS2TokenConfig):
            token_config = token_dim
            token_dim = token_config.total_token_dim
            hidden_dim = token_config.hidden_dim
        elif token_config is not None:
            token_dim = token_config.total_token_dim
            hidden_dim = token_config.hidden_dim

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.token_config = token_config
        self.team_dim = hidden_dim * 2

        self.history_attention = CS2PlayerHistoryAttention(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            config=token_config,
        )
        self.team_transformer = CS2TeamSetTransformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=transformer_layers,
            dim_feedforward=hidden_dim * 2,
        )

        if head is not None:
            self.head = head
        else:
            self.head = nn.Sequential(
                nn.Linear(self.team_dim * 2, head_hidden_dim),
                nn.GELU(),
                nn.Linear(head_hidden_dim, 1),
            )

    def forward(
        self,
        history: torch.Tensor | dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
        return_logit: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            history: Either a dict containing 'history' and optional 'mask', or
                     a tensor of shape (B, 2, 5, 16, token_dim) or (2, 5, 16, token_dim).
            mask: Optional boolean tensor of shape (B, 2, 5, 16) or (2, 5, 16).
            return_logit: If True, returns tuple (prob, logit).

        Returns:
            prob: Win probability of team 1 over team 2, shape (B, 1) in [0, 1].
            logit (optional): Logit difference 0.5 * (score_ab - score_ba), shape (B, 1).
        """
        if isinstance(history, dict):
            mask = history.get("mask", mask)
            history = history["history"]

        unbatched = False
        if history.dim() == 4:
            history = history.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            unbatched = True

        # Attend player histories: (B, 2, 5, hidden_dim)
        player_reps = self.history_attention(history, mask)

        # Team set representations: (B, hidden_dim * 2)
        team_a = self.team_transformer(player_reps[:, 0])
        team_b = self.team_transformer(player_reps[:, 1])

        # Pairwise anti-symmetric scoring head
        pair_ab = torch.cat([team_a, team_b], dim=-1)
        pair_ba = torch.cat([team_b, team_a], dim=-1)

        score_ab = self.head(pair_ab)
        score_ba = self.head(pair_ba)

        # Anti-symmetric logit: logit(B, A) = -logit(A, B)
        logit = 0.5 * (score_ab - score_ba)

        # Double precision intermediate guarantees exact machine-precision antisymmetry:
        # P(A beats B) + P(B beats A) == 1.0
        prob = torch.sigmoid(logit.double()).to(dtype=logit.dtype)

        if unbatched:
            prob = prob.squeeze(0)
            logit = logit.squeeze(0)

        if return_logit:
            return prob, logit
        return prob

    def encode_team(
        self,
        team_history: torch.Tensor,
        team_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a 5-player team history tensor into a team representation.

        Args:
            team_history: Tensor of shape (..., 5, 16, token_dim).
            team_mask: Optional boolean tensor of shape (..., 5, 16).

        Returns:
            Tensor of shape (..., hidden_dim * 2).
        """
        player_reps = self.history_attention(team_history, team_mask)
        return self.team_transformer(player_reps)

    def compute_logits(
        self,
        history: torch.Tensor | dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the anti-symmetric logit difference without sigmoid."""
        _, logit = self.forward(history, mask=mask, return_logit=True, **kwargs)
        return logit

    def predict_proba(
        self,
        history: torch.Tensor | dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict win probability for team 1 against team 2."""
        return self.forward(history, mask=mask, return_logit=False, **kwargs)
