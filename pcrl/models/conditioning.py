"""FiLM, concat, and attention conditioning modules.

Three interchangeable ways to inject purpose embeddings into hidden features.
All modules share the same interface: forward(h, purpose_emb) -> h_conditioned.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning.

    Applies an affine transformation conditioned on purpose embedding:
        h_out = gamma(e_p) * h + beta(e_p)

    Initialized to identity (gamma=1, beta=0) so the encoder starts
    unconditioned and gradually learns purpose-specific modulations.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(conditioning_dim, feature_dim)
        self.beta_proj = nn.Linear(conditioning_dim, feature_dim)

        # Identity initialization
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, purpose_emb: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            h: Hidden features (batch_size, feature_dim).
            purpose_emb: Purpose embedding (batch_size, conditioning_dim).

        Returns:
            Modulated features (batch_size, feature_dim).
        """
        gamma = self.gamma_proj(purpose_emb)
        beta = self.beta_proj(purpose_emb)
        return gamma * h + beta


class ConcatConditioner(nn.Module):
    """Concatenation-based conditioning.

    Concatenates the purpose embedding with hidden features, then projects
    back to the original feature dimensionality.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim + conditioning_dim, feature_dim)

    def forward(self, h: torch.Tensor, purpose_emb: torch.Tensor) -> torch.Tensor:
        """Concatenate purpose embedding and project back.

        Args:
            h: Hidden features (batch_size, feature_dim).
            purpose_emb: Purpose embedding (batch_size, conditioning_dim).

        Returns:
            Conditioned features (batch_size, feature_dim).
        """
        combined = torch.cat([h, purpose_emb], dim=-1)
        return self.proj(combined)


class AttentionConditioner(nn.Module):
    """Single-head cross-attention conditioning.

    Uses purpose embedding as query and hidden features as keys/values.
    Features are treated as a sequence of length 1 (broadcast) so the
    attention reduces to a soft gating.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(feature_dim)

        # Project purpose embedding to query space
        self.q_proj = nn.Linear(conditioning_dim, feature_dim)
        # Keys and values from features
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, h: torch.Tensor, purpose_emb: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention conditioning.

        Args:
            h: Hidden features (batch_size, feature_dim).
            purpose_emb: Purpose embedding (batch_size, conditioning_dim).

        Returns:
            Attended features (batch_size, feature_dim).
        """
        q = self.q_proj(purpose_emb)  # (B, D)
        k = self.k_proj(h)  # (B, D)
        v = self.v_proj(h)  # (B, D)

        # Compute attention score (scalar per sample)
        attn = (q * k).sum(dim=-1, keepdim=True) / self.scale  # (B, 1)
        attn = torch.sigmoid(attn)  # soft gate

        # Apply gated value + residual connection
        out = attn * v
        out = self.out_proj(out)
        return h + out


def build_conditioner(
    conditioning_type: str,
    feature_dim: int,
    conditioning_dim: int,
) -> nn.Module:
    """Factory function for conditioning modules.

    Args:
        conditioning_type: One of "film", "concat", "attention".
        feature_dim: Dimension of hidden features.
        conditioning_dim: Dimension of purpose embedding.

    Returns:
        Conditioning module.
    """
    if conditioning_type == "film":
        return FiLMConditioner(feature_dim, conditioning_dim)
    elif conditioning_type == "concat":
        return ConcatConditioner(feature_dim, conditioning_dim)
    elif conditioning_type == "attention":
        return AttentionConditioner(feature_dim, conditioning_dim)
    else:
        raise ValueError(
            f"Unknown conditioning type '{conditioning_type}'. "
            f"Must be 'film', 'concat', or 'attention'."
        )
