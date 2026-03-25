"""Encoders for PCRL and baselines."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from pcrl.models.conditioning import build_conditioner


class StandardEncoder(nn.Module):
    """Standard MLP encoder without purpose conditioning.

    Matches PurposeConditionedEncoder's architecture (Linear → BatchNorm →
    ReLU → Dropout per hidden layer, final Linear projection) but ignores
    purpose_idx entirely. Used as a baseline to isolate the effect of
    purpose conditioning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        repr_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.repr_dim = repr_dim

        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.repr_proj = nn.Linear(hidden_dims[-1], repr_dim)

    def forward(
        self,
        x: torch.Tensor,
        purpose_idx: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Encode input features (purpose_idx is accepted but ignored)."""
        h = self.network(x)
        return self.repr_proj(h)


class PurposeConditionedEncoder(nn.Module):
    """Purpose-conditioned encoder network.

    Learns representations that are conditioned on the specified purpose,
    enabling purpose-specific information filtering.

    Architecture per hidden layer:
        Linear -> Conditioning(purpose_emb) -> BatchNorm -> ReLU -> Dropout

    Conditioning is injected BETWEEN each hidden layer (not just at input).
    The same input x with different purpose_idx will produce different
    representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        repr_dim: int,
        num_purposes: int,
        purpose_emb_dim: int,
        conditioning: Literal["film", "concat", "attention"] = "film",
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder.

        Args:
            input_dim: Dimensionality of input features.
            hidden_dims: List of hidden layer dimensions.
            repr_dim: Dimensionality of output representation.
            num_purposes: Number of distinct purposes.
            purpose_emb_dim: Dimensionality of purpose embeddings.
            conditioning: Conditioning method ("film", "concat", or "attention").
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.repr_dim = repr_dim
        self.num_purposes = num_purposes
        self.purpose_emb_dim = purpose_emb_dim
        self.conditioning_type = conditioning

        # Learnable purpose embeddings
        self.purpose_embedding = nn.Embedding(
            num_embeddings=num_purposes,
            embedding_dim=purpose_emb_dim,
        )

        # Build encoder layers
        self.linear_layers = nn.ModuleList()
        self.conditioning_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            self.linear_layers.append(nn.Linear(in_dim, out_dim))
            self.conditioning_layers.append(
                build_conditioner(conditioning, out_dim, purpose_emb_dim)
            )
            self.norm_layers.append(nn.BatchNorm1d(out_dim))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Final projection to representation space
        self.repr_proj = nn.Linear(hidden_dims[-1], repr_dim)

    def forward(
        self,
        x: torch.Tensor,
        purpose_idx: torch.Tensor | int,
    ) -> torch.Tensor:
        """Encode input features conditioned on purpose.

        Args:
            x: Input features of shape (batch_size, input_dim).
            purpose_idx: Purpose index. Either a scalar int or tensor of
                shape (batch_size,).

        Returns:
            Encoded representation h_p of shape (batch_size, repr_dim).
        """
        batch_size = x.shape[0]

        # Get purpose embedding
        if isinstance(purpose_idx, int):
            purpose_idx_tensor = torch.full(
                (batch_size,),
                purpose_idx,
                device=x.device,
                dtype=torch.long,
            )
        else:
            purpose_idx_tensor = purpose_idx

        purpose_emb = self.purpose_embedding(purpose_idx_tensor)

        # Forward through layers
        h = x
        for linear, cond, norm in zip(
            self.linear_layers, self.conditioning_layers, self.norm_layers
        ):
            h = linear(h)
            h = cond(h, purpose_emb)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)

        # Project to representation space
        h_p = self.repr_proj(h)

        return h_p

    def forward_with_embedding(
        self,
        x: torch.Tensor,
        purpose_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input features using a precomputed purpose embedding.

        This is used for compositional purposes where the embedding is
        derived from combining multiple purpose embeddings (e.g., additive
        for AND, element-wise max for OR).

        Args:
            x: Input features of shape (batch_size, input_dim).
            purpose_emb: Purpose embedding of shape (batch_size, purpose_emb_dim).

        Returns:
            Encoded representation h_p of shape (batch_size, repr_dim).
        """
        h = x
        for linear, cond, norm in zip(
            self.linear_layers, self.conditioning_layers, self.norm_layers
        ):
            h = linear(h)
            h = cond(h, purpose_emb)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)

        return self.repr_proj(h)

    def encode_all_purposes(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Encode input for all purposes.

        Args:
            x: Input features of shape (batch_size, input_dim).

        Returns:
            Dictionary mapping purpose index to encoded representation.
        """
        representations = {}
        for idx in range(self.num_purposes):
            representations[idx] = self.forward(x, idx)
        return representations

    def get_purpose_embedding(self, purpose_idx: int) -> torch.Tensor:
        """Get the embedding vector for a specific purpose.

        Args:
            purpose_idx: Index of the purpose.

        Returns:
            Purpose embedding of shape (purpose_emb_dim,).
        """
        idx_tensor = torch.tensor(
            [purpose_idx], device=self.purpose_embedding.weight.device
        )
        return self.purpose_embedding(idx_tensor).squeeze(0)
