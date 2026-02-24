"""Purpose-conditioned encoder for PCRL."""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Applies affine transformation conditioned on purpose embedding:
        h = gamma(purpose_emb) * h + beta(purpose_emb)
    where gamma and beta are learned linear projections from the purpose embedding.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int) -> None:
        """Initialize FiLM layer.

        Args:
            feature_dim: Dimension of features to modulate.
            conditioning_dim: Dimension of conditioning embedding.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim

        self.gamma_proj = nn.Linear(conditioning_dim, feature_dim)
        self.beta_proj = nn.Linear(conditioning_dim, feature_dim)

        # Initialize to identity transform: gamma=1, beta=0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, purpose_emb: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            h: Input features of shape (batch_size, feature_dim).
            purpose_emb: Purpose embedding of shape (batch_size, conditioning_dim).

        Returns:
            Modulated features: gamma * h + beta, shape (batch_size, feature_dim).
        """
        gamma = self.gamma_proj(purpose_emb)
        beta = self.beta_proj(purpose_emb)
        return gamma * h + beta


class PurposeConditionedEncoder(nn.Module):
    """Purpose-conditioned encoder network.

    Learns representations that are conditioned on the specified purpose,
    enabling purpose-specific information filtering.

    Architecture per layer:
        Linear -> FiLM -> LayerNorm -> GELU -> Dropout

    Same input x with different purpose_idx will produce different representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        repr_dim: int,
        num_purposes: int,
        purpose_emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder.

        Args:
            input_dim: Dimensionality of input features.
            hidden_dims: List of hidden layer dimensions.
            repr_dim: Dimensionality of output representation.
            num_purposes: Number of distinct purposes.
            purpose_emb_dim: Dimensionality of purpose embeddings.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.repr_dim = repr_dim
        self.num_purposes = num_purposes
        self.purpose_emb_dim = purpose_emb_dim

        # Learnable purpose embeddings
        self.purpose_embedding = nn.Embedding(
            num_embeddings=num_purposes,
            embedding_dim=purpose_emb_dim,
        )

        # Build encoder layers: Linear -> FiLM -> LayerNorm -> GELU -> Dropout
        self.linear_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            self.linear_layers.append(nn.Linear(in_dim, out_dim))
            self.film_layers.append(FiLMLayer(out_dim, purpose_emb_dim))
            self.norm_layers.append(nn.LayerNorm(out_dim))

        self.activation = nn.GELU()
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
            purpose_idx: Purpose index. Either a scalar int or tensor of shape (batch_size,).

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

        # Forward through layers: Linear -> FiLM -> LayerNorm -> GELU -> Dropout
        h = x
        for linear, film, norm in zip(
            self.linear_layers, self.film_layers, self.norm_layers
        ):
            h = linear(h)
            h = film(h, purpose_emb)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)

        # Project to representation space
        h_p = self.repr_proj(h)

        return h_p

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
        idx_tensor = torch.tensor([purpose_idx], device=self.purpose_embedding.weight.device)
        return self.purpose_embedding(idx_tensor).squeeze(0)


class EncoderWithProjection(nn.Module):
    """Encoder with optional projection head for contrastive learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        repr_dim: int,
        num_purposes: int,
        purpose_emb_dim: int,
        projection_dim: int = 128,
        use_projection: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """Initialize encoder with projection.

        Args:
            input_dim: Dimensionality of input features.
            hidden_dims: List of hidden layer dimensions.
            repr_dim: Dimensionality of representation.
            num_purposes: Number of distinct purposes.
            purpose_emb_dim: Dimensionality of purpose embeddings.
            projection_dim: Dimension of projection head output.
            use_projection: Whether to use projection head.
            dropout: Dropout probability.
        """
        super().__init__()
        self.encoder = PurposeConditionedEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            repr_dim=repr_dim,
            num_purposes=num_purposes,
            purpose_emb_dim=purpose_emb_dim,
            dropout=dropout,
        )
        self.use_projection = use_projection
        self.repr_dim = repr_dim

        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(repr_dim, repr_dim),
                nn.GELU(),
                nn.Linear(repr_dim, projection_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        purpose_idx: torch.Tensor | int,
        return_projection: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features.
            purpose_idx: Purpose index(es).
            return_projection: Whether to also return projection.

        Returns:
            Representation, or tuple of (representation, projection).
        """
        h = self.encoder(x, purpose_idx)

        if return_projection and self.use_projection:
            proj = self.projection(h)
            return h, proj

        return h

    def encode_all_purposes(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Encode input for all purposes."""
        return self.encoder.encode_all_purposes(x)
