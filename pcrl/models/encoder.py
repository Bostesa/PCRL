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

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Kaiming for hidden layers (ReLU), Xavier for projection."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(self.repr_proj.weight)
        if self.repr_proj.bias is not None:
            nn.init.zeros_(self.repr_proj.bias)

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

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Kaiming for hidden layers (ReLU), Xavier for projection."""
        for layer in self.linear_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.repr_proj.weight)
        if self.repr_proj.bias is not None:
            nn.init.zeros_(self.repr_proj.bias)

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

    def forward_sequential_composition(
        self,
        x: torch.Tensor,
        purpose_embs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Encode x by applying multiple purpose FiLM heads in sequence.

        At every hidden layer, the conditioning module is applied once per
        purpose embedding, in the given order. For FiLM specifically this
        realises multiplicative gate composition:

            h <- linear(h)
            for e_p in purpose_embs:
                gamma_p, beta_p = film(e_p)
                h = gamma_p * h + beta_p
            h = norm(h); ReLU; dropout

        The effective per-layer gamma is the elementwise product of the
        per-purpose gammas: gamma_eff = prod_p gamma_p. Suppression by any
        single purpose is preserved because zero anywhere in the product
        forces the composed gate to zero, regardless of the other gates.

        This contrasts with additive embedding composition
        (forward_with_embedding(x, sum_p e_p)), where the combined gamma
        equals sum_p gamma_p - (K - 1) * b_gamma — a learned-bias-dependent
        sum that does not preserve multiplicative suppression.

        Args:
            x: Input features of shape (batch_size, input_dim).
            purpose_embs: List of K purpose embeddings, each shape
                (batch_size, purpose_emb_dim) or (purpose_emb_dim,).
                Must contain at least one embedding.

        Returns:
            Composed representation h_p of shape (batch_size, repr_dim).
        """
        if not purpose_embs:
            raise ValueError("forward_sequential_composition requires at least one embedding")

        h = x
        for linear, cond, norm in zip(
            self.linear_layers, self.conditioning_layers, self.norm_layers
        ):
            h = linear(h)
            for emb in purpose_embs:
                h = cond(h, emb)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)

        return self.repr_proj(h)

    def encode_all_purposes(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Encode input for all purposes in a single batched forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim).

        Returns:
            Dictionary mapping purpose index to encoded representation.
        """
        batch_size = x.shape[0]
        K = self.num_purposes

        # Expand x: (batch*K, input_dim)
        x_expanded = x.repeat(K, 1)

        # Build purpose indices: [0,0,...,1,1,...,K-1,K-1,...]
        purpose_idxs = torch.arange(K, device=x.device).repeat_interleave(batch_size)

        # Single forward pass for all purposes
        all_reps = self.forward(x_expanded, purpose_idxs)

        # Split back into per-purpose tensors
        return {idx: all_reps[idx * batch_size : (idx + 1) * batch_size]
                for idx in range(K)}

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
