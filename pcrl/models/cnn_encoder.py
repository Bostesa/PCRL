"""CNN-based encoders for image data.

Two conditioning strategies:
- CNNEncoder: FiLM conditioning on conv layers (per-channel scale+shift).
  Works for non-conflicting purposes but collapses when attributes are
  both allowed and disallowed across different purposes.
- CNNPurposeProjectionEncoder: Shared conv backbone with per-purpose
  projection heads. Each purpose gets its own MLP from flattened conv
  features to repr_dim, enabling selective feature routing without the
  entanglement issues of FiLM.

Designed for CelebA face images (64x64x3).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from pcrl.models.conditioning import build_conditioner


class CNNEncoder(nn.Module):
    """CNN encoder with purpose conditioning for image data.

    Architecture per conv layer:
        Conv2d -> FiLM(purpose_emb) -> BatchNorm2d -> ReLU

    FiLM is applied channel-wise: gamma and beta are predicted per channel
    from the purpose embedding, then broadcast over spatial dimensions.
    """

    def __init__(
        self,
        repr_dim: int,
        num_purposes: int,
        purpose_emb_dim: int = 32,
        in_channels: int = 3,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.num_purposes = num_purposes
        self.purpose_emb_dim = purpose_emb_dim

        # Learnable purpose embeddings
        self.purpose_embedding = nn.Embedding(num_purposes, purpose_emb_dim)

        # Conv layers with FiLM conditioning
        self.conv_layers = nn.ModuleList()
        self.film_gamma = nn.ModuleList()
        self.film_beta = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        channels = [in_channels] + list(conv_channels)
        for i in range(len(conv_channels)):
            self.conv_layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1)
            )
            # FiLM: predict per-channel gamma and beta from purpose embedding
            self.film_gamma.append(nn.Linear(purpose_emb_dim, channels[i + 1]))
            self.film_beta.append(nn.Linear(purpose_emb_dim, channels[i + 1]))
            self.bn_layers.append(nn.BatchNorm2d(channels[i + 1]))

        # Initialize FiLM to identity
        for gamma_layer, beta_layer in zip(self.film_gamma, self.film_beta):
            nn.init.zeros_(gamma_layer.weight)
            nn.init.ones_(gamma_layer.bias)
            nn.init.zeros_(beta_layer.weight)
            nn.init.zeros_(beta_layer.bias)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Compute flattened size: 64x64 with N stride-2 layers
        # Each stride-2 conv halves spatial dims: 64 -> 32 -> 16 -> 8 -> 4 ...
        spatial = 64
        for _ in range(len(conv_channels)):
            spatial = (spatial + 1) // 2  # ceil(spatial / 2) with padding=1
        self._flat_dim = conv_channels[-1] * spatial * spatial

        # Linear projection to repr_dim
        proj_hidden = max(256, repr_dim)
        self.repr_proj = nn.Sequential(
            nn.Linear(self._flat_dim, proj_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, repr_dim),
        )

    def _apply_film(
        self,
        h: torch.Tensor,
        purpose_emb: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Apply channel-wise FiLM conditioning to a 4D feature map."""
        gamma = self.film_gamma[layer_idx](purpose_emb)  # (B, C)
        beta = self.film_beta[layer_idx](purpose_emb)    # (B, C)
        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * h + beta

    def _forward_impl(
        self,
        x: torch.Tensor,
        purpose_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Core forward pass with a purpose embedding."""
        h = x
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            h = conv(h)
            h = self._apply_film(h, purpose_emb, i)
            h = bn(h)
            h = self.activation(h)

        # Flatten and project
        h = h.flatten(1)
        h = self.dropout(h)
        return self.repr_proj(h)

    def forward(
        self,
        x: torch.Tensor,
        purpose_idx: torch.Tensor | int,
    ) -> torch.Tensor:
        """Encode image features conditioned on purpose.

        Args:
            x: Input images of shape (batch_size, 3, 64, 64).
            purpose_idx: Purpose index (scalar int or tensor of shape (batch_size,)).

        Returns:
            Representation of shape (batch_size, repr_dim).
        """
        batch_size = x.shape[0]
        if isinstance(purpose_idx, int):
            idx = torch.full((batch_size,), purpose_idx, device=x.device, dtype=torch.long)
        else:
            idx = purpose_idx
        purpose_emb = self.purpose_embedding(idx)
        return self._forward_impl(x, purpose_emb)

    def forward_with_embedding(
        self,
        x: torch.Tensor,
        purpose_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Encode using a precomputed purpose embedding (for composition)."""
        return self._forward_impl(x, purpose_emb)

    def get_purpose_embedding(self, purpose_idx: int) -> torch.Tensor:
        """Get the embedding vector for a specific purpose."""
        idx_tensor = torch.tensor(
            [purpose_idx], device=self.purpose_embedding.weight.device
        )
        return self.purpose_embedding(idx_tensor).squeeze(0)


class StandardCNNEncoder(nn.Module):
    """CNN encoder without purpose conditioning (baseline).

    Same architecture as CNNEncoder but ignores purpose_idx.
    """

    def __init__(
        self,
        repr_dim: int,
        in_channels: int = 3,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim

        layers: list[nn.Module] = []
        channels = [in_channels] + list(conv_channels)
        for i in range(len(conv_channels)):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)
        spatial = 64
        for _ in range(len(conv_channels)):
            spatial = (spatial + 1) // 2
        self._flat_dim = conv_channels[-1] * spatial * spatial
        proj_hidden = max(256, repr_dim)
        self.repr_proj = nn.Sequential(
            nn.Linear(self._flat_dim, proj_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, repr_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        purpose_idx: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Encode image features (purpose_idx is accepted but ignored)."""
        h = self.conv(x)
        h = h.flatten(1)
        h = self.dropout(h)
        return self.repr_proj(h)


class _BackboneGradientScale(torch.autograd.Function):
    """Scale gradients flowing back to the conv backbone.

    Forward: identity (no change to activations).
    Backward: multiply gradient by scale factor.

    With scale=0.1, only 10% of the adversarial gradient reaches the
    conv layers. This protects the backbone from being overwhelmed by
    conflicting adversarial signals from multiple purposes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output * ctx.scale, None


class CNNPurposeProjectionEncoder(nn.Module):
    """CNN encoder with per-purpose projection heads and backbone gradient protection.

    Shared conv backbone extracts visual features, then each purpose has
    its own MLP projection from flattened features to repr_dim. A gradient
    scaling layer between backbone and projections protects the shared conv
    weights from being overwhelmed by adversarial gradients.

    Architecture:
        Shared: Conv2d -> BatchNorm2d -> ReLU  (× N layers)
        Gradient scale: reduces backward gradient by backbone_grad_scale
        Per-purpose: Flatten -> MLP_p -> repr_dim

    The gradient scaling solves the core CelebA problem: with 5 purposes
    and 4 conflicting attributes, the adversary contributes ~8× more
    gradient to the shared conv than the task loss. Without scaling, the
    conv backbone collapses. With scale=0.1, the conv receives mostly
    task-relevant gradients and stays healthy.
    """

    def __init__(
        self,
        repr_dim: int,
        num_purposes: int,
        in_channels: int = 3,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
        backbone_grad_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.num_purposes = num_purposes
        self._backbone_grad_scale = backbone_grad_scale

        # Shared conv backbone (no conditioning — just feature extraction)
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        channels = [in_channels] + list(conv_channels)
        for i in range(len(conv_channels)):
            self.conv_layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(channels[i + 1]))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Compute flattened size
        spatial = 64
        for _ in range(len(conv_channels)):
            spatial = (spatial + 1) // 2
        self._flat_dim = conv_channels[-1] * spatial * spatial

        # Per-purpose projection heads
        proj_hidden = max(256, repr_dim)
        self.purpose_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._flat_dim, proj_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_hidden, repr_dim),
            )
            for _ in range(num_purposes)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for conv layers, Xavier for projection output."""
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for proj in self.purpose_projections:
            for i, module in enumerate(proj):
                if isinstance(module, nn.Linear):
                    if i == len(list(proj)) - 1:
                        nn.init.xavier_normal_(module.weight)
                    else:
                        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run shared conv backbone with gradient protection."""
        h = x
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h)
            h = self.activation(h)
        h = h.flatten(1)
        h = self.dropout(h)
        # Scale down gradients flowing back to conv layers.
        # Forward: identity. Backward: grad *= backbone_grad_scale.
        # This protects the conv from adversarial gradient domination.
        if self.training and self._backbone_grad_scale < 1.0:
            h = _BackboneGradientScale.apply(h, self._backbone_grad_scale)
        return h

    def forward(
        self,
        x: torch.Tensor,
        purpose_idx: torch.Tensor | int,
    ) -> torch.Tensor:
        """Encode image features with purpose-specific projection.

        Args:
            x: Input images of shape (batch_size, 3, 64, 64).
            purpose_idx: Purpose index (scalar int or tensor of shape (batch_size,)).

        Returns:
            Representation of shape (batch_size, repr_dim).
        """
        flat = self._conv_forward(x)

        if isinstance(purpose_idx, int):
            return self.purpose_projections[purpose_idx](flat)

        # Handle per-sample purpose indices (rare but supported)
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.repr_dim, device=x.device)
        for p in range(self.num_purposes):
            mask = purpose_idx == p
            if mask.any():
                output[mask] = self.purpose_projections[p](flat[mask])
        return output

    def encode_all_purposes(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Encode input for all purposes efficiently (single conv pass)."""
        flat = self._conv_forward(x)
        return {p: self.purpose_projections[p](flat) for p in range(self.num_purposes)}
