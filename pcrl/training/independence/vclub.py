"""vCLUB: variational Contrastive Log-ratio Upper Bound on mutual information.

Provides a tractable upper bound on I(X; Z) by training a small variational
posterior q(Z|X) and using a contrastive log-ratio estimator. The encoder
backprops through `mi_upper_bound`; the variational posterior is trained
separately by minimizing `learning_loss`.

Two practical safeguards differ from the bare textbook formulation:
  * `learning_loss` adds an L2 weight-decay term (controlled by `l2`,
    default 1e-1). Without it, q overfits on small fixed batches and the
    bound diverges away from the true I(X;Z) toward the per-sample
    memorization gap.
  * The continuous-Z log-prob clamps logvar to [-2, 2]. Without the clamp,
    a memorizing q drives logvar toward -inf so paired log-probs grow
    unboundedly, breaking the bound estimator numerically.

Both are knobs on the constructor and can be relaxed when training on
i.i.d. minibatches against a held-out evaluation set, where overfitting
is naturally suppressed.

Ref:
    Cheng et al., ICML 2020, "CLUB: A Contrastive Log-ratio Upper Bound
        of Mutual Information."
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VCLUB(nn.Module):
    """Variational CLUB upper bound on I(X; Z).

    Args:
        x_dim: representation dimension.
        z_dim: dimension of Z (output dim of mu/logvar nets when continuous,
            number of classes when categorical).
        hidden_dim: MLP hidden width.
        z_categorical: if True, Z is categorical (class indices) and q
            outputs class logits; if False, Z is continuous and q outputs
            Gaussian mean and log-variance.
        l2: coefficient on the L2 weight-decay term added to learning_loss.
            Set to 0 to disable.
        logvar_min, logvar_max: clamp range for the continuous-Z logvar
            head. Defaults to [-2, 2] (variance in [0.135, 7.39]).
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        z_categorical: bool = False,
        l2: float = 1e-1,
        logvar_min: float = -2.0,
        logvar_max: float = 2.0,
    ):
        super().__init__()
        self.z_categorical = z_categorical
        self.z_dim = z_dim
        self.l2 = l2
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.shared = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if z_categorical:
            self.head = nn.Linear(hidden_dim, z_dim)
        else:
            self.mu_head = nn.Linear(hidden_dim, z_dim)
            self.logvar_head = nn.Linear(hidden_dim, z_dim)

    def _q_log_prob(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """log q(Z | X), shape [batch]."""
        h = self.shared(X)
        if self.z_categorical:
            logits = self.head(h)
            return -F.cross_entropy(logits, Z, reduction="none")
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=self.logvar_min, max=self.logvar_max)
        log_two_pi = math.log(2.0 * math.pi)
        return -0.5 * (logvar + (Z - mu) ** 2 / logvar.exp() + log_two_pi).sum(dim=-1)

    def learning_loss(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Loss used to train q. NLL plus an L2 weight-decay term.

        X is detached so the encoder is not updated by this loss.
        """
        nll = -self._q_log_prob(X.detach(), Z).mean()
        if self.l2 > 0.0:
            wd = sum((p ** 2).sum() for p in self.parameters())
            return nll + self.l2 * wd
        return nll

    def mi_upper_bound(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """vCLUB upper bound on I(X; Z).

        Encoder backprops through this. Implementation:
            bound = E_i[log q(Z_i | X_i)] - E_i E_j[log q(Z_j | X_i)]
        Approximated by paired log-prob minus the mean over 3 random
        permutations of Z.
        """
        log_prob_paired = self._q_log_prob(X, Z)
        n = X.shape[0]
        all_pairs = []
        for _ in range(3):
            perm = torch.randperm(n, device=X.device)
            Z_shuffled = Z[perm]
            all_pairs.append(self._q_log_prob(X, Z_shuffled))
        log_prob_negative = torch.stack(all_pairs).mean(0)
        return (log_prob_paired - log_prob_negative).mean()
