# -*- coding: utf-8 -*-
"""
VACE: Variance-Adaptive Cross-Entropy (Supervised Only)
-------------------------------------------------------
A minimally invasive modification to Cross-Entropy (CE): introduces a class-wise temperature τ_c for each class,
using **batch variance of own logit** (smoothed by EMA) as an intra-class dispersion estimate, defined as
    τ_c = clip(a + b * σ_c^2, τ_min, τ_max)
Finally, logits are scaled by per-class temperature before CE.

**Features**
- Only O(K) class-scale parameters and EMA statistics; no contrastive learning, no extra memory buffer.
- Almost same speed and memory as CE; can directly replace `nn.CrossEntropyLoss`.
- Compatible with any `timm` CNN backbone: use `model.forward_features(x)` to get pooled features,
  then a linear layer for logits (or use logits from existing classifier head).

**Usage (minimal change)**
>>> loss_fn = VACE(num_classes=K, a=1.0, b=1.0, tau_min=0.5, tau_max=2.0, ema_decay=0.1)
>>> logits = model(x)                         # use model output logits directly
>>> loss = loss_fn(logits, target)            # EMA stats auto-updated in training mode

**Advanced (custom classifier head using forward_features)**
- See `VACEHead`: wraps a linear classifier on top of `forward_features`, keeping timm compatibility.

Author: Provided for WACV prototype use
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAStats(nn.Module):
    """Maintains EMA-smoothed statistics (mean and variance; scalar) of own logit for *each class*.

    Motivation: In linear classifier z_{i,k} = w_k^T f_i + b_k,
    intra-class dispersion Cov(f|y=k) and own logit variance Var(z_{i,k}|y=k) are monotonically related via w_k,
    using the latter as an approximation greatly reduces memory (only two scalars per class), no need to save vector means.

    Args:
        num_classes: Number of classes K
        ema_decay: EMA smoothing coefficient ρ ∈ (0,1], larger means more reliance on recent batch.
        eps: Small constant to avoid numerical issues
    """

    def __init__(self, num_classes: int, ema_decay: float = 0.1, eps: float = 1e-12):
        super().__init__()
        assert num_classes > 1
        assert 0.0 < ema_decay <= 1.0
        self.num_classes = num_classes
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)

        # Use register_buffer to save with model and move across devices
        self.register_buffer("mean", torch.zeros(num_classes))       # E[z_k | y=k]
        self.register_buffer("var", torch.zeros(num_classes))        # Var[z_k | y=k]
        self.register_buffer("count", torch.zeros(num_classes, dtype=torch.long))  # Accumulated sample count

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        """Update EMA mean and variance of own logit for each class using current batch.

        Only depends on *own* logit: for class c, take x = logits[ target==c, c ].
        Skip if no samples of that class in batch.
        """
        assert logits.dim() == 2, "logits should be (B, K)"
        B, K = logits.shape
        assert K == self.num_classes, "logits class count must match num_classes"
        assert target.shape[0] == B

        device = logits.device
        # Ensure buffer is on same device
        if self.mean.device != device:
            self.mean = self.mean.to(device)
            self.var = self.var.to(device)
            self.count = self.count.to(device)

        for c in range(K):
            mask = (target == c)
            n_c = int(mask.sum().item())
            if n_c == 0:
                continue
            x = logits[mask, c].detach()
            # Batch statistics (biased)
            batch_mean = x.mean()
            # Use biased estimate (matches PyTorch var(unbiased=False); more stable)
            batch_var = x.var(unbiased=False)

            if self.count[c] == 0:
                new_mean = batch_mean
                new_var = batch_var
            else:
                m = self.ema_decay
                new_mean = (1 - m) * self.mean[c] + m * batch_mean
                new_var = (1 - m) * self.var[c] + m * batch_var

            self.mean[c] = new_mean
            self.var[c] = torch.clamp(new_var, min=self.eps)
            self.count[c] += n_c

    def get_variance(self) -> torch.Tensor:
        """Return EMA-smoothed own logit variance (σ_c^2) for each class, shape (K,)."""
        return self.var

    def extra_repr(self) -> str:
        return f"EMAStats(K={self.num_classes}, decay={self.ema_decay})"


class VACE(nn.Module):
    """Variance-Adaptive Cross-Entropy (VACE)

    Loss definition: For each class k, use class-wise temperature τ_k to scale logits column-wise,
        L = CE( z / τ ), where τ_k = clip(a + b * σ_k^2, τ_min, τ_max).
    Here σ_k^2 is estimated by `EMAStats` from own logit variance.

    Args:
        num_classes: Number of classes K
        a, b: τ function base and slope (recommend a=1.0)
        tau_min, tau_max: Numerical stability bounds (recommend 0.5 ~ 2.0)
        ema_decay: EMAStats smoothing coefficient
        label_smoothing: Optional, add label smoothing (default 0.0)
        reduction: 'mean' | 'sum' | 'none'
        init_tau: Temporary τ for unseen classes (default = a)
        eps: Small constant
    """

    def __init__(
        self,
        num_classes: int,
        a: float = 1.0,
        b: float = 1.0,
        tau_min: float = 0.5,
        tau_max: float = 2.0,
        ema_decay: float = 0.1,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        init_tau: Optional[float] = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        assert num_classes > 1
        assert tau_min > 0 and tau_max > tau_min
        assert a > 0 and b >= 0
        if init_tau is None:
            init_tau = a

        self.num_classes = num_classes
        self.a = float(a)
        self.b = float(b)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self.eps = float(eps)
        self.init_tau = float(init_tau)

        self.stats = EMAStats(num_classes=num_classes, ema_decay=ema_decay, eps=eps)
        # Cache τ vector (updated during training), stored as buffer
        self.register_buffer("_tau", torch.full((num_classes,), self.init_tau))

    def _compute_tau(self) -> torch.Tensor:
        """Compute τ from current EMA variance, clipped to [tau_min, tau_max].
        For unseen classes (count=0), use init_tau.
        """
        var = self.stats.get_variance()  # (K,)
        tau = self.a + self.b * var
        tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)

        # Replace unseen classes with init_tau
        unseen = (self.stats.count == 0)
        if unseen.any():
            tau = tau.clone()
            tau[unseen] = self.init_tau
        return tau

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute VACE loss.

        Args:
            logits: (B, K) model output (pre-softmax)
            target: (B,) integer class labels
        Returns:
            Scalar loss or per-sample loss (depends on reduction)
        """
        assert logits.dim() == 2 and logits.size(1) == self.num_classes
        assert target.dim() == 1 and target.size(0) == logits.size(0)

        if self.training:
            # Update EMA stats (no grad)
            self.stats.update(logits, target)

        # Get and cache τ; match logits dtype/device
        tau = self._compute_tau().to(dtype=logits.dtype, device=logits.device)
        self._tau = tau.detach()

        # Per-class scaling: scale columns (class dimension)
        logits_scaled = logits / (tau.clamp(min=self.eps))  # broadcasting (B,K) / (K,)

        loss = F.cross_entropy(
            logits_scaled,
            target,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        return loss

    @torch.no_grad()
    def get_tau(self) -> torch.Tensor:
        """Get current τ (size K), for logging/visualization."""
        return self._tau.detach().clone()

    def extra_repr(self) -> str:
        return (
            f"VACE(K={self.num_classes}, a={self.a}, b={self.b}, "
            f"tau∈[{self.tau_min},{self.tau_max}], ema_decay={self.stats.ema_decay}, "
            f"ls={self.label_smoothing}, red='{self.reduction}')"
        )


class VACEHead(nn.Module):
    """(Optional) Minimal classifier head template compatible with `timm` models.

    Usage: If you want to always get features from `forward_features` / `feature_forward()`,
    then use a linear layer for logits to feed into `VACE` loss, but don't want to modify backbone.

    Typical usage:
        model = timm.create_model('resnet50', pretrained=True, num_classes=0)  # remove classifier head
        head = VACEHead(in_features=model.num_features, num_classes=K)
        logits = head(model, x)  # internally calls model.forward_features(x)
        loss = loss_fn(logits, target)

    Note: This class is a template; in practice, implement pooled features according to backbone.
    For most CNNs (timm), `forward_features(x)` returns global pooled feature.
    If your model provides `.feature_forward()`, you can call that instead.
    """

    def __init__(self, in_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(backbone, "forward_features"):
            feat = backbone.forward_features(x)
        elif hasattr(backbone, "feature_forward"):
            feat = backbone.feature_forward(x)
        else:
            raise AttributeError("backbone must provide forward_features or feature_forward")
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        # If forward_features returns dict or multi-scale, extract pooled vector yourself
        if feat.dim() > 2:
            # Assume (B, C, H, W) -> global average pooling
            feat = feat.mean(dim=(2, 3))
        return self.fc(feat)


if __name__ == "__main__":
    # Simple self-test: random logits and labels
    torch.manual_seed(0)
    B, K = 8, 5
    logits = torch.randn(B, K)
    target = torch.randint(0, K, (B,))

    loss_fn = VACE(num_classes=K, a=1.0, b=1.0, tau_min=0.5, tau_max=2.0, ema_decay=0.2)
    loss = loss_fn(logits, target)
    print("loss:", float(loss))
    print("tau:", loss_fn.get_tau().tolist())
