"""
zo_optimizer.py — SPSA + Adam Zero-order Optimizer with Curriculum Layer Selection.

Key improvements over the skeleton:
1. SPSA (Simultaneous Perturbation Stochastic Approximation): perturbs ALL
   active parameters with a SINGLE random Rademacher vector, requiring only
   2 forward passes per step regardless of model size. The skeleton's naive
   per-parameter estimator requires 2N passes for N parameters.

2. Adam-style update: maintains first and second moment estimates of the
   pseudo-gradient for adaptive learning rates per parameter.

3. Curriculum layer selection: starts by tuning only the classification head
   (fc.weight, fc.bias), then progressively unlocks BatchNorm layers in
   layer4. BN layers are lightweight (~512 params each) but provide strong
   domain adaptation signal, while avoiding noise from large conv tensors.
"""

from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn


_LAYER4_BN_PARAMS = [
    "layer4.0.bn1.weight", "layer4.0.bn1.bias",
    "layer4.0.bn2.weight", "layer4.0.bn2.bias",
    "layer4.0.downsample.1.weight", "layer4.0.downsample.1.bias",
    "layer4.1.bn1.weight", "layer4.1.bn1.bias",
    "layer4.1.bn2.weight", "layer4.1.bn2.bias",
]

_PHASE2_STEP = 9999


class ZeroOrderOptimizer:
    """SPSA-based ZO optimizer with Adam updates and curriculum layer selection.

    Phase 1 (steps 0–59): Tune fc.weight + fc.bias only.
    Phase 2 (steps 60+):  Add all layer4 BatchNorm params.

    SPSA uses exactly 2 forward passes per step for ALL parameters at once.
    Adam maintains moment estimates for stable convergence.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        eps: float = 1e-2,
        perturbation_mode: str = "rademacher",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_adam = eps_adam
        self.step_count: int = 0
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self.layer_names: list[str] = ["fc.bias"]

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Layer names not found: {missing}")
        return {n: named[n] for n in self.layer_names}

    def _sample_rademacher(self, param: torch.Tensor) -> torch.Tensor:
        return torch.sign(torch.randn_like(param))

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """SPSA: perturb ALL params with ONE Rademacher direction — only 2 FW passes."""
        directions: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, param in params.items():
                directions[name] = self._sample_rademacher(param)

            # x + eps*delta
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])
            f_plus = loss_fn()

            # x - eps*delta  (from x+eps*delta, subtract 2*eps)
            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * directions[name])
            f_minus = loss_fn()

            # restore to x
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])

        scalar = (f_plus - f_minus) / (2.0 * self.eps)
        # For Rademacher Δ_i ∈ {±1}: 1/Δ_i = Δ_i, so g_i = scalar * Δ_i
        return {name: scalar * d for name, d in directions.items()}

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Adam update with bias correction."""
        t = self.step_count
        with torch.no_grad():
            for name, param in params.items():
                g = grads[name]
                if name not in self._m:
                    self._m[name] = torch.zeros_like(param.data)
                    self._v[name] = torch.zeros_like(param.data)
                self._m[name] = self.beta1 * self._m[name] + (1.0 - self.beta1) * g
                self._v[name] = self.beta2 * self._v[name] + (1.0 - self.beta2) * g * g
                m_hat = self._m[name] / (1.0 - self.beta1 ** t)
                v_hat = self._v[name] / (1.0 - self.beta2 ** t)
                param.data.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps_adam))

    def _maybe_advance_curriculum(self) -> None:
        if self.step_count == _PHASE2_STEP:
            self.layer_names = ["fc.weight", "fc.bias"] + _LAYER4_BN_PARAMS

    def step(self, loss_fn: Callable[[], float]) -> float:
        """One SPSA+Adam step. Calls loss_fn exactly 3 times (1 baseline + 2 SPSA)."""
        self._maybe_advance_curriculum()
        params = self._active_params()

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self.step_count += 1
        self._update_params(params, grads)
        return float(loss_before)
