"""Scanner ULTRA — Differential Privacy for Federated Learning.

Adds calibrated noise to model updates to preserve privacy.
Implements ε-differential privacy (ε-DP) for GDPR/KVKK compliance.

References:
- Differential Privacy: https://arxiv.org/abs/1607.00133
- DP-SGD: https://arxiv.org/abs/1607.00133
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """Add differential privacy noise to model updates.

    Implements:
    - Gaussian mechanism for ε-differential privacy
    - Gradient clipping (sensitivity bound)
    - Privacy budget tracking
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        noise_multiplier: float | None = None,
    ) -> None:
        """Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (lower = more private, typical: 0.1-10)
            delta: Failure probability (typical: 1e-5)
            sensitivity: L2 sensitivity bound (gradient clipping threshold)
            noise_multiplier: Noise scale multiplier (auto-computed if None)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

        # Compute noise multiplier from epsilon/delta if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

        # Privacy budget tracking
        self.privacy_spent = 0.0
        self.num_queries = 0

        logger.info(
            f"DifferentialPrivacy initialized "
            f"(ε={epsilon}, δ={delta}, sensitivity={sensitivity}, "
            f"noise_multiplier={self.noise_multiplier:.4f})"
        )

    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier from epsilon and delta.

        Uses the analytic Gaussian mechanism formula.

        Returns:
            Noise multiplier σ
        """
        # Simplified formula (for better accuracy, use accountant)
        # σ = sqrt(2 * ln(1.25/δ)) / ε
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return float(sigma)

    def add_noise(
        self,
        update: dict[str, torch.Tensor],
        clip_norm: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Add differential privacy noise to model update.

        Args:
            update: Model update (gradients or parameters)
            clip_norm: Whether to clip gradients to sensitivity bound

        Returns:
            Privatized update with added noise
        """
        privatized_update = {}

        # Clip gradients if requested
        if clip_norm:
            update = self._clip_gradients(update)

        # Add Gaussian noise to each parameter
        for key, value in update.items():
            # Compute noise scale
            noise_scale = self.noise_multiplier * self.sensitivity

            # Generate Gaussian noise
            noise = torch.randn_like(value) * noise_scale

            # Add noise
            privatized_update[key] = value + noise

        # Update privacy budget
        self.num_queries += 1
        self.privacy_spent = self._compute_privacy_spent()

        logger.debug(
            f"Added DP noise (σ={self.noise_multiplier * self.sensitivity:.4f}, "
            f"privacy spent: ε={self.privacy_spent:.4f})"
        )

        return privatized_update

    def _clip_gradients(
        self,
        update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Clip gradients to sensitivity bound (L2 norm clipping).

        Args:
            update: Model update

        Returns:
            Clipped update
        """
        # Compute L2 norm of all parameters
        total_norm = torch.sqrt(sum(torch.sum(param**2) for param in update.values()))

        # Clip if norm exceeds sensitivity
        clip_coef = self.sensitivity / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)

        # Apply clipping
        clipped_update = {key: value * clip_coef for key, value in update.items()}

        if clip_coef < 1.0:
            logger.debug(f"Gradients clipped (norm: {total_norm:.4f} → {self.sensitivity})")

        return clipped_update

    def _compute_privacy_spent(self) -> float:
        """Compute total privacy spent so far.

        Uses composition theorem: privacy degrades with more queries.
        For Gaussian mechanism, we use basic composition (conservative).

        Returns:
            Total epsilon spent
        """
        # Basic composition: ε_total = n * ε_single
        # (conservative; advanced composition gives better bounds)
        epsilon_single = 1.0 / self.noise_multiplier
        epsilon_total = self.num_queries * epsilon_single

        return float(epsilon_total)

    def add_noise_to_tensor(
        self,
        tensor: torch.Tensor,
        sensitivity: float | None = None,
    ) -> torch.Tensor:
        """Add DP noise to a single tensor.

        Args:
            tensor: Input tensor
            sensitivity: Override sensitivity for this tensor

        Returns:
            Privatized tensor
        """
        sens = sensitivity if sensitivity is not None else self.sensitivity
        noise_scale = self.noise_multiplier * sens

        noise = torch.randn_like(tensor) * noise_scale
        privatized = tensor + noise

        self.num_queries += 1
        self.privacy_spent = self._compute_privacy_spent()

        return privatized

    def get_privacy_budget(self) -> dict[str, Any]:
        """Get current privacy budget status.

        Returns:
            Privacy budget info
        """
        budget_remaining = max(0, self.epsilon - self.privacy_spent)
        budget_fraction = min(1.0, self.privacy_spent / self.epsilon)

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "privacy_spent": round(self.privacy_spent, 4),
            "budget_remaining": round(budget_remaining, 4),
            "budget_fraction_used": round(budget_fraction, 4),
            "num_queries": self.num_queries,
            "sensitivity": self.sensitivity,
            "noise_multiplier": round(self.noise_multiplier, 4),
        }

    def reset_budget(self) -> None:
        """Reset privacy budget (for new training session)."""
        self.privacy_spent = 0.0
        self.num_queries = 0
        logger.info("Privacy budget reset")

    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted.

        Returns:
            True if budget exhausted
        """
        return self.privacy_spent >= self.epsilon

    def get_recommended_queries(self) -> int:
        """Get recommended max number of queries for current budget.

        Returns:
            Max recommended queries
        """
        epsilon_per_query = 1.0 / self.noise_multiplier
        max_queries = int(self.epsilon / epsilon_per_query)
        return max_queries

    def calibrate_noise(self, target_epsilon: float, num_queries: int) -> float:
        """Calibrate noise multiplier for target epsilon and query count.

        Args:
            target_epsilon: Target privacy budget
            num_queries: Expected number of queries

        Returns:
            Recommended noise multiplier
        """
        # ε_total = n * ε_single
        # ε_single = ε_total / n
        # σ = 1 / ε_single
        epsilon_per_query = target_epsilon / num_queries
        recommended_multiplier = 1.0 / epsilon_per_query

        logger.info(
            f"Recommended noise multiplier for ε={target_epsilon}, n={num_queries}: σ={recommended_multiplier:.4f}"
        )

        return recommended_multiplier

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DifferentialPrivacy(ε={self.epsilon}, δ={self.delta}, "
            f"sensitivity={self.sensitivity}, "
            f"spent={self.privacy_spent:.4f})"
        )
