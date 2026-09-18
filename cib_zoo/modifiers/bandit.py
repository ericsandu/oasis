"""Thompson Sampling multi-armed bandit modifier for dynamic policy tuning."""

from __future__ import annotations

import random
from typing import Any, Optional

from cib_zoo.modifiers.base import BaseModifier


class ThompsonSamplingBanditModifier(BaseModifier):
    """Adaptive modifier optimizing attack arm selection via Thompson Sampling with Beta posteriors.

    Each arm k maintains Beta(alpha_k, beta_k). At each decision point,
    theta_k ~ Beta(alpha_k, beta_k) is sampled, and the arm maximizing theta_k is chosen.
    When environmental feedback arrives (e.g. recommendation rank, organic reach),
    the chosen arm updates its posterior.
    """

    def __init__(
        self,
        arms: list[str],
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if not arms:
            raise ValueError("arms list cannot be empty.")
        if len(set(arms)) != len(arms):
            raise ValueError("arms must contain unique identifiers.")
        if alpha_init <= 0 or beta_init <= 0:
            raise ValueError("alpha_init and beta_init must be strictly positive.")

        self.arms = list(arms)
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        self._rng = random.Random(random_state)

        self.alphas: dict[str, float] = {arm: self.alpha_init for arm in self.arms}
        self.betas: dict[str, float] = {arm: self.beta_init for arm in self.arms}
        self.pull_counts: dict[str, int] = {arm: 0 for arm in self.arms}
        self.total_rewards: dict[str, float] = {arm: 0.0 for arm in self.arms}

    def sample_arm(self) -> str:
        """Draw samples from each arm's posterior and select the arm with maximum sample."""
        best_arm = self.arms[0]
        max_sample = -1.0

        for arm in self.arms:
            sample = self._rng.betavariate(self.alphas[arm], self.betas[arm])
            if sample > max_sample:
                max_sample = sample
                best_arm = arm

        return best_arm

    def update(self, arm: str, reward: float) -> None:
        """Update Beta posterior for the specified arm given reward in [0, 1].

        Args:
            arm: The selected arm identifier.
            reward: Environmental feedback scalar, clamped to [0.0, 1.0].
        """
        if arm not in self.alphas:
            raise KeyError(f"Unknown arm '{arm}'. Available arms: {self.arms}")

        clamped_reward = max(0.0, min(1.0, float(reward)))
        self.alphas[arm] += clamped_reward
        self.betas[arm] += (1.0 - clamped_reward)
        self.pull_counts[arm] += 1
        self.total_rewards[arm] += clamped_reward

    def get_expected_values(self) -> dict[str, float]:
        """Calculate mean expected reward for each arm: alpha / (alpha + beta)."""
        return {
            arm: self.alphas[arm] / (self.alphas[arm] + self.betas[arm])
            for arm in self.arms
        }

    def get_arm_stats(self) -> dict[str, dict[str, Any]]:
        """Return comprehensive statistical state for all arms."""
        stats = {}
        for arm in self.arms:
            alpha = self.alphas[arm]
            beta = self.betas[arm]
            stats[arm] = {
                "alpha": alpha,
                "beta": beta,
                "expected_value": alpha / (alpha + beta),
                "pulls": self.pull_counts[arm],
                "total_reward": self.total_rewards[arm],
            }
        return stats
