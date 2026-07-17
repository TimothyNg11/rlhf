"""Reward/verifier module public API."""

from grpo_math.rewards.extraction import extract_boxed
from grpo_math.rewards.verify import RewardResult, compute_reward, verify_answer

__all__ = ["extract_boxed", "verify_answer", "compute_reward", "RewardResult"]
