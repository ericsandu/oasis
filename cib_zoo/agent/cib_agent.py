"""CIBAgent class inheriting OASIS SocialAgent with rate limiting and client perception."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from camel.models import BaseModelBackend
from camel.types import ModelType
from camel.utils import BaseTokenCounter
from oasis.environment.env_action import ManualAction
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType

from cib_zoo.agent.budget import BudgetLimiter, normalize_action_type
from cib_zoo.agent.perception import CIBEnvironment, PerceptionCache

logger = logging.getLogger("cib_zoo.agent")


class _NoOpTokenCounter(BaseTokenCounter):
    """Token counter that returns constant zero for non-LLM CIB agents."""

    def count_tokens_from_messages(self, messages: Any) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return []

    def decode(self, token_ids: list[int]) -> str:
        return ""


class NoOpModelBackend(BaseModelBackend):
    """Hermetic no-op model backend satisfying ChatAgent requirements without LLM calls."""

    def __init__(self) -> None:
        self._tc = _NoOpTokenCounter()
        super().__init__(
            model_type=ModelType.DEFAULT,
            token_counter=self._tc,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        return self._tc

    @property
    def token_limit(self) -> int:
        return 4096

    def preprocess_messages(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CIBAgent does not use LLM inference.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CIBAgent does not use LLM inference.")

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CIBAgent does not use LLM inference.")

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CIBAgent does not use LLM inference.")

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CIBAgent does not use LLM inference.")


class CIBAgent(SocialAgent):
    """Modular CIB Agent inheriting OASIS SocialAgent.

    Guarantees:
    - Zero backdoor SQLite queries during step decisions.
    - Strict enforcement of max_actions_per_step and total_budget.
    - Strict occurrence threshold quotas per ActionType.
    - In-memory perception caching for client actions (REFRESH, SEARCH_POSTS, SEARCH_USER).
    - Dynamic class-level bot registry eliminating hardcoded magic-number ID partitioning.
    """

    _registry: set[int] = set()

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        channel: Channel,
        max_actions_per_step: int = 1,
        total_budget: Optional[int] = None,
        occurrence_thresholds: Optional[dict[Union[ActionType, str], int]] = None,
        model: Any = None,
        cache_ttl_steps: Optional[int] = None,
        cache_ttl_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # If no model is supplied, provide NoOpModelBackend to satisfy ChatAgent hermetically
        effective_model = model if model is not None else NoOpModelBackend()

        super().__init__(
            agent_id=agent_id,
            user_info=user_info,
            channel=channel,
            model=effective_model,
            **kwargs,
        )

        # Register this bot instance
        CIBAgent._registry.add(agent_id)

        # Initialize CIB budget limiter
        self.budget: BudgetLimiter = BudgetLimiter(
            max_actions_per_step=max_actions_per_step,
            total_budget=total_budget,
            occurrence_thresholds=occurrence_thresholds,
        )

        # Initialize perception cache and safe client environment
        self.perception: PerceptionCache = PerceptionCache(
            ttl_steps=cache_ttl_steps,
            ttl_seconds=cache_ttl_seconds,
        )
        self.env = CIBEnvironment(
            action=SocialAction(agent_id, self.channel),
            perception=self.perception,
        )

    @classmethod
    def get_instance_count(cls) -> int:
        """Return the total number of registered CIB bot instances."""
        return len(cls._registry)

    @classmethod
    def get_bot_ids(cls) -> list[int]:
        """Return a sorted list of all registered CIB bot IDs."""
        return sorted(cls._registry)

    @classmethod
    def is_bot(cls, agent_id: int) -> bool:
        """Check if an agent ID belongs to a registered CIB bot instance."""
        return agent_id in cls._registry

    @classmethod
    def reset_registry(cls) -> None:
        """Reset the bot instance registry."""
        cls._registry.clear()

    def reset_step_budget(self) -> None:
        """Called at the beginning of each simulation step."""
        self.budget.reset_step_budget()
        self.perception.advance_step()

    def can_execute(self, action_type: Union[ActionType, str]) -> bool:
        """Check if action is allowed by step limit, total budget, and occurrence thresholds."""
        return self.budget.can_execute(action_type)

    def record_action(self, action_type: Union[ActionType, str]) -> None:
        """Record an executed action against budgets."""
        self.budget.record_action(action_type)

    def update_perception(
        self,
        action_type: Union[ActionType, str],
        data: dict[str, Any] | list[dict[str, Any]],
        query: Optional[str] = None,
    ) -> None:
        """Populate internal PerceptionCache with platform responses."""
        self.perception.update(action_type, data, query=query)

    def filter_actions(
        self,
        actions: list[ManualAction],
        emit_do_nothing_on_exhaust: bool = False,
    ) -> list[ManualAction]:
        """Clamp actions to remaining step budget, total budget, and occurrence thresholds."""
        return self.budget.filter_actions(
            actions, emit_do_nothing_on_exhaust=emit_do_nothing_on_exhaust
        )

    async def perform_action_by_data(
        self, func_name: Union[ActionType, str], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a manual action through SocialAction gated by budget limiter."""
        canonical = normalize_action_type(func_name)

        if not self.can_execute(canonical):
            logger.warning(
                f"CIBAgent {self.social_agent_id}: Action {canonical.value} rejected by budget limiter."
            )
            return {
                "success": False,
                "error": f"Budget limit reached for action {canonical.value}",
                "budget_exhausted": True,
            }

        function_list = self.env.action.get_openai_function_list()
        for item in function_list:
            if item.func.__name__ == canonical.value:
                func = item.func
                result = await func(*args, **kwargs)

                # Record action under budget
                self.record_action(canonical)

                # Update perception cache if action returns observable platform data
                if isinstance(result, dict) and result.get("success"):
                    query = kwargs.get("query")
                    self.update_perception(canonical, result, query=query)

                return result

        raise ValueError(f"Function {canonical.value} not found in SocialAction list.")

    async def perform_action_by_llm(self) -> Any:
        """CIB agents do not perform actions via LLM inference."""
        raise NotImplementedError(
            "CIBAgent uses deterministic/pattern action generation, not LLM inference."
        )
