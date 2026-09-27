# =========== Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. ===========
"""JEV Execution Loop and Integrated Simulation Environment (Track 5).

Connects:
- Track 1 (JEVPromptBuilder): Inverted static post prefix KV-caching.
- Track 2 (BeliefState): Dynamic stance persuasion & compact episodic memory.
- Track 3 (JEVClassifierClient & resolve_intra_feed_budget): 1-token logit classification,
  intra-feed budget filtering, and secondary comment generation.
- Track 4 (MicroTimeScheduler & ChronologicalActionQueue): Micro-time Poisson arrivals
  and chronologically ordered queue draining to OASIS Channel.

Adheres strictly to:
- Zero direct sqlite3 imports inside agent decision logic paths.
- Capped CPU thread execution to max 4 (OMP_NUM_THREADS=4).
- Complete backward compatibility with OasisEnv.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
)

# Strictly enforce resource caps to max 4 threads
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(var, "4")

from oasis.clock.micro_time_scheduler import (
    ChronologicalActionQueue,
    MicroTimeScheduler,
    ScheduledAction,
)
from oasis.environment.env import OasisEnv
from oasis.environment.env_action import LLMAction, ManualAction
from oasis.inference.jev_classifier import (
    ClassificationResult,
    EvalItem,
    JEVClassifierClient,
    MockJEVClassifierClient,
    resolve_intra_feed_budget,
)
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.belief_state import BeliefState
from oasis.social_agent.jev_prompt_builder import (
    AgentSuffixData,
    JEVPromptBuilder,
    PostPrefixData,
)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (
    ActionType,
    DefaultPlatformType,
    RecsysType,
)

logger = logging.getLogger(__name__)


@dataclass
class JEVExecutionConfig:
    """Configuration options for JEV (Joint Evaluation Vectorization) execution loop.

    Attributes:
        batch_size: Number of item evaluations to chunk per forward pass (0 or negative for unbounded).
        step_duration_seconds: Virtual continuous duration of each simulation step in seconds (default: 900.0).
        default_lambda: Baseline arrival rate parameter lambda_0 for Poisson micro-time scheduling.
        enable_belief_updates: Whether to apply bounded confidence stance dynamics on exposed posts.
        max_actions_per_agent: Intra-feed budget constraint for non-skip actions per agent (default: 1).
        classifier_client: JEVClassifierClient implementation (e.g. MockJEVClassifierClient or VLLMJEVClassifierClient).
        default_topic: Fallback topic when a post does not specify a hashtag or category.
        seed: PRNG seed for deterministic micro-time offset and simulation reproducibility.
        downgrade_to_skip: When True, excess actions beyond budget are converted to 'S' (Skip).
        send_skips_to_platform: When True, skip actions are dispatched to platform as DO_NOTHING.
        record_skips_in_memory: When True, skip actions are recorded in agent Level 2 episodic action log.
        update_recsys: Whether to invoke platform.update_rec_table() at the start of each step.
        wait_for_platform: Whether to wait for platform channel receive_queue to drain before step ends.
        channel_formatter: Optional custom callable to convert ScheduledAction into channel payload.
        base_time: Optional base simulation start time fallback.
        stance_persuasion_alpha: Stance shift sensitivity multiplier alpha.
        stance_delta_max: Bounded confidence maximum step limit delta_max.
    """

    batch_size: int = 64
    step_duration_seconds: float = 900.0
    default_lambda: float = 1.0
    enable_belief_updates: bool = True
    max_actions_per_agent: int = 1
    classifier_client: JEVClassifierClient | None = None
    default_topic: str = "general"
    seed: int | None = None
    downgrade_to_skip: bool = True
    send_skips_to_platform: bool = False
    record_skips_in_memory: bool = False
    update_recsys: bool = True
    wait_for_platform: bool = False
    channel_formatter: Callable[[ScheduledAction], Any] | None = None
    base_time: datetime | str | float | None = None
    stance_persuasion_alpha: float = 0.15
    stance_delta_max: float = 1.0


@dataclass
class JEVStepResult:
    """Analytics and outcome records for a single JEV execution step.

    Attributes:
        step_index: Integer index of the executed step.
        total_evaluations: Total number of (agent, post) pairs evaluated in parallel.
        action_counts: Frequency distribution across reaction types {'L': ..., 'R': ..., 'C': ..., 'S': ...}.
        execution_time_seconds: Wall-clock duration of the step execution in seconds.
        scheduled_actions: Chronologically ordered list of ScheduledAction instances produced.
    """

    step_index: int
    total_evaluations: int
    action_counts: dict[str, int] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    scheduled_actions: list[ScheduledAction] = field(default_factory=list)

    @property
    def num_likes(self) -> int:
        """Total number of like actions emitted."""
        return self.action_counts.get("L", 0)

    @property
    def num_reposts(self) -> int:
        """Total number of repost actions emitted."""
        return self.action_counts.get("R", 0)

    @property
    def num_quotes(self) -> int:
        """Total number of quote actions emitted."""
        return self.action_counts.get("Q", 0)

    @property
    def num_comments(self) -> int:
        """Total number of comment actions emitted."""
        return self.action_counts.get("C", 0)

    @property
    def num_skips(self) -> int:
        """Total number of skip actions emitted."""
        return self.action_counts.get("S", 0)

    @property
    def num_actions(self) -> int:
        """Total number of active (non-skip) actions emitted."""
        return self.num_likes + self.num_reposts + self.num_quotes + self.num_comments

    def to_dict(self) -> dict[str, Any]:
        """Convert step result to dictionary."""
        return {
            "step_index": self.step_index,
            "total_evaluations": self.total_evaluations,
            "action_counts": dict(self.action_counts),
            "execution_time_seconds": self.execution_time_seconds,
            "num_actions": self.num_actions,
            "num_likes": self.num_likes,
            "num_reposts": self.num_reposts,
            "num_quotes": self.num_quotes,
            "num_comments": self.num_comments,
            "num_skips": self.num_skips,
            "scheduled_actions": [sa.to_dict() for sa in self.scheduled_actions],
        }


def default_oasis_channel_formatter(item: ScheduledAction) -> tuple[int, Any, Any]:
    """Format ScheduledAction into standard 3-tuple expected by OASIS Platform.running().

    Yields (agent_id, message_payload, action_type).
    """
    act = item.action_dict
    if isinstance(act, dict):
        action_type = act.get("action_type", ActionType.DO_NOTHING)
        message = act.get("message", None)
        return (item.user_id, message, action_type)
    return (item.user_id, act, ActionType.DO_NOTHING)


def _get_agent_id(agent: Any) -> int:
    """Safely extract integer identifier from agent object."""
    if hasattr(agent, "social_agent_id"):
        return int(agent.social_agent_id)
    if hasattr(agent, "agent_id"):
        return int(agent.agent_id)
    if hasattr(agent, "user_id"):
        return int(agent.user_id)
    if isinstance(agent, (int, float)):
        return int(agent)
    return 0


def _get_agent_activity_frequency(agent: Any) -> float:
    """Extract activity frequency multiplier from agent or user profile."""
    if hasattr(agent, "activity_frequency"):
        return float(agent.activity_frequency)
    if hasattr(agent, "activity_level_frequency"):
        freq = agent.activity_level_frequency
        return float(freq[0] if isinstance(freq, (list, tuple)) else freq)
    if hasattr(agent, "user_info"):
        info = agent.user_info
        if hasattr(info, "activity_frequency"):
            return float(info.activity_frequency)
        if hasattr(info, "activity_level_frequency"):
            freq = info.activity_level_frequency
            return float(freq[0] if isinstance(freq, (list, tuple)) else freq)
        if isinstance(getattr(info, "profile", None), dict):
            other = info.profile.get("other_info", {})
            if isinstance(other, dict) and "activity_level_frequency" in other:
                freq = other["activity_level_frequency"]
                return float(freq[0] if isinstance(freq, (list, tuple)) else freq)
    return 1.0


def _create_agent_suffix(
    agent: Any,
    belief_state: BeliefState,
    topic: str,
) -> AgentSuffixData:
    """Construct AgentSuffixData from agent profile and dynamic belief state."""
    user_id = _get_agent_id(agent)
    user_name = f"user_{user_id}"
    mbti = "INTJ"
    country = "US"
    bio = f"User profile for user_{user_id}"

    user_info = getattr(agent, "user_info", None)
    if user_info is not None:
        if getattr(user_info, "user_name", None):
            user_name = str(user_info.user_name)
        elif getattr(user_info, "name", None):
            user_name = str(user_info.name)

        if getattr(user_info, "description", None):
            bio = str(user_info.description)
        elif getattr(user_info, "name", None):
            bio = str(user_info.name)

        profile = getattr(user_info, "profile", None)
        if isinstance(profile, dict):
            other = profile.get("other_info", {})
            if isinstance(other, dict):
                if other.get("mbti"):
                    mbti = str(other["mbti"])
                if other.get("country"):
                    country = str(other["country"])
                if other.get("user_profile"):
                    bio = str(other["user_profile"])

    if hasattr(agent, "mbti") and agent.mbti:
        mbti = str(agent.mbti)
    if hasattr(agent, "country") and agent.country:
        country = str(agent.country)

    stance_score = belief_state.get_stance(topic)
    stance_label = belief_state.get_stance_label(topic)
    recent_actions = belief_state.get_episodic_summary()

    return AgentSuffixData(
        user_id=user_id,
        user_name=user_name,
        mbti=mbti,
        country=country,
        bio=bio,
        stance_label=stance_label,
        stance_score=stance_score,
        recent_actions=recent_actions,
        topic=topic,
    )


def _get_or_create_post_prefix(
    raw_post: Any,
    cache: dict[int, PostPrefixData],
    default_topic: str = "general",
) -> PostPrefixData:
    """Extract or cache PostPrefixData ensuring prefix KV-cache reuse."""
    if isinstance(raw_post, PostPrefixData):
        cache[raw_post.post_id] = raw_post
        return raw_post

    if isinstance(raw_post, dict):
        post_id = int(raw_post.get("post_id", raw_post.get("id", 0)))
        if post_id in cache:
            return cache[post_id]

        author_name = str(
            raw_post.get("author_name")
            or raw_post.get("user_name")
            or raw_post.get("author")
            or f"user_{raw_post.get('user_id', 0)}"
        )
        content = str(raw_post.get("content") or raw_post.get("text", ""))
        quote_content = raw_post.get("quote_content")
        original_author = raw_post.get("original_author")
        num_likes = int(raw_post.get("num_likes", 0) or 0)
        num_shares = int(raw_post.get("num_shares", 0) or 0)

        topic = raw_post.get("topic") or raw_post.get("hashtag")
        if not topic:
            tags = re.findall(r"#(\w+)", content)
            topic = tags[0] if tags else default_topic
        topic = str(topic)

        prefix = PostPrefixData(
            post_id=post_id,
            author_name=author_name,
            topic=topic,
            content=content,
            quote_content=str(quote_content) if quote_content else None,
            original_author=str(original_author) if original_author else None,
            num_likes=num_likes,
            num_shares=num_shares,
        )
        cache[post_id] = prefix
        return prefix

    post_id = int(getattr(raw_post, "post_id", getattr(raw_post, "id", 0)))
    if post_id in cache:
        return cache[post_id]

    author_name = str(
        getattr(raw_post, "author_name", None)
        or getattr(raw_post, "user_name", None)
        or getattr(raw_post, "author", None)
        or f"user_{getattr(raw_post, 'user_id', 0)}"
    )
    content = str(getattr(raw_post, "content", getattr(raw_post, "text", "")))
    quote_content = getattr(raw_post, "quote_content", None)
    original_author = getattr(raw_post, "original_author", None)
    num_likes = int(getattr(raw_post, "num_likes", 0) or 0)
    num_shares = int(getattr(raw_post, "num_shares", 0) or 0)

    topic = getattr(raw_post, "topic", None)
    if not topic:
        tags = re.findall(r"#(\w+)", content)
        topic = tags[0] if tags else default_topic
    topic = str(topic)

    prefix = PostPrefixData(
        post_id=post_id,
        author_name=author_name,
        topic=topic,
        content=content,
        quote_content=str(quote_content) if quote_content else None,
        original_author=str(original_author) if original_author else None,
        num_likes=num_likes,
        num_shares=num_shares,
    )
    cache[post_id] = prefix
    return prefix


def _extract_post_stance(raw_post: Any) -> float:
    """Extract numeric stance score from post data if present."""
    if isinstance(raw_post, dict):
        val = raw_post.get("stance", raw_post.get("post_stance", 0.0))
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    val = getattr(raw_post, "stance", getattr(raw_post, "post_stance", 0.0))
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _extract_peer_engagement(raw_post: Any) -> int:
    """Extract peer engagement signal (likes, shares, reddit score) from post."""
    if isinstance(raw_post, dict):
        likes = raw_post.get("num_likes", 0) or 0
        shares = raw_post.get("num_shares", 0) or 0
        score = raw_post.get("score", 0) or 0
        try:
            return int(likes) + int(shares) + max(0, int(score))
        except (ValueError, TypeError):
            return 0
    likes = getattr(raw_post, "num_likes", 0) or 0
    shares = getattr(raw_post, "num_shares", 0) or 0
    score = getattr(raw_post, "score", 0) or 0
    try:
        return int(likes) + int(shares) + max(0, int(score))
    except (ValueError, TypeError):
        return 0


class JEVEnvironment(OasisEnv):
    """High-throughput Joint Evaluation Vectorization (JEV) simulation environment.

    Subclasses OasisEnv to provide seamless backward compatibility while unlocking
    15x inference acceleration and 70-90% KV-cache reuse:
    - Inverted post-prefix KV-caching prompts.
    - 1-token logit classification with intra-feed budget resolution.
    - Dynamic Stance & 3-Level opinion memory tracking.
    - Continuous Poisson micro-time scheduling and chronological queue execution.

    Args:
        env_or_graph: Existing OasisEnv instance or AgentGraph to wrap/subclass.
        platform: Platform instance or DefaultPlatformType enum.
        database_path: SQLite DB path for platform storage.
        config: JEVExecutionConfig options.
        classifier_client: JEVClassifierClient implementation.
        belief_states: Optional pre-existing dictionary of BeliefState instances.
        semaphore: Async semaphore limit for concurrent operations.
    """

    def __init__(
        self,
        env_or_graph: OasisEnv | AgentGraph | Any,
        platform: DefaultPlatformType | Platform | None = None,
        database_path: str | None = None,
        config: JEVExecutionConfig | None = None,
        classifier_client: JEVClassifierClient | None = None,
        belief_states: dict[int, BeliefState] | None = None,
        semaphore: int = 128,
        **kwargs: Any,
    ) -> None:
        if isinstance(env_or_graph, OasisEnv):
            self._oasis_env = env_or_graph
            self.agent_graph = env_or_graph.agent_graph
            self.platform = env_or_graph.platform
            self.platform_type = getattr(
                env_or_graph, "platform_type", DefaultPlatformType.TWITTER
            )
            self.channel = getattr(env_or_graph, "channel", None) or getattr(
                env_or_graph.platform, "channel", None
            )
            self.database_path = env_or_graph.database_path
            self.llm_semaphore = env_or_graph.llm_semaphore
            self.platform_task = getattr(env_or_graph, "platform_task", None)
        else:
            self._oasis_env = None
            self.agent_graph = env_or_graph
            self.llm_semaphore = asyncio.Semaphore(semaphore)
            self.platform_task = None

            if isinstance(platform, Platform):
                self.platform = platform
                self.channel = platform.channel
                self.database_path = database_path or getattr(
                    platform, "db_path", None
                )
                self.platform_type = (
                    DefaultPlatformType.REDDIT
                    if getattr(platform, "recsys_type", None) == RecsysType.REDDIT
                    else DefaultPlatformType.TWITTER
                )
            elif isinstance(platform, DefaultPlatformType):
                db = database_path or ":memory:"
                super().__init__(
                    agent_graph=env_or_graph,
                    platform=platform,
                    database_path=db,
                    semaphore=semaphore,
                )
            else:
                self.platform = None
                self.channel = Channel()
                self.database_path = database_path or ":memory:"
                self.platform_type = DefaultPlatformType.TWITTER

        self.config: JEVExecutionConfig = config or JEVExecutionConfig()

        if classifier_client is not None:
            self.classifier_client: JEVClassifierClient = classifier_client
        elif self.config.classifier_client is not None:
            self.classifier_client = self.config.classifier_client
        else:
            self.classifier_client = MockJEVClassifierClient()

        self.scheduler: MicroTimeScheduler = MicroTimeScheduler(
            step_duration_seconds=self.config.step_duration_seconds,
            default_lambda=self.config.default_lambda,
            seed=self.config.seed,
        )
        self.action_queue: ChronologicalActionQueue = ChronologicalActionQueue()
        self.belief_states: dict[int, BeliefState] = dict(belief_states or {})
        self.step_index: int = 0

        self._initialize_belief_states()

    def _initialize_belief_states(self) -> None:
        """Initialize BeliefState instances for all agents in agent_graph if not present."""
        if not self.agent_graph:
            return
        agents = []
        if hasattr(self.agent_graph, "agent_mappings"):
            agents = list(self.agent_graph.agent_mappings.values())
        elif hasattr(self.agent_graph, "get_agents"):
            raw = self.agent_graph.get_agents()
            agents = [a[1] if isinstance(a, tuple) else a for a in raw]

        for agent in agents:
            aid = _get_agent_id(agent)
            if aid not in self.belief_states:
                self.belief_states[aid] = BeliefState(
                    user_id=aid,
                    persuasion_rate_alpha=self.config.stance_persuasion_alpha,
                    delta_max=self.config.stance_delta_max,
                )

    def get_belief_state(self, user_id: int) -> BeliefState:
        """Retrieve the BeliefState for a user, initializing with defaults if missing."""
        if user_id not in self.belief_states:
            self.belief_states[user_id] = BeliefState(
                user_id=user_id,
                persuasion_rate_alpha=self.config.stance_persuasion_alpha,
                delta_max=self.config.stance_delta_max,
            )
        return self.belief_states[user_id]

    def set_belief_state(self, user_id: int, state: BeliefState) -> None:
        """Assign or override BeliefState for a user."""
        self.belief_states[user_id] = state

    async def _retrieve_agent_feed(self, agent: Any) -> list[Any]:
        """Fetch personalized post feed for an individual agent."""
        user_id = _get_agent_id(agent)

        # 1. Direct query to platform recsys/feed
        if hasattr(self.platform, "refresh"):
            try:
                res = await self.platform.refresh(user_id)
                if isinstance(res, dict) and res.get("success"):
                    return list(res.get("posts", []))
                elif isinstance(res, list):
                    return list(res)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"platform.refresh({user_id}) failed: {e}")

        # 2. Agent social environment action fallback
        if (
            hasattr(agent, "env")
            and hasattr(agent.env, "action")
            and hasattr(agent.env.action, "refresh")
        ):
            try:
                res = await agent.env.action.refresh()
                if isinstance(res, dict) and res.get("success"):
                    return list(res.get("posts", []))
                elif isinstance(res, list):
                    return list(res)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"agent.env.action.refresh() failed for {user_id}: {e}")

        # 3. Agent custom feed hook
        if hasattr(agent, "get_feed"):
            try:
                feed = (
                    await agent.get_feed()
                    if asyncio.iscoroutinefunction(agent.get_feed)
                    else agent.get_feed()
                )
                if isinstance(feed, list):
                    return list(feed)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"agent.get_feed() failed for {user_id}: {e}")

        return []

    async def step_jev(
        self,
        step_index: int | None = None,
        base_time: datetime | str | float | None = None,
        agent_feeds: dict[int, list[Any]] | None = None,
        active_agent_ids: Sequence[int] | None = None,
    ) -> JEVStepResult:
        """Execute one simulation step using JEV (Joint Evaluation Vectorization) pipeline.

        Execution stages:
        a. Retrieve personalized feeds per active agent.
        b. Construct PostPrefixData (shared KV cache) and AgentSuffixData.
        c. Assemble item-level EvalItem batch via JEVPromptBuilder.
        d. Batch-classify with 1-token logit biasing.
        e. Resolve intra-feed budget per agent.
        f. Trigger comment worker for 'C' actions.
        g. Revise agent dynamic BeliefState (stance scalar & episodic action log).
        h. Format actions into OASIS action payloads.
        i. Micro-time continuous Poisson scheduling.
        j. Enqueue into ChronologicalActionQueue and drain to OASIS Channel.
        k. Return JEVStepResult analytics.

        Args:
            step_index: Integer simulation step index (auto-increments if None).
            base_time: Base simulation start timestamp.
            agent_feeds: Optional pre-constructed personalized feeds mapping user_id -> posts.
            active_agent_ids: Optional subset of agent IDs to evaluate.

        Returns:
            JEVStepResult containing evaluation totals, action distribution, and scheduled items.
        """
        start_time = time.perf_counter()

        if step_index is None:
            step_index = self.step_index
            self.step_index += 1
        else:
            self.step_index = max(self.step_index, step_index + 1)

        # Determine effective base timestamp
        if base_time is not None:
            effective_base_time = base_time
        elif self.config.base_time is not None:
            effective_base_time = self.config.base_time
        elif (
            hasattr(self.platform, "start_time")
            and self.platform.start_time is not None
        ):
            effective_base_time = self.platform.start_time
        else:
            effective_base_time = datetime.now(timezone.utc)

        # Stage 0: Recommendation system refresh if enabled
        if self.config.update_recsys and hasattr(self.platform, "update_rec_table"):
            try:
                await self.platform.update_rec_table()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to refresh platform rec table: {e}")

        # Collect active agents from agent graph
        all_agents_map: dict[int, Any] = {}
        if hasattr(self.agent_graph, "agent_mappings"):
            all_agents_map = dict(self.agent_graph.agent_mappings)
        elif hasattr(self.agent_graph, "get_agents"):
            raw = self.agent_graph.get_agents()
            for item in raw:
                if isinstance(item, tuple):
                    all_agents_map[item[0]] = item[1]
                elif hasattr(item, "social_agent_id"):
                    all_agents_map[item.social_agent_id] = item
                elif hasattr(item, "agent_id"):
                    all_agents_map[item.agent_id] = item

        if active_agent_ids is not None:
            active_set = set(active_agent_ids)
            agents = [a for aid, a in all_agents_map.items() if aid in active_set]
        else:
            agents = list(all_agents_map.values())

        if not agents:
            exec_time = time.perf_counter() - start_time
            return JEVStepResult(
                step_index=step_index,
                total_evaluations=0,
                action_counts={"L": 0, "R": 0, "Q": 0, "C": 0, "S": 0},
                execution_time_seconds=exec_time,
                scheduled_actions=[],
            )

        # Stage a: Retrieve personalized feeds
        if agent_feeds is not None:
            feeds: list[list[Any]] = [
                agent_feeds.get(_get_agent_id(agent), []) for agent in agents
            ]
        else:
            feed_tasks = [self._retrieve_agent_feed(agent) for agent in agents]
            feeds = await asyncio.gather(*feed_tasks)

        # Stage b & c: Construct inverted prompts & assemble EvalItems
        prefix_cache: dict[int, PostPrefixData] = {}
        eval_items: list[EvalItem] = []
        item_context: dict[
            tuple[int, int], tuple[PostPrefixData, AgentSuffixData, Any, float]
        ] = {}

        for agent, feed in zip(agents, feeds):
            user_id = _get_agent_id(agent)
            belief_state = self.get_belief_state(user_id)
            act_freq = _get_agent_activity_frequency(agent)

            for raw_post in feed:
                post_prefix = _get_or_create_post_prefix(
                    raw_post, prefix_cache, self.config.default_topic
                )
                agent_suffix = _create_agent_suffix(
                    agent, belief_state, post_prefix.topic
                )
                full_prompt = JEVPromptBuilder.assemble_eval_prompt(
                    post_prefix, agent_suffix
                )

                eval_item = EvalItem(
                    user_id=user_id,
                    post_id=post_prefix.post_id,
                    topic=post_prefix.topic,
                    full_prompt=full_prompt,
                    post_content=post_prefix.content,
                )
                eval_items.append(eval_item)
                item_context[(user_id, post_prefix.post_id)] = (
                    post_prefix,
                    agent_suffix,
                    raw_post,
                    act_freq,
                )

        if not eval_items:
            exec_time = time.perf_counter() - start_time
            return JEVStepResult(
                step_index=step_index,
                total_evaluations=0,
                action_counts={"L": 0, "R": 0, "Q": 0, "C": 0, "S": 0},
                execution_time_seconds=exec_time,
                scheduled_actions=[],
            )

        # Stage d: Batch classification with 1-token logit biasing
        total_evals = len(eval_items)
        if self.config.batch_size > 0 and len(eval_items) > self.config.batch_size:
            raw_results: list[ClassificationResult] = []
            for i in range(0, len(eval_items), self.config.batch_size):
                chunk = eval_items[i : i + self.config.batch_size]
                chunk_results = await self.classifier_client.classify_batch(
                    chunk, generate_comments=False
                )
                raw_results.extend(chunk_results)
        else:
            raw_results = await self.classifier_client.classify_batch(
                eval_items, generate_comments=False
            )

        # Stage e: Resolve intra-feed budget constraints
        resolved_results = resolve_intra_feed_budget(
            raw_results,
            budget=self.config.max_actions_per_agent,
            downgrade_to_skip=self.config.downgrade_to_skip,
        )

        # Stage f: Conditional comment and quote generation fallback for surviving 'C' and 'Q' actions
        comment_requests: list[tuple[int, str, str]] = []
        quote_requests: list[tuple[int, str, str]] = []
        for idx, res in enumerate(resolved_results):
            if res.action_char == "C" and not res.comment_text:
                ctx = item_context.get((res.user_id, res.post_id))
                if ctx:
                    post_prefix, agent_suffix, _, _ = ctx
                    prompt = JEVPromptBuilder.assemble_eval_prompt(
                        post_prefix, agent_suffix
                    )
                    comment_requests.append((idx, prompt, post_prefix.content))
            elif res.action_char == "Q" and not res.quote_text:
                ctx = item_context.get((res.user_id, res.post_id))
                if ctx:
                    post_prefix, agent_suffix, _, _ = ctx
                    prompt = JEVPromptBuilder.assemble_eval_prompt(
                        post_prefix, agent_suffix
                    )
                    quote_requests.append((idx, prompt, post_prefix.content))

        if comment_requests:
            req_pairs = [(r[1], r[2]) for r in comment_requests]
            if hasattr(self.classifier_client, "generate_comments_batch"):
                generated_texts = (
                    await self.classifier_client.generate_comments_batch(req_pairs)
                )
            elif hasattr(self.classifier_client, "generate_comment"):
                generated_texts = [
                    await self.classifier_client.generate_comment(p, c)
                    for p, c in req_pairs
                ]
            else:
                generated_texts = ["Interesting post."] * len(req_pairs)
            for (idx, _, _), comment_text in zip(comment_requests, generated_texts):
                resolved_results[idx].comment_text = comment_text

        if quote_requests:
            q_req_pairs = [(r[1], r[2]) for r in quote_requests]
            if hasattr(self.classifier_client, "generate_quotes_batch"):
                generated_quotes = (
                    await self.classifier_client.generate_quotes_batch(q_req_pairs)
                )
            elif hasattr(self.classifier_client, "generate_quote"):
                generated_quotes = [
                    await self.classifier_client.generate_quote(p, c)
                    for p, c in q_req_pairs
                ]
            else:
                generated_quotes = ["Thought-provoking post."] * len(q_req_pairs)
            for (idx, _, _), quote_text in zip(quote_requests, generated_quotes):
                resolved_results[idx].quote_text = quote_text

        # Stage h & i: Format action payloads and schedule continuous arrivals
        agent_actions_to_schedule: list[tuple[int, dict[str, Any], float]] = []
        for res in resolved_results:
            ctx = item_context.get((res.user_id, res.post_id))
            post_prefix, _, raw_post, act_freq = (
                ctx if ctx else (None, None, {}, 1.0)
            )
            topic = post_prefix.topic if post_prefix else self.config.default_topic

            if res.action_char == "L":
                action_type = ActionType.LIKE_POST
                message: Any = res.post_id
                action_name = "like_post"
            elif res.action_char == "R":
                action_type = ActionType.REPOST
                message = res.post_id
                action_name = "repost"
            elif res.action_char == "Q":
                action_type = ActionType.QUOTE_POST
                message = (res.post_id, res.quote_text or "")
                action_name = "quote_post"
            elif res.action_char == "C":
                action_type = ActionType.CREATE_COMMENT
                message = (res.post_id, res.comment_text or "")
                action_name = "create_comment"
            else:  # "S"
                action_type = ActionType.DO_NOTHING
                message = None
                action_name = "do_nothing"

            action_dict = {
                "action_type": action_type,
                "action_char": res.action_char,
                "action_name": action_name,
                "post_id": res.post_id,
                "message": message,
                "comment_text": res.comment_text,
                "quote_text": res.quote_text,
                "confidence": res.confidence,
                "topic": topic,
                "raw_post": raw_post,
            }
            agent_actions_to_schedule.append((res.user_id, action_dict, act_freq))

        scheduled_actions = self.scheduler.schedule_actions(
            step_index=step_index,
            base_time=effective_base_time,
            agent_actions=agent_actions_to_schedule,
        )

        # Stage g: Revise agent BeliefState (stances & episodic memory)
        for scheduled in scheduled_actions:
            uid = scheduled.user_id
            act_dict = scheduled.action_dict
            belief_state = self.get_belief_state(uid)
            topic = act_dict.get("topic", self.config.default_topic)
            raw_post = act_dict.get("raw_post", {})

            if self.config.enable_belief_updates:
                post_stance = _extract_post_stance(raw_post)
                peer_engagement = _extract_peer_engagement(raw_post)
                belief_state.update_stance(
                    topic=topic,
                    post_stance=post_stance,
                    peer_engagement=peer_engagement,
                    delta_max=self.config.stance_delta_max,
                )

            act_char = act_dict.get("action_char", "S")
            if act_char != "S" or self.config.record_skips_in_memory:
                belief_state.record_action(
                    action_type=act_dict.get("action_type", act_char),
                    post_id=act_dict.get("post_id", 0),
                    topic=topic,
                    timestamp_iso=scheduled.iso_timestamp,
                )

        # Stage j: Enqueue into ChronologicalActionQueue and drain to OASIS Channel
        to_dispatch = [
            sa
            for sa in scheduled_actions
            if self.config.send_skips_to_platform
            or sa.action_dict.get("action_char") != "S"
        ]
        self.action_queue.push_batch(to_dispatch)

        if self.channel is not None and not self.action_queue.empty():
            formatter = (
                self.config.channel_formatter or default_oasis_channel_formatter
            )
            await self.action_queue.drain_to_channel(self.channel, formatter=formatter)

            if self.config.wait_for_platform and hasattr(
                self.channel, "receive_queue"
            ):
                while not self.channel.receive_queue.empty():
                    await asyncio.sleep(0.01)

        # Advance sandbox clock step if available
        if hasattr(self.platform, "sandbox_clock") and hasattr(
            self.platform.sandbox_clock, "time_step"
        ):
            self.platform.sandbox_clock.time_step += 1

        # Stage k: Compile action distribution and return JEVStepResult
        action_counts = {"L": 0, "R": 0, "Q": 0, "C": 0, "S": 0}
        for sa in scheduled_actions:
            c = sa.action_dict.get("action_char", "S")
            action_counts[c] = action_counts.get(c, 0) + 1

        exec_time = time.perf_counter() - start_time
        return JEVStepResult(
            step_index=step_index,
            total_evaluations=total_evals,
            action_counts=action_counts,
            execution_time_seconds=exec_time,
            scheduled_actions=scheduled_actions,
        )

    async def step(
        self,
        actions: dict[
            SocialAgent, ManualAction | LLMAction | list[ManualAction | LLMAction]
        ]
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a step in either CAMEL action mode or JEV mode.

        If actions dictionary is provided, executes standard CAMEL step logic.
        If actions is None, executes JEV mode step (step_jev).
        """
        if actions is not None:
            if self._oasis_env is not None:
                return await self._oasis_env.step(actions)
            return await super().step(actions)

        return await self.step_jev(**kwargs)

    async def reset(self) -> None:
        """Reset the environment, starting platform loop and initializing agents."""
        if self._oasis_env is not None:
            await self._oasis_env.reset()
            self.agent_graph = self._oasis_env.agent_graph
            self.platform_task = self._oasis_env.platform_task
        else:
            await super().reset()
        self._initialize_belief_states()

    async def close(self) -> None:
        """Close the environment and stop background platform worker."""
        if self._oasis_env is not None:
            await self._oasis_env.close()
        else:
            await super().close()
