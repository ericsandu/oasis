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
import math
import os
import random
import re
import time
from collections import defaultdict
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
from oasis.social_platform.database import fetch_table_from_db
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
    enable_organic_posting: bool = False
    organic_post_rate: float = 0.10
    bot_organic_post_rate: float = 0.15
    community_map: dict[int, str] = field(default_factory=dict)


@dataclass
class JEVStepResult:
    """Analytics and outcome records for a single JEV execution step.

    Attributes:
        step_index: Integer index of the executed step.
        total_evaluations: Total number of (agent, post) pairs evaluated in parallel.
        action_counts: Frequency distribution across reaction types {'L': ..., 'R': ..., 'C': ..., 'S': ...}.
        execution_time_seconds: Wall-clock duration of the step execution in seconds.
        scheduled_actions: Chronologically ordered list of ScheduledAction instances produced.
        num_organic_posts: Number of spontaneous root posts dispatched in this step.
    """

    step_index: int
    total_evaluations: int
    action_counts: dict[str, int] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    scheduled_actions: list[ScheduledAction] = field(default_factory=list)
    num_organic_posts: int = 0

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
    def num_posts(self) -> int:
        """Total number of spontaneous root posts emitted."""
        return self.num_organic_posts or self.action_counts.get("P", 0)

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
        if action_type == ActionType.CREATE_POST:
            stance = act.get("stance", act.get("post_stance", 0.0))
            if isinstance(message, dict):
                if "stance" not in message:
                    message["stance"] = stance
            elif isinstance(message, tuple):
                if len(message) == 1:
                    message = (message[0], stance)
            elif isinstance(message, str):
                message = (message, stance)
        elif action_type == ActionType.QUOTE_POST:
            stance = act.get("stance", act.get("post_stance", 0.0))
            if isinstance(message, tuple) and len(message) == 2:
                message = (message[0], message[1], stance)
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
        original_post_id = raw_post.get("original_post_id")
        if original_post_id is not None:
            try:
                original_post_id = int(original_post_id)
            except (ValueError, TypeError):
                original_post_id = None

        if not quote_content:
            quote_match = re.search(
                r"User \d+ quoted a post from User (\d+)\.\s*Quote content:\s*(.+?)\.\s*Original Content:\s*(.+)",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if quote_match:
                if not original_author:
                    original_author = f"user_{quote_match.group(1)}"
                quote_content = quote_match.group(2).strip()
                content = quote_match.group(3).strip()

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
            original_post_id=original_post_id,
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
    original_post_id = getattr(raw_post, "original_post_id", None)
    if original_post_id is not None:
        try:
            original_post_id = int(original_post_id)
        except (ValueError, TypeError):
            original_post_id = None

    if not quote_content:
        quote_match = re.search(
            r"User \d+ quoted a post from User (\d+)\.\s*Quote content:\s*(.+?)\.\s*Original Content:\s*(.+)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if quote_match:
            if not original_author:
                original_author = f"user_{quote_match.group(1)}"
            quote_content = quote_match.group(2).strip()
            content = quote_match.group(3).strip()

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
        original_post_id=original_post_id,
        num_likes=num_likes,
        num_shares=num_shares,
    )
    cache[post_id] = prefix
    return prefix


def _extract_post_stance(raw_post: Any) -> float:
    """Extract numeric stance score from post data if present.

    Extracts genuine post stance values from platform/database post records,
    supporting numeric stances, string-encoded numbers, qualitative labels,
    sqlite3.Row mapping objects, and nested metadata dictionaries. Clamps output to [-1.0, 1.0].
    Defaults to 0.0 only when no stance information is present.
    """
    if raw_post is None:
        return 0.0

    val = None

    # 1. Dictionary or Mapping (e.g. sqlite3.Row, dict, custom mapping)
    if hasattr(raw_post, "keys"):
        try:
            keys_set = set(raw_post.keys())
        except Exception:
            keys_set = set()

        for key in ("stance", "post_stance", "opinion_score", "sentiment_score"):
            if key in keys_set:
                try:
                    candidate = raw_post[key]
                    if candidate is not None:
                        val = candidate
                        break
                except Exception:
                    pass

        if val is None:
            # Check nested metadata, trace info, or post dictionary
            for sub_key in ("info", "metadata", "properties", "extra", "post"):
                if sub_key in keys_set:
                    try:
                        sub_dict = raw_post[sub_key]
                        if hasattr(sub_dict, "keys"):
                            sub_keys = set(sub_dict.keys())
                            for key in ("stance", "post_stance", "opinion_score"):
                                if key in sub_keys and sub_dict[key] is not None:
                                    val = sub_dict[key]
                                    break
                        elif isinstance(sub_dict, str) and "stance" in sub_dict:
                            import json

                            try:
                                parsed = json.loads(sub_dict)
                                if hasattr(parsed, "keys"):
                                    for key in ("stance", "post_stance"):
                                        if key in parsed and parsed[key] is not None:
                                            val = parsed[key]
                                            break
                            except Exception:
                                pass
                    except Exception:
                        pass
                if val is not None:
                    break

    # 2. Object attributes (dataclasses, namedtuples, ORM / SocialAgent post models)
    if val is None:
        for attr in ("stance", "post_stance", "opinion_score", "sentiment_score"):
            if hasattr(raw_post, attr):
                try:
                    attr_val = getattr(raw_post, attr)
                    if attr_val is not None:
                        val = attr_val
                        break
                except Exception:
                    pass

    # 3. Nested post attribute
    if val is None and hasattr(raw_post, "post"):
        try:
            nested = getattr(raw_post, "post")
            if nested is not None:
                val = _extract_post_stance(nested)
        except Exception:
            pass

    # 4. Content / body text parsing for explicit embedded stance metadata
    if val is None:
        content_text = ""
        if hasattr(raw_post, "get"):
            content_text = str(raw_post.get("content", "") or raw_post.get("text", ""))
        elif hasattr(raw_post, "content"):
            content_text = str(getattr(raw_post, "content", ""))
        elif hasattr(raw_post, "text"):
            content_text = str(getattr(raw_post, "text", ""))

        if content_text:
            m = re.search(
                r"\[STANCE\]:?\s*([+-]?\d+(?:\.\d+)?)", content_text, re.IGNORECASE
            )
            if not m:
                m = re.search(
                    r"\(Stance:\s*([+-]?\d+(?:\.\d+)?)\)", content_text, re.IGNORECASE
                )
            if not m:
                m = re.search(
                    r"\(Stance:\s*([a-zA-Z]+)\)", content_text, re.IGNORECASE
                )
            if m:
                val = m.group(1)

    # 5. Type normalization and numeric conversion
    if val is not None:
        if isinstance(val, bool):
            return 1.0 if val else -1.0
        if isinstance(val, (int, float)):
            try:
                num = float(val)
                if not math.isnan(num):
                    return max(-1.0, min(1.0, num))
            except (ValueError, TypeError):
                return 0.0
        if isinstance(val, str):
            v_clean = val.strip().lower()
            if v_clean in ("supportive", "support", "pro", "positive"):
                return 0.8
            elif v_clean in ("skeptical", "oppose", "against", "anti", "negative"):
                return -0.8
            elif v_clean in ("neutral", "undecided", "balanced"):
                return 0.0
            try:
                num = float(val)
                if not math.isnan(num):
                    return max(-1.0, min(1.0, num))
            except (ValueError, TypeError):
                pass

    return 0.0


def _extract_peer_engagement(raw_post: Any) -> int:
    """Extract peer engagement signal (likes, shares, reddit score) from post."""
    if hasattr(raw_post, "keys"):
        try:
            keys = set(raw_post.keys())
            likes = raw_post["num_likes"] if "num_likes" in keys else 0
            shares = raw_post["num_shares"] if "num_shares" in keys else 0
            score = raw_post["score"] if "score" in keys else 0
            return int(likes or 0) + int(shares or 0) + max(0, int(score or 0))
        except (ValueError, TypeError, KeyError):
            pass
    elif hasattr(raw_post, "get"):
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

    def _enrich_posts_with_stance(self, posts: list[Any]) -> list[Any]:
        """Enriches raw post dictionaries from feed with their real database stance if missing."""
        if not posts or self.platform is None or getattr(self.platform, "db_cursor", None) is None:
            return posts

        missing_pids = []
        for p in posts:
            if isinstance(p, dict) and "stance" not in p:
                pid = p.get("post_id")
                if pid is not None:
                    missing_pids.append(pid)

        if missing_pids:
            try:
                cursor = self.platform.db_cursor
                db_posts = fetch_table_from_db(cursor, "post")
                lookup = {
                    row["post_id"]: float(row.get("stance", 0.0) or 0.0)
                    for row in db_posts
                    if "post_id" in row
                }
                for p in posts:
                    if isinstance(p, dict) and "stance" not in p:
                        pid = p.get("post_id")
                        if pid in lookup:
                            p["stance"] = lookup[pid]
                            p["post_stance"] = lookup[pid]
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to enrich post stances from database: %s", e)

        return posts

    async def _retrieve_agent_feed(self, agent: Any) -> list[Any]:
        """Fetch personalized post feed for an individual agent."""
        user_id = _get_agent_id(agent)

        # 1. Direct query to platform recsys/feed
        if hasattr(self.platform, "refresh"):
            try:
                res = await self.platform.refresh(user_id)
                if isinstance(res, dict) and res.get("success"):
                    return self._enrich_posts_with_stance(list(res.get("posts", [])))
                elif isinstance(res, list):
                    return self._enrich_posts_with_stance(list(res))
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
                    return self._enrich_posts_with_stance(list(res.get("posts", [])))
                elif isinstance(res, list):
                    return self._enrich_posts_with_stance(list(res))
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
                    return self._enrich_posts_with_stance(list(feed))
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
            # Fire all chunks CONCURRENTLY: vLLM batches them server-side, so
            # the GPU stays saturated instead of idling between serial
            # round-trips (observed GPU KV-cache usage was ~2% with the old
            # serial await loop). Preserves ordering by gathering in order.
            chunks = [
                eval_items[i : i + self.config.batch_size]
                for i in range(0, len(eval_items), self.config.batch_size)
            ]
            chunk_results_list = await asyncio.gather(*[
                self.classifier_client.classify_batch(c, generate_comments=False)
                for c in chunks
            ])
            raw_results: list[ClassificationResult] = []
            for cr in chunk_results_list:
                raw_results.extend(cr)
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
                    persona_context = JEVPromptBuilder.build_agent_persona_context(
                        agent_suffix, post_prefix.topic
                    )
                    comment_requests.append((idx, persona_context, post_prefix.content))
            elif res.action_char == "Q" and not res.quote_text:
                ctx = item_context.get((res.user_id, res.post_id))
                if ctx:
                    post_prefix, agent_suffix, _, _ = ctx
                    persona_context = JEVPromptBuilder.build_agent_persona_context(
                        agent_suffix, post_prefix.topic
                    )
                    quote_requests.append((idx, persona_context, post_prefix.content))

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
            elif res.action_char == "F":
                # Follow is user-targeted: in the feed-reaction model the
                # agent follows the AUTHOR of the post it saw (paper §2.1).
                # Derive followee_id from the post's author user_id.
                followee_id = None
                if isinstance(raw_post, dict):
                    followee_id = (
                        raw_post.get("user_id")
                        or raw_post.get("author_id")
                        or raw_post.get("author")
                    )
                else:
                    followee_id = (
                        getattr(raw_post, "user_id", None)
                        or getattr(raw_post, "author_id", None)
                    )
                try:
                    followee_id = int(followee_id) if followee_id is not None else None
                except (ValueError, TypeError):
                    followee_id = None
                if followee_id is None or followee_id == res.user_id:
                    # No valid author target (or self-follow): downgrade to skip.
                    action_type = ActionType.DO_NOTHING
                    message = None
                    action_name = "do_nothing"
                else:
                    action_type = ActionType.FOLLOW
                    message = followee_id
                    action_name = "follow"
            else:  # "S"
                action_type = ActionType.DO_NOTHING
                message = None
                action_name = "do_nothing"

            post_stance = _extract_post_stance(raw_post)
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
                "stance": post_stance,
                "post_stance": post_stance,
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
                post_stance = act_dict.get("stance")
                if post_stance is None:
                    post_stance = _extract_post_stance(raw_post)
                else:
                    try:
                        post_stance = float(post_stance)
                    except (ValueError, TypeError):
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
            # Assign each action a distinct sub-step time in micro-time order so
            # reposts/likes don't all collapse to one created_at. We do NOT wait
            # for the platform between actions -- that would serialize JEV's
            # parallel dispatch (its whole point). The clock is bumped as the
            # (already micro-time-sorted) queue is written; under async the
            # platform may read a slightly later sub-step value for an early
            # action, but ordering is approximately preserved and the recsys
            # ranks by hot-score, not sub-step created_at, so this does not
            # change recommendations.
            _clock = getattr(self.platform, "sandbox_clock", None)
            _base_step = getattr(_clock, "time_step", None) if _clock else None
            _recv = getattr(self.channel, "receive_queue", None)

            def _advance_clock(item, idx, total):
                if _clock is None or _base_step is None:
                    return
                _clock.time_step = _base_step + (idx + 1) / (total + 1)

            await self.action_queue.drain_to_channel(
                self.channel, formatter=formatter, on_dispatch=_advance_clock)

            # Restore the integer step after sub-step draining.
            if _clock is not None and _base_step is not None:
                _clock.time_step = _base_step

            if self.config.wait_for_platform and _recv is not None:
                while not _recv.empty():
                    await asyncio.sleep(0.01)
                # Allow platform task to commit final action to SQLite
                await asyncio.sleep(0.05)

        # Advance sandbox clock step if available
        if hasattr(self.platform, "sandbox_clock") and hasattr(
            self.platform.sandbox_clock, "time_step"
        ):
            self.platform.sandbox_clock.time_step += 1

        # Stage j.2: Optional spontaneous organic posting track
        organic_posts: list[ScheduledAction] = []
        if self.config.enable_organic_posting:
            organic_posts = await self.step_organic_posts(
                step_index=step_index,
                candidate_agents=agents,
                effective_base_time=effective_base_time,
            )
            scheduled_actions.extend(organic_posts)

        # Stage k: Compile action distribution and return JEVStepResult
        action_counts = {"L": 0, "R": 0, "Q": 0, "C": 0, "S": 0, "P": 0}
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
            num_organic_posts=len(organic_posts),
        )

    async def step_organic_posts(
        self,
        step_index: int | None = None,
        candidate_agents: list[Any] | None = None,
        effective_base_time: datetime | str | float | None = None,
    ) -> list[ScheduledAction]:
        """Executes the separate spontaneous organic posting track.

        Runs AFTER the parallel feed evaluation and interaction pass is complete.
        Allows both organic agents and CIB bots to publish authentic root posts
        (ActionType.CREATE_POST) aligned with their persona, community topic, and stance.

        Args:
            step_index: Integer simulation step index.
            candidate_agents: Optional list of agent instances to consider for posting.
                              Defaults to all agents in self.agent_graph.
            effective_base_time: Base simulation start timestamp.

        Returns:
            List of ScheduledAction instances dispatched for CREATE_POST.
        """
        if step_index is None:
            step_index = self.step_index

        if effective_base_time is None:
            if self.config.base_time is not None:
                effective_base_time = self.config.base_time
            elif (
                hasattr(self.platform, "start_time")
                and self.platform.start_time is not None
            ):
                effective_base_time = self.platform.start_time
            else:
                effective_base_time = datetime.now(timezone.utc)

        # Collect candidate agents
        if candidate_agents is not None:
            agents = list(candidate_agents)
        else:
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
            agents = list(all_agents_map.values())

        if not agents:
            return []

        # Select agents to post based on organic post probability
        selected_candidates: list[tuple[int, Any, str, str, str, float]] = []
        for agent in agents:
            user_id = _get_agent_id(agent)
            act_freq = _get_agent_activity_frequency(agent)

            # Check if agent is a CIB bot
            is_bot = (
                hasattr(agent, "budget")
                or hasattr(agent, "budget_limiter")
                or getattr(agent, "is_bot", False)
                or type(agent).__name__ == "CIBAgent"
            )
            try:
                from cib_zoo.agent.cib_agent import CIBAgent

                if user_id in CIBAgent._registry or isinstance(agent, CIBAgent):
                    is_bot = True
            except ImportError:
                pass

            # Check bot budget limiter
            if is_bot:
                if hasattr(agent, "can_execute") and not agent.can_execute(ActionType.CREATE_POST):
                    continue
                elif hasattr(agent, "budget") and hasattr(agent.budget, "can_execute") and not agent.budget.can_execute(ActionType.CREATE_POST):
                    continue
                elif hasattr(agent, "budget_limiter") and hasattr(agent.budget_limiter, "can_execute") and not agent.budget_limiter.can_execute(ActionType.CREATE_POST):
                    continue

            post_rate = (
                self.config.bot_organic_post_rate
                if is_bot
                else self.config.organic_post_rate
            )
            eff_prob = min(1.0, post_rate * act_freq)
            if eff_prob <= 0.0 or random.random() > eff_prob:
                continue

            # Determine topic
            topic = self.config.community_map.get(user_id)
            if not topic:
                bio = getattr(getattr(agent, "user_info", None), "description", "") or ""
                bio_lower = bio.lower()
                if "tech" in bio_lower or "engineer" in bio_lower or "developer" in bio_lower:
                    topic = "tech"
                elif "sport" in bio_lower or "football" in bio_lower or "basketball" in bio_lower:
                    topic = "sports"
                elif "policy" in bio_lower or "politics" in bio_lower or "gov" in bio_lower:
                    topic = "politics"
                else:
                    topic = self.config.default_topic

            belief_state = self.get_belief_state(user_id)
            stance_label = belief_state.get_stance_label(topic)
            agent_suffix = _create_agent_suffix(agent, belief_state, topic)
            agent_context = JEVPromptBuilder.build_agent_persona_context(
                agent_suffix, topic
            )

            selected_candidates.append(
                (user_id, agent, agent_context, topic, stance_label, act_freq)
            )

        if not selected_candidates:
            return []

        # Batch generate post texts
        reqs = [
            (ctx, topic, stance)
            for _, _, ctx, topic, stance, _ in selected_candidates
        ]
        if hasattr(self.classifier_client, "generate_posts_batch"):
            post_texts = await self.classifier_client.generate_posts_batch(reqs)
        elif hasattr(self.classifier_client, "generate_post"):
            post_texts = [
                await self.classifier_client.generate_post(ctx, topic, stance)
                for ctx, topic, stance in reqs
            ]
        else:
            post_texts = [
                f"Sharing updates on #{topic} today."
                for _, _, _, topic, _, _ in selected_candidates
            ]

        # Schedule and dispatch CREATE_POST actions
        post_actions_to_schedule: list[tuple[int, dict[str, Any], float]] = []
        for (user_id, agent, _, topic, _, act_freq), content in zip(
            selected_candidates, post_texts
        ):
            b_state = self.get_belief_state(user_id)
            agent_stance = b_state.get_stance(topic)
            action_dict = {
                "action_type": ActionType.CREATE_POST,
                "action_char": "P",
                "action_name": "create_post",
                "message": (content, agent_stance),
                "content": content,
                "topic": topic,
                "stance": agent_stance,
                "post_stance": agent_stance,
                "post_id": 0,
                "raw_post": {
                    "content": content,
                    "topic": topic,
                    "stance": agent_stance,
                    "post_stance": agent_stance,
                },
            }
            if hasattr(agent, "record_action"):
                agent.record_action(ActionType.CREATE_POST)
            elif hasattr(agent, "budget") and hasattr(agent.budget, "record_action"):
                agent.budget.record_action(ActionType.CREATE_POST)
            elif hasattr(agent, "budget_limiter") and hasattr(agent.budget_limiter, "record_action"):
                agent.budget_limiter.record_action(ActionType.CREATE_POST)

            post_actions_to_schedule.append((user_id, action_dict, act_freq))

        scheduled_posts = self.scheduler.schedule_actions(
            step_index=step_index,
            base_time=effective_base_time,
            agent_actions=post_actions_to_schedule,
        )

        # Record in agent BeliefState
        for sa in scheduled_posts:
            uid = sa.user_id
            b_state = self.get_belief_state(uid)
            top = sa.action_dict.get("topic", self.config.default_topic)
            b_state.record_action(
                action_type="create_post",
                post_id=0,
                topic=top,
                timestamp_iso=sa.iso_timestamp,
            )

        # Enqueue and drain to OASIS Channel
        self.action_queue.push_batch(scheduled_posts)
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
                await asyncio.sleep(0.05)

        return scheduled_posts

    async def seed_initial_posts(
        self,
        agents: list[Any] | None = None,
        num_posts: int = 15,
        effective_base_time: datetime | str | float | None = None,
    ) -> list[ScheduledAction]:
        """Seeds initial organic background posts across communities at Step 0.

        Populates the platform feed and recommendation index before simulation steps
        begin, replacing empty feeds with realistic community discussions.

        Args:
            agents: Optional list of organic agents to author initial posts.
                    Defaults to all agents in self.agent_graph.
            num_posts: Target number of initial posts to generate (distributed across communities).
            effective_base_time: Base simulation start timestamp.

        Returns:
            List of ScheduledAction instances dispatched for CREATE_POST.
        """
        if num_posts <= 0:
            return []

        # Collect candidate agents
        if agents is None:
            all_agents_map = {}
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
            candidate_list = list(all_agents_map.values())
        else:
            candidate_list = list(agents)

        if not candidate_list:
            return []

        # Filter out bots so initial background inventory is authentic organic posts
        organic_candidates = []
        for a in candidate_list:
            uid = _get_agent_id(a)
            is_bot = (
                hasattr(a, "budget")
                or hasattr(a, "budget_limiter")
                or getattr(a, "is_bot", False)
                or type(a).__name__ == "CIBAgent"
            )
            try:
                from cib_zoo.agent.cib_agent import CIBAgent

                if isinstance(a, CIBAgent):
                    is_bot = True
            except ImportError:
                pass
            if not is_bot:
                organic_candidates.append(a)

        if not organic_candidates:
            organic_candidates = candidate_list

        # Group by community/topic to ensure balanced topic distribution
        by_topic: dict[str, list[Any]] = defaultdict(list)
        for a in organic_candidates:
            uid = _get_agent_id(a)
            topic = self.config.community_map.get(uid)
            if not topic:
                bio = getattr(getattr(a, "user_info", None), "description", "") or ""
                bio_lower = bio.lower()
                if "tech" in bio_lower or "engineer" in bio_lower or "developer" in bio_lower:
                    topic = "tech"
                elif "sport" in bio_lower or "football" in bio_lower or "basketball" in bio_lower:
                    topic = "sports"
                elif "policy" in bio_lower or "politics" in bio_lower or "gov" in bio_lower:
                    topic = "politics"
                else:
                    topic = self.config.default_topic
            by_topic[topic].append(a)

        # Sample agents evenly across topics up to num_posts
        topics = list(by_topic.keys())
        posts_per_topic = max(1, num_posts // len(topics)) if topics else num_posts
        selected_agents: list[tuple[Any, str]] = []
        for t in topics:
            agents_in_topic = by_topic[t]
            k = min(len(agents_in_topic), posts_per_topic)
            selected_agents.extend((a, t) for a in random.sample(agents_in_topic, k))

        # If we need more to reach num_posts, sample from remainder
        if len(selected_agents) < num_posts and len(organic_candidates) > len(selected_agents):
            remaining = [
                (
                    a,
                    self.config.community_map.get(
                        _get_agent_id(a), self.config.default_topic
                    ),
                )
                for a in organic_candidates
                if a not in [sa[0] for sa in selected_agents]
            ]
            needed = min(num_posts - len(selected_agents), len(remaining))
            selected_agents.extend(random.sample(remaining, needed))

        # Build requests
        reqs = []
        author_data = []
        for agent, topic in selected_agents:
            user_id = _get_agent_id(agent)
            belief_state = self.get_belief_state(user_id)
            stance_label = belief_state.get_stance_label(topic)
            agent_suffix = _create_agent_suffix(agent, belief_state, topic)
            agent_context = JEVPromptBuilder.build_agent_persona_context(
                agent_suffix, topic
            )
            reqs.append((agent_context, topic, stance_label))
            author_data.append((user_id, agent, topic))

        if not reqs:
            return []

        # Batch generate post texts
        if hasattr(self.classifier_client, "generate_posts_batch"):
            post_texts = await self.classifier_client.generate_posts_batch(reqs)
        elif hasattr(self.classifier_client, "generate_post"):
            post_texts = [
                await self.classifier_client.generate_post(ctx, topic, stance)
                for ctx, topic, stance in reqs
            ]
        else:
            post_texts = [
                f"Sharing updates on #{topic} today." for _, topic, _ in reqs
            ]

        # Dispatch via channel
        base_time = (
            effective_base_time
            or self.config.base_time
            or datetime.now(timezone.utc)
        )
        post_actions_to_schedule = []
        for (user_id, agent, topic), content in zip(author_data, post_texts):
            action_dict = {
                "action_type": ActionType.CREATE_POST,
                "action_char": "P",
                "action_name": "create_post",
                "message": content,
                "content": content,
                "topic": topic,
                "post_id": 0,
            }
            post_actions_to_schedule.append((user_id, action_dict, 1.0))

        scheduled_posts = self.scheduler.schedule_actions(
            step_index=0,
            base_time=base_time,
            agent_actions=post_actions_to_schedule,
        )

        for sa in scheduled_posts:
            uid = sa.user_id
            b_state = self.get_belief_state(uid)
            top = sa.action_dict.get("topic", self.config.default_topic)
            b_state.record_action(
                action_type="create_post",
                post_id=0,
                topic=top,
                timestamp_iso=sa.iso_timestamp,
            )

        self.action_queue.push_batch(scheduled_posts)
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
                await asyncio.sleep(0.05)

        return scheduled_posts

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
