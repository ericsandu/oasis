# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
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
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""Belief State and 3-Level Memory Engine for OASIS JEV Architecture.

This module implements Track 2 of the JEV optimization plan:
- Level 2: Compact episodic action log window (bounded rolling history).
- Level 3: Dynamic per-topic stance tracking (s_i in [-1.0, 1.0]) with
  bounded confidence persuasion dynamics (Deffuant / Hegselmann-Krause hybrid).

Zero SQLite dependencies: this module is purely in-memory and computational.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# Stance label constants adhering to JEV Section 8.2
LABEL_STRONGLY_SUPPORTIVE: str = "Strongly Supportive"
LABEL_SUPPORTIVE: str = "Supportive"
LABEL_NEUTRAL: str = "Neutral / Undecided"
LABEL_SKEPTICAL: str = "Skeptical"
LABEL_STRONGLY_OPPOSED: str = "Strongly Opposed"

# Standard action verb normalization mapping
_ACTION_VERB_MAP: dict[str, str] = {
    "like": "Liked",
    "like_post": "Liked",
    "liked": "Liked",
    "l": "Liked",
    "repost": "Reposted",
    "reposted": "Reposted",
    "r": "Reposted",
    "comment": "Commented on",
    "create_comment": "Commented on",
    "commented": "Commented on",
    "commented on": "Commented on",
    "c": "Commented on",
    "dislike": "Disliked",
    "dislike_post": "Disliked",
    "disliked": "Disliked",
    "quote": "Quoted",
    "quote_post": "Quoted",
    "quoted": "Quoted",
    "skip": "Skipped",
    "s": "Skipped",
    "skipped": "Skipped",
    "follow": "Followed",
    "followed": "Followed",
    "unfollow": "Unfollowed",
    "unfollowed": "Unfollowed",
    "create_post": "Posted",
    "post": "Posted",
    "posted": "Posted",
}


def _sigmoid(x: float) -> float:
    """Compute numerically stable sigmoid function 1 / (1 + exp(-x))."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _format_action_verb(action_type: Any) -> str:
    """Normalize raw action string or enum into past-tense display verb."""
    if hasattr(action_type, "value"):
        raw = str(action_type.value)
    else:
        raw = str(action_type)

    cleaned = raw.strip().lower()
    if cleaned in _ACTION_VERB_MAP:
        return _ACTION_VERB_MAP[cleaned]

    # Heuristic fallback for custom or unmapped actions
    if cleaned.endswith("ed"):
        return cleaned.capitalize()
    if cleaned.endswith("e"):
        return cleaned.capitalize() + "d"
    return cleaned.capitalize() + "ed"


@dataclass
class ActionLogItem:
    """Represents a single episodic action record in Level 2 memory.

    Attributes:
        action_type: Action type identifier (e.g., 'like', 'repost', 'comment',
            or ActionType enum value).
        post_id: Target post identifier.
        topic: Topic hashtag or classification label (e.g. 'tech', '#politics').
        timestamp_iso: Timestamp of the action in ISO-8601 UTC format.
    """

    action_type: str
    post_id: int
    topic: str
    timestamp_iso: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize action item to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionLogItem:
        """Instantiate action item from dictionary."""
        return cls(
            action_type=str(data["action_type"]),
            post_id=int(data["post_id"]),
            topic=str(data.get("topic", "")),
            timestamp_iso=str(data["timestamp_iso"]),
        )


@dataclass
class BeliefState:
    """In-memory belief state and 3-level cognitive dynamics engine for an agent.

    Maintains:
    - Level 2 Memory: Bounded rolling window of recent actions.
    - Level 3 Memory: Dynamic stance vector s_i(topic) in [-1.0, 1.0] with
      bounded confidence persuasion updates.

    Attributes:
        user_id: Unique identifier for the agent/user.
        stances: Mapping of topic name to stance scalar s_i in [-1.0, +1.0].
        recent_actions: Chronological list of recent episodic actions (max N).
        max_history_items: Maximum capacity of episodic rolling action window.
        persuasion_rate_alpha: Base persuasion susceptibility parameter alpha.
        delta_max: Maximum stance jump threshold delta_max in bounded confidence model.
    """

    user_id: int
    # topic -> stance scalar in [-1.0, +1.0]
    stances: dict[str, float] = field(default_factory=dict)
    # Compact episodic log: max N items
    recent_actions: list[ActionLogItem] = field(default_factory=list)
    max_history_items: int = 5
    persuasion_rate_alpha: float = 0.15
    delta_max: float = 1.0

    def __post_init__(self) -> None:
        """Enforce domain invariants upon initialization."""
        # Ensure all initial stance scalars are strictly clamped to [-1.0, 1.0]
        self.stances = {
            topic: max(-1.0, min(1.0, float(val)))
            for topic, val in self.stances.items()
        }
        # Enforce bounded rolling window capacity
        if self.max_history_items <= 0:
            self.recent_actions.clear()
        elif len(self.recent_actions) > self.max_history_items:
            self.recent_actions = self.recent_actions[-self.max_history_items:]

    def get_stance(self, topic: str) -> float:
        """Retrieve current stance scalar s_i for a topic.

        Args:
            topic: Topic identifier string.

        Returns:
            Stance scalar s_i in [-1.0, 1.0]. Defaults to 0.0 (neutral) if unobserved.
        """
        return float(self.stances.get(topic, 0.0))

    def set_stance(self, topic: str, stance: float) -> float:
        """Set stance scalar s_i directly for a topic, clamped to [-1.0, 1.0].

        Args:
            topic: Topic identifier string.
            stance: Stance scalar value to set.

        Returns:
            The clamped stance scalar stored in state.
        """
        clamped = max(-1.0, min(1.0, float(stance)))
        self.stances[topic] = clamped
        return clamped

    @staticmethod
    def label_from_score(score: float) -> str:
        """Map a numerical stance score s_i to a qualitative label.

        Interval partitioning:
            s_i > 0.3          -> "Strongly Supportive"
            0.1 < s_i <= 0.3   -> "Supportive"
            -0.1 <= s_i <= 0.1 -> "Neutral / Undecided"
            -0.3 <= s_i < -0.1 -> "Skeptical"
            s_i < -0.3         -> "Strongly Opposed"

        Args:
            score: Numeric stance scalar.

        Returns:
            Qualitative label string.
        """
        if score > 0.3:
            return LABEL_STRONGLY_SUPPORTIVE
        if score > 0.1:
            return LABEL_SUPPORTIVE
        if score >= -0.1:
            return LABEL_NEUTRAL
        if score >= -0.3:
            return LABEL_SKEPTICAL
        return LABEL_STRONGLY_OPPOSED

    def get_stance_label(self, topic: str) -> str:
        """Get the qualitative stance label for the specified topic.

        Args:
            topic: Topic identifier string.

        Returns:
            One of "Strongly Supportive", "Supportive", "Neutral / Undecided",
            "Skeptical", "Strongly Opposed".
        """
        score = self.get_stance(topic)
        return self.label_from_score(score)

    def update_stance(
        self,
        topic: str,
        post_stance: float,
        peer_engagement: int,
        delta_max: float | None = None,
        omega_peer: float | None = None,
    ) -> float:
        r"""Adjusts the agent's stance scalar for a topic using bounded confidence dynamics.

        Mathematical formulation:
            \Delta = s_{post} - s_i
            \Delta s_i = \alpha \cdot \text{sign}(\Delta) \cdot \min(|\Delta|, \delta_{\max}) \cdot \omega_{peer}
            s_i^{(t+1)} = \text{clip}(s_i + \Delta s_i, -1.0, 1.0)

        where:
            - s_i is current stance on topic (defaults to 0.0).
            - \alpha is self.persuasion_rate_alpha.
            - \delta_{\max} is bounded confidence jump cap (defaults to self.delta_max).
            - \omega_{peer} is social proof weight derived from peer engagement:
              \omega_{peer} = \text{sigmoid}(peer\_engagement) if not explicitly provided.

        Args:
            topic: Topic identifier string.
            post_stance: Stance scalar of the encountered post in [-1.0, 1.0].
            peer_engagement: Social proof signal (e.g. number of likes/reposts >= 0).
            delta_max: Optional override for maximum step size bound.
            omega_peer: Optional override for social proof weight factor.

        Returns:
            Updated stance scalar s_i^{(t+1)} in [-1.0, 1.0].
        """
        # Current agent stance (default 0.0)
        s_current = self.get_stance(topic)

        # Clamped post stance
        s_post = max(-1.0, min(1.0, float(post_stance)))

        # Stance gap Delta
        delta = s_post - s_current

        if delta == 0.0:
            return s_current

        # Direction of pull
        sign_delta = 1.0 if delta > 0.0 else -1.0

        # Effective step bound delta_max
        effective_delta_max = (
            float(delta_max) if delta_max is not None else self.delta_max
        )
        effective_delta_max = max(0.0, effective_delta_max)

        # Social proof weight omega_peer
        if omega_peer is not None:
            effective_omega_peer = float(omega_peer)
        else:
            effective_omega_peer = _sigmoid(float(peer_engagement))

        # Bounded confidence update calculation
        bounded_step = min(abs(delta), effective_delta_max)
        delta_s = (
            self.persuasion_rate_alpha
            * sign_delta
            * bounded_step
            * effective_omega_peer
        )

        # Apply update and clip strictly within [-1.0, 1.0]
        s_new = max(-1.0, min(1.0, s_current + delta_s))
        self.stances[topic] = s_new
        return s_new

    def record_action(
        self,
        action_type: Any,
        post_id: int,
        topic: str,
        timestamp_iso: str,
    ) -> None:
        """Record an executed action into the Level 2 episodic action log.

        Maintains a bounded rolling window of size `max_history_items`. When the
        log exceeds capacity, older items roll off automatically (FIFO).

        Args:
            action_type: Action descriptor (e.g. 'like', 'comment', ActionType).
            post_id: Target post identifier.
            topic: Associated topic hashtag or label (e.g. 'tech', '#target').
            timestamp_iso: ISO-8601 UTC timestamp of execution.
        """
        raw_action = (
            str(action_type.value)
            if hasattr(action_type, "value")
            else str(action_type)
        )
        item = ActionLogItem(
            action_type=raw_action,
            post_id=int(post_id),
            topic=str(topic).strip() if topic is not None else "",
            timestamp_iso=str(timestamp_iso),
        )

        if self.max_history_items <= 0:
            self.recent_actions.clear()
            return

        self.recent_actions.append(item)
        if len(self.recent_actions) > self.max_history_items:
            self.recent_actions = self.recent_actions[-self.max_history_items:]

    def get_episodic_summary(self) -> str:
        """Generate a compact single-line summary of recent actions for Level 2 prompt suffix.

        Produces a standardized, token-efficient string representation suitable
        for injection into JEVPromptBuilder agent suffixes:
            "Recent: Liked P12 (#tech), Liked P42 (#target), Commented on P42"

        Returns:
            Single-line summary string, or empty string "" if no actions are recorded.
        """
        if not self.recent_actions:
            return ""

        parts: list[str] = []
        for item in self.recent_actions:
            verb = _format_action_verb(item.action_type)
            topic_str = item.topic.strip()
            if topic_str:
                tag = topic_str if topic_str.startswith("#") else f"#{topic_str}"
                parts.append(f"{verb} P{item.post_id} ({tag})")
            else:
                parts.append(f"{verb} P{item.post_id}")

        return "Recent: " + ", ".join(parts)

    def clear_history(self) -> None:
        """Clear all Level 2 episodic action items."""
        self.recent_actions.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize belief state to a dictionary."""
        return {
            "user_id": self.user_id,
            "stances": dict(self.stances),
            "recent_actions": [a.to_dict() for a in self.recent_actions],
            "max_history_items": self.max_history_items,
            "persuasion_rate_alpha": self.persuasion_rate_alpha,
            "delta_max": self.delta_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefState:
        """Instantiate belief state from a dictionary."""
        actions_raw = data.get("recent_actions", [])
        actions = [ActionLogItem.from_dict(a) for a in actions_raw]
        return cls(
            user_id=int(data["user_id"]),
            stances={k: float(v) for k, v in data.get("stances", {}).items()},
            recent_actions=actions,
            max_history_items=int(data.get("max_history_items", 5)),
            persuasion_rate_alpha=float(data.get("persuasion_rate_alpha", 0.15)),
            delta_max=float(data.get("delta_max", 1.0)),
        )
