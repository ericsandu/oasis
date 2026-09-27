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
"""JEV Inverted Prompt Engine.

Implements inverted prefix KV-caching prompt serialization for large-scale social simulations.
By structuring prompts with the shared post content as the static prefix and the personalized
user persona as the suffix, vLLM and SGLang RadixAttention compute the post attention once and
reuse it across all recipient agents evaluating that post.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

# Enforce resource guardrails
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


@dataclass(frozen=True)
class PostPrefixData:
    """Standardized post metadata and body for static prompt prefix caching.

    Attributes:
        post_id: Unique integer identifier for the post.
        author_name: Handle or username of the post creator (e.g. 'alice' or '@alice').
        topic: Topic category or hashtag (e.g. 'tech' or '#tech').
        content: Textual content body of the post.
        quote_content: Optional commentary text if this post is a quote post.
        original_author: Optional handle of the original author being quoted.
        original_post_id: Optional ID of the original post being quoted.
        num_likes: Count of likes on the post for social proof signals.
        num_shares: Count of reposts/quotes on the post for social proof signals.
    """

    post_id: int
    author_name: str
    topic: str
    content: str
    quote_content: str | None = None
    original_author: str | None = None
    original_post_id: int | None = None
    num_likes: int = 0
    num_shares: int = 0


@dataclass(frozen=True)
class AgentSuffixData:
    """Agent persona and dynamic stance state formatted as the personalized suffix.

    Attributes:
        user_id: Unique integer identifier for the evaluating agent.
        user_name: Username or handle of the evaluating agent.
        mbti: Psychological archetype or MBTI trait (e.g. 'INTJ').
        country: Geographic location or nationality (e.g. 'US').
        bio: Short biography describing the agent's background/persona.
        stance_label: Qualitative stance descriptor (e.g. 'Supportive', 'Neutral', 'Skeptical').
        stance_score: Quantitative stance scalar in [-1.0, +1.0].
        recent_actions: Optional compact summary of recent episodic interactions (Level 2 memory).
        topic: Optional topic context for stance display override.
    """

    user_id: int
    user_name: str
    mbti: str
    country: str
    bio: str
    stance_label: str
    stance_score: float
    recent_actions: str = ""
    topic: str | None = None


class JEVPromptBuilder:
    """Joint Evaluation Vectorization (JEV) prompt constructor.

    Inverts conventional LLM social agent prompts into [Shared Post Prefix] + [Personalized Agent Suffix]
    to maximize RadixAttention KV-cache prefix hits across multiple agents evaluating identical posts.
    """

    VALID_ACTIONS: Sequence[str] = ("L", "R", "Q", "C", "S")

    @staticmethod
    def build_post_prefix(post: PostPrefixData) -> str:
        """Constructs a standardized static prompt prefix for a post.

        Ensures byte-for-byte identical output for any PostPrefixData with identical attributes,
        enabling 70-90% KV-cache reuse across agents in vLLM / SGLang.

        For quote posts, places the original post first so that it matches the
        already-cached prefix of the original post in the RadixAttention trie,
        followed by the quote commentary extension.

        Args:
            post: The post prefix data structure.

        Returns:
            Standardized prefix string ending with double newline separator.
        """
        author_clean = post.author_name.strip().lstrip("@")
        topic_clean = post.topic.strip().lstrip("#")
        content_clean = post.content.strip()

        metrics_part = ""
        if post.num_likes > 0 or post.num_shares > 0:
            metrics_part = f" | Likes: {post.num_likes} | Reposts: {post.num_shares}"

        if post.quote_content and post.quote_content.strip():
            quote_clean = post.quote_content.strip()
            orig_author = (post.original_author or "user").strip().lstrip("@")
            orig_id_str = (
                f"{post.original_post_id}"
                if post.original_post_id is not None
                else "orig"
            )
            orig_part = (
                f"[POST ID: {orig_id_str}] Author: @{orig_author} | Topic: #{topic_clean}\n"
                f'Content: "{content_clean}"\n\n'
            )
            quote_part = (
                f"[QUOTE POST ID: {post.post_id}] Author: @{author_clean} | Topic: #{topic_clean}{metrics_part}\n"
                f'Quote Commentary: "{quote_clean}"\n\n'
            )
            return orig_part + quote_part

        return (
            f"[POST ID: {post.post_id}] Author: @{author_clean} | Topic: #{topic_clean}{metrics_part}\n"
            f'Content: "{content_clean}"\n\n'
        )

    @classmethod
    def build_inverted_prefix(cls, post: PostPrefixData) -> str:
        """Alias for build_post_prefix matching specification terminology."""
        return cls.build_post_prefix(post)

    @staticmethod
    def build_agent_suffix(
        agent: AgentSuffixData,
        topic: str | None = None,
    ) -> str:
        """Constructs the personalized agent suffix with persona, stance, and 1-token reaction task.

        Args:
            agent: The evaluating agent's persona, stance, and memory state.
            topic: Optional post topic to contextualize the stance label line.
                   Falls back to agent.topic if not explicitly supplied.

        Returns:
            Personalized suffix string terminating in 'Action: '.
        """
        user_clean = agent.user_name.strip().lstrip("@")
        mbti_clean = agent.mbti.strip()
        country_clean = agent.country.strip()
        bio_clean = agent.bio.strip()

        # Contextualize topic prefix for stance line if available
        effective_topic = topic or agent.topic
        topic_prefix = ""
        if effective_topic:
            t_clean = effective_topic.strip().lstrip("#")
            if t_clean:
                topic_prefix = f"#{t_clean}: "

        lines = [
            f"[OBSERVER]: @{user_clean} | Traits: {mbti_clean}, {country_clean} | Bio: {bio_clean}",
            f"[STANCE]: {topic_prefix}{agent.stance_label.strip()} ({agent.stance_score:+.2f})",
        ]

        if agent.recent_actions and agent.recent_actions.strip():
            recent_clean = agent.recent_actions.strip()
            if recent_clean.lower().startswith("recent:"):
                recent_clean = recent_clean[7:].strip()
            lines.append(f"[RECENT ACTIONS]: {recent_clean}")

        lines.append("[TASK]: Choose single reaction: [L]ike, [R]epost, [Q]uote, [C]omment, [S]kip.")
        lines.append("Action: ")

        return "\n".join(lines)

    @classmethod
    def assemble_eval_prompt(
        cls,
        post: PostPrefixData,
        agent: AgentSuffixData,
    ) -> str:
        """Concatenates post prefix and agent suffix into a single item evaluation prompt.

        Guarantees:
        1. prompt.startswith(cls.build_post_prefix(post)) is True.
        2. The prefix slice is byte-for-byte identical across any number of agents evaluating post.

        Args:
            post: Shared post prefix data.
            agent: Personalized agent suffix data.

        Returns:
            Complete assembled prompt ready for batched model forward pass.
        """
        prefix = cls.build_post_prefix(post)
        suffix = cls.build_agent_suffix(agent, topic=post.topic)
        return f"{prefix}{suffix}"

    @classmethod
    def parse_action_char(cls, raw_response: str) -> str:
        """Parses and validates a single reaction character ('L', 'R', 'C', 'S') from model output.

        Handles common variations such as single tokens ('L'), leading whitespace (' L'),
        action prefixes ('Action: L'), brackets ('[L]'), or full action names ('Like').
        Defaults to 'S' (Skip) if unparseable or neutral.

        Args:
            raw_response: Raw completion string emitted by LLM.

        Returns:
            One of 'L', 'R', 'C', 'S'.
        """
        if not raw_response or not raw_response.strip():
            return "S"

        text = raw_response.strip()

        # Handle 'Action: ...' or '[Action]: ...'
        if ":" in text:
            prefix_part, after_colon = text.split(":", 1)
            if "action" in prefix_part.lower():
                text = after_colon.strip()

        if not text:
            return "S"

        # Check bracketed token: [L], [R], [Q], [C], [S]
        bracket_match = re.search(r"\[([LRQCS])\]", text, re.IGNORECASE)
        if bracket_match:
            return bracket_match.group(1).upper()

        # Check immediate first character
        first_char = text[0].upper()
        if first_char in cls.VALID_ACTIONS:
            return first_char

        # Check full word matches
        upper_text = text.upper()
        if "LIKE" in upper_text:
            return "L"
        if "QUOTE" in upper_text:
            return "Q"
        if "REPOST" in upper_text:
            return "R"
        if "COMMENT" in upper_text:
            return "C"
        if "SKIP" in upper_text:
            return "S"

        # Check word boundary regex for isolated action characters
        isolated_match = re.search(r"\b([LRQCS])\b", upper_text)
        if isolated_match:
            return isolated_match.group(1).upper()

        return "S"
