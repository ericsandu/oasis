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

    VALID_ACTIONS: Sequence[str] = ("L", "R", "Q", "C", "F", "S")

    # Canonical action-char -> (tool name, human description) used to render the
    # action menu. Mirrors the align driver's char_by_name and the FunctionTool
    # names base OASIS exposes, so the JEV prompt lists the SAME actions a
    # classic agent would be given for the same `available_actions`.
    ACTION_DESCRIPTIONS: dict = {
        "L": ("like_post", "like the post"),
        "R": ("repost", "repost the post"),
        "Q": ("quote_post", "quote the post with your own commentary"),
        "C": ("create_comment", "comment on the post"),
        "F": ("follow", "follow the post's author"),
        "S": ("do_nothing", "do nothing"),
    }

    # Base-OASIS agent prompt, VERBATIM. Upstream carries the anti-"just like"
    # steer in TWO places, both reproduced here so JEV sees the same framing a
    # classic agent does (base OASIS itself steers away from defaulting to
    # `like` -- evidently the researchers hit the same over-like tendency the
    # per-post classifier does):
    #   1. the user message (oasis/social_agent/agent.py, perform_action_by_llm)
    #   2. the env_template trailing line (oasis/social_agent/agent_environment.py)
    # The classic path relies on TOOL SCHEMAS to name the actions; JEV has no
    # tool schemas, so build_task_instruction() also enumerates the ENABLED
    # actions (from available_actions), never a hardcoded set.
    OASIS_PROMPT_STEM: str = (
        "Please perform social media actions after observing the platform "
        "environments. Notice that don't limit your actions for example to "
        "just like the posts."
    )
    # Verbatim trailing steer from agent_environment.py env_template.
    OASIS_ENV_STEER: str = (
        "Pick one action you want to perform that best reflects your current "
        "inclination based on your profile and the post content. Do not limit "
        "your action in just `like` to like posts."
    )

    @classmethod
    def build_task_instruction(cls, allowed_chars: Sequence[str] | None = None) -> str:
        """Render the run-constant instruction preamble from the ENABLED actions.

        Not a hardcoded menu: the action list is derived from `allowed_chars`
        (the same set that drives the logit-bias), so a run that enables only
        [like_post, repost, follow, do_nothing] advertises exactly L/R/F/S and
        never mentions Quote/Comment the classifier cannot emit. The wording
        reproduces base OASIS's own prompt lines (OASIS_PROMPT_STEM +
        OASIS_ENV_STEER, both carrying upstream's anti-"just like" steer) for 1:1
        parity, then lists the enabled single-letter actions. Byte-identical for
        a given `allowed_chars`, so it still caches at the RadixAttention root.
        """
        chars = [c for c in (allowed_chars or cls.VALID_ACTIONS)
                 if c in cls.ACTION_DESCRIPTIONS]
        if "S" not in chars:
            chars.append("S")  # a no-op choice must always be available
        menu = ", ".join(
            f"{c} ({cls.ACTION_DESCRIPTIONS[c][1]})" for c in chars
        )
        letters = "/".join(chars)
        return (
            f"{cls.OASIS_PROMPT_STEM}\n"
            f"{cls.OASIS_ENV_STEER}\n"
            f"For the post below, choose exactly ONE action and output ONLY its "
            f"letter ({letters}):\n"
            f"{menu}.\n\n"
        )

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
    def build_agent_persona_context(
        agent: AgentSuffixData,
        topic: str | None = None,
    ) -> str:
        """Constructs clean persona and stance context without 1-token reaction task instructions.

        Used for secondary text generation (comments, quotes, spontaneous posts) to prevent
        instruction conflict between the 1-token reaction task and generative response tasks.

        Args:
            agent: The agent's persona, stance, and memory state.
            topic: Optional topic to contextualize the stance label line.
                   Falls back to agent.topic if not explicitly supplied.

        Returns:
            Clean persona context string with observer traits, stance, and recent actions.
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

        return "\n".join(lines)

    @classmethod
    def build_agent_suffix(
        cls,
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
        persona_context = cls.build_agent_persona_context(agent, topic=topic)
        # NOTE: the task instruction + action menu live in the GLOBAL cached
        # preamble (build_task_instruction(allowed_chars), prepended in
        # assemble_eval_prompt), NOT here -- it is byte-identical for every agent
        # and post in a run, so it belongs at the trie root where it caches once.
        # This suffix is the ONLY per-agent-varying part; keep it minimal
        # (persona + the 'Action: ' cue). Output stays a single letter.
        return (
            f"{persona_context}\n"
            "Action: "
        )

    @classmethod
    def assemble_eval_prompt(
        cls,
        post: PostPrefixData,
        agent: AgentSuffixData,
        allowed_chars: Sequence[str] | None = None,
    ) -> str:
        """Concatenates instruction + post prefix + agent suffix into one eval prompt.

        Layout (ordered for maximal RadixAttention KV-cache reuse):
          [task instruction] run-constant  -> caches at the trie ROOT (1 prefill/run)
          [post prefix]       per-post      -> caches per post, shared across agents
          [agent suffix]      per-agent     -> the only varying tail
        The instruction is built from `allowed_chars` (the ENABLED actions, same
        set as the logit-bias) via build_task_instruction, NOT a hardcoded menu.
        It is byte-identical for a given allowed_chars, so the root-cache
        property holds within a run.

        Guarantees (for a fixed allowed_chars):
        1. prompt.startswith(build_task_instruction(allowed_chars) + build_post_prefix(post)) is True.
        2. That leading slice is byte-for-byte identical across any number of agents evaluating post.

        Args:
            post: Shared post prefix data.
            agent: Personalized agent suffix data.
            allowed_chars: Enabled action letters for this run (defaults to all).

        Returns:
            Complete assembled prompt ready for batched model forward pass.
        """
        instruction = cls.build_task_instruction(allowed_chars)
        prefix = cls.build_post_prefix(post)
        suffix = cls.build_agent_suffix(agent, topic=post.topic)
        return f"{instruction}{prefix}{suffix}"

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

        # Handle 'Action: ...' or '[Action]: ...' or 'Reaction: ...'
        if ":" in text:
            prefix_part, after_colon = text.split(":", 1)
            if any(
                k in prefix_part.lower()
                for k in ("action", "reaction", "decision", "choice", "response")
            ):
                text = after_colon.strip()

        if not text:
            return "S"

        # Check bracketed or parenthesized token: [L], (L), [R], (R), etc.
        bracket_match = re.search(r"[\[\(]([LRQCFS])[\]\)]", text, re.IGNORECASE)
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
        if "FOLLOW" in upper_text:
            return "F"
        if "SKIP" in upper_text:
            return "S"

        # Check word boundary regex for isolated action characters
        isolated_match = re.search(r"\b([LRQCFS])\b", upper_text)
        if isolated_match:
            return isolated_match.group(1).upper()

        return "S"
