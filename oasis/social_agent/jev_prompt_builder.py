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
    # Audience context (matches classic OASIS env: "I have N followers. I have
    # N follows."). This broadcaster framing is what motivates reposting; its
    # absence was why JEV agents reacted as readers (Like) not broadcasters
    # (Repost). -1 => unknown/not provided (line omitted).
    num_followers: int = -1
    num_follows: int = -1


class JEVPromptBuilder:
    """Joint Evaluation Vectorization (JEV) prompt constructor.

    Inverts conventional LLM social agent prompts into [Shared Post Prefix] + [Personalized Agent Suffix]
    to maximize RadixAttention KV-cache prefix hits across multiple agents evaluating identical posts.
    """

    VALID_ACTIONS: Sequence[str] = ("L", "R", "Q", "C", "F", "S")

    SEMANTIC_ACTIONS: tuple[str, ...] = (
        "like_post",
        "repost",
        "follow",
        "do_nothing",
    )

    MCQ_OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

    # Canonical mapping: Option letter -> (ActionChar, ActionName, Description)
    # Places primary amplification action (repost) in Slot A to match diffusion dynamics.
    MCQ_ACTION_SPECS: list[tuple[str, str, str]] = [
        ("R", "repost", "Repost the post to your followers"),
        ("L", "like_post", "Like the post"),
        ("F", "follow", "Follow the post author"),
        ("S", "do_nothing", "Do nothing (skip without interacting)"),
        ("Q", "quote_post", "Quote the post with your commentary"),
        ("C", "create_comment", "Write a comment on the post"),
    ]

    MCQ_LETTER_TO_CHAR: dict[str, str] = {
        "A": "R",
        "B": "L",
        "C": "F",
        "D": "S",
        "E": "Q",
        "F": "C",
    }

    MCQ_LETTER_TO_NAME: dict[str, str] = {
        "A": "repost",
        "B": "like_post",
        "C": "follow",
        "D": "do_nothing",
        "E": "quote_post",
        "F": "create_comment",
    }

    MCQ_CHAR_TO_LETTER: dict[str, str] = {
        "R": "A",
        "L": "B",
        "F": "C",
        "S": "D",
        "Q": "E",
        "C": "F",
    }

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

    ACTION_CHAR_TO_NAME: dict[str, str] = {
        "L": "like_post",
        "R": "repost",
        "Q": "quote_post",
        "C": "create_comment",
        "F": "follow",
        "S": "do_nothing",
    }

    ACTION_NAME_TO_CHAR: dict[str, str] = {
        "like_post": "L",
        "like": "L",
        "repost": "R",
        "quote_post": "Q",
        "quote": "Q",
        "create_comment": "C",
        "comment": "C",
        "follow": "F",
        "do_nothing": "S",
        "skip": "S",
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
    def resolve_action_specs(
        cls,
        allowed_chars: Sequence[str] | None = None,
    ) -> list[tuple[str, str, str]]:
        """Dynamically resolves the active actions for this experiment in canonical priority order.

        Preserves canonical priority: Repost -> Like -> Follow -> Skip -> Quote -> Comment,
        ensuring affirmative cascade actions are prioritized on supportive items.

        Returns:
            List of (action_char, action_name, action_description) tuples.
        """
        enabled_chars: set[str] = set()
        if allowed_chars:
            for item in allowed_chars:
                if item.upper() in cls.ACTION_DESCRIPTIONS:
                    enabled_chars.add(item.upper())
                else:
                    c = cls.ACTION_NAME_TO_CHAR.get(item.lower(), "")
                    if c:
                        enabled_chars.add(c)
        if not enabled_chars:
            enabled_chars = {"R", "L", "F", "S"}
        if "S" not in enabled_chars:
            enabled_chars.add("S")

        return [spec for spec in cls.MCQ_ACTION_SPECS if spec[0] in enabled_chars]

    @classmethod
    def get_mcq_mappings(
        cls,
        allowed_chars: Sequence[str] | None = None,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, str], list[str]]:
        """Dynamically builds option-letter mappings for the specific active actions in an experiment.

        Returns:
            (letter_to_action, action_to_letter, option_letters)
            where letter_to_action maps e.g. 'A' -> ('R', 'repost'),
            and option_letters is ['A', 'B', 'C', 'D'] matching active specs.
        """
        specs = cls.resolve_action_specs(allowed_chars)
        letter_to_action: dict[str, tuple[str, str]] = {}
        action_to_letter: dict[str, str] = {}
        option_letters: list[str] = []
        for idx, (ch, name, _desc) in enumerate(specs):
            letter = cls.MCQ_OPTION_LETTERS[idx]
            letter_to_action[letter] = (ch, name)
            action_to_letter[ch] = letter
            action_to_letter[name] = letter
            option_letters.append(letter)
        return letter_to_action, action_to_letter, option_letters

    @classmethod
    def build_task_instruction(
        cls,
        allowed_chars: Sequence[str] | None = None,
        competitive: bool = False,
        use_mcq_options: bool = True,
        **kwargs: Any,
    ) -> str:
        """Render the run-constant instruction preamble dynamically from the ENABLED actions.

        Formats decisions as canonical discrete choice options (A, B, C, D) corresponding
        to the TypeSafe Jev decision model paradigm, eliminating unigram continuation bias
        while maintaining 100% byte-identical RadixAttention root KV-cache reuse.
        """
        specs = cls.resolve_action_specs(allowed_chars)
        letter_to_action, _, option_letters = cls.get_mcq_mappings(allowed_chars)

        scarcity = ""
        if competitive:
            scarcity = (
                "You are seeing MANY posts this round, but only ONE action -- "
                "your single strongest choice across all of them -- will actually "
                "be performed. Reserve a high-impact action (such as repost) for "
                "the post that most deserves it; do not spend your one action on a "
                "low-value reaction to a post you merely find agreeable.\n"
            )

        if use_mcq_options:
            menu_lines = [
                f"({letter}) {desc}"
                for letter, (_, _, desc) in zip(option_letters, specs)
            ]
            letters_str = ", ".join(option_letters)
            return (
                f"{cls.OASIS_PROMPT_STEM}\n"
                f"{cls.OASIS_ENV_STEER}\n"
                f"{scarcity}"
                f"For the post below, choose exactly ONE action and output ONLY its option letter ({letters_str}):\n"
                + "\n".join(menu_lines)
                + ".\n\n"
            )

        enabled_chars = {s[0] for s in specs}
        chars = [c for c in cls.VALID_ACTIONS if c in enabled_chars]
        menu = ", ".join(f"{c} ({cls.ACTION_DESCRIPTIONS[c][1]})" for c in chars)
        letters = "/".join(chars)
        return (
            f"{cls.OASIS_PROMPT_STEM}\n"
            f"{cls.OASIS_ENV_STEER}\n"
            f"{scarcity}"
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

        # Audience context, matching base-OASIS env ("I have N followers. I have
        # N follows."). This broadcaster framing motivates reposting; omit when
        # unknown (-1).
        if agent.num_followers >= 0 or agent.num_follows >= 0:
            nf = agent.num_followers if agent.num_followers >= 0 else 0
            ng = agent.num_follows if agent.num_follows >= 0 else 0
            lines.append(
                f"[AUDIENCE]: I have {nf} followers. I have {ng} follows."
            )

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
    def build_shared_prefix(
        cls,
        post: PostPrefixData,
        allowed_chars: Sequence[str] | None = None,
        competitive: bool = False,
        use_mcq_options: bool = True,
        **kwargs: Any,
    ) -> str:
        """Constructs the shared [Task Instruction] + [Post Prefix] slice.

        This slice is 100% byte-for-byte identical across ALL candidate agents evaluating
        the same post, guaranteeing maximum RadixAttention KV-cache prefix hits.

        Args:
            post: Shared post prefix data.
            allowed_chars: Enabled actions for this run.
            competitive: Whether scarcity framing is enabled.
            use_mcq_options: Whether discrete choice (A, B, C, D) options are used.

        Returns:
            Byte-identical shared prefix string.
        """
        instruction = cls.build_task_instruction(
            allowed_chars=allowed_chars,
            competitive=competitive,
            use_mcq_options=use_mcq_options,
            **kwargs,
        )
        post_prefix = cls.build_post_prefix(post)
        return f"{instruction}{post_prefix}"

    @classmethod
    def get_shared_prefix_hash(
        cls,
        post: PostPrefixData,
        allowed_chars: Sequence[str] | None = None,
        competitive: bool = False,
        use_mcq_options: bool = True,
        **kwargs: Any,
    ) -> str:
        """Returns the SHA-256 hex digest of the shared cacheable prefix."""
        import hashlib

        prefix_bytes = cls.build_shared_prefix(
            post=post,
            allowed_chars=allowed_chars,
            competitive=competitive,
            use_mcq_options=use_mcq_options,
            **kwargs,
        ).encode("utf-8")
        return hashlib.sha256(prefix_bytes).hexdigest()

    @classmethod
    def assemble_eval_prompt(
        cls,
        post: PostPrefixData,
        agent: AgentSuffixData,
        allowed_chars: Sequence[str] | None = None,
        competitive: bool = False,
        use_mcq_options: bool = True,
        **kwargs: Any,
    ) -> str:
        """Concatenates instruction + post prefix + agent suffix into one eval prompt.

        Layout (ordered for maximal RadixAttention KV-cache reuse):
          [task instruction] run-constant  -> caches at the trie ROOT (1 prefill/run)
          [post prefix]       per-post      -> caches per post, shared across agents
          [agent suffix]      per-agent     -> the only varying tail

        Guarantees (for fixed allowed_chars and use_mcq_options):
        1. prompt.startswith(build_shared_prefix(post, allowed_chars=...)) is True.
        2. That leading slice is 100% byte-for-byte identical across any number of agents evaluating post.

        Args:
            post: Shared post prefix data.
            agent: Personalized agent suffix data.
            allowed_chars: Enabled action letters or names for this run.
            competitive: Whether scarcity framing is enabled.
            use_mcq_options: Whether discrete choice (A, B, C, D) options are used.

        Returns:
            Complete assembled prompt ready for batched model forward pass.
        """
        shared_prefix = cls.build_shared_prefix(
            post=post,
            allowed_chars=allowed_chars,
            competitive=competitive,
            use_mcq_options=use_mcq_options,
            **kwargs,
        )
        suffix = cls.build_agent_suffix(agent, topic=post.topic)
        return f"{shared_prefix}{suffix}"

    @classmethod
    def build_feed_prompt(
        cls,
        agent: "AgentSuffixData",
        posts: list,
        allowed_chars: Sequence[str] | None = None,
    ) -> str:
        """WHOLE-FEED prompt: one agent sees ALL feed posts and selects one post
        + one action -- mirroring base OASIS classic (which shows the whole feed
        and asks the agent to pick). Used by the feed-mode experiment to test
        whether comparative/feed-level context restores reposting.

        Only the root task stem is run-constant (cacheable); the per-agent feed
        block is per-agent (feeds differ), so this trades the per-post cache win.
        """
        chars = [c for c in (allowed_chars or cls.VALID_ACTIONS)
                 if c in cls.ACTION_DESCRIPTIONS]
        if "S" not in chars:
            chars.append("S")
        menu = ", ".join(f"{c} ({cls.ACTION_DESCRIPTIONS[c][1]})" for c in chars)
        persona = cls.build_agent_persona_context(agent)
        feed_lines = []
        for p in posts:
            feed_lines.append(cls.build_post_prefix(p).rstrip())
        feed_block = "\n".join(feed_lines)
        return (
            f"{cls.OASIS_PROMPT_STEM}\n"
            f"{cls.OASIS_ENV_STEER}\n"
            f"{persona}\n"
            f"After refreshing, you see these posts:\n{feed_block}\n\n"
            f"Pick ONE post and ONE action that best reflects your inclination. "
            f"Available actions: {menu}.\n"
            f"Respond with JSON: {{\"post_id\": <id>, \"action\": \"<letter>\"}}."
        )

    @classmethod
    def build_verbatim_prompt(
        cls,
        agent: "AgentSuffixData",
        posts: list,
        allowed_chars: Sequence[str] | None = None,
        whole_feed: bool = True,
    ) -> str:
        """Reproduce the EXACT upstream OASIS agent prompt, byte-for-byte, so JEV
        sees precisely what a classic agent sees. Three upstream pieces:
          1. system message  (user.py: to_twitter_system_message)
          2. user message    (agent.py: perform_action_by_llm)
          3. env block        (agent_environment.py: env_template), with the feed
             rendered as json.dumps(posts, indent=4) exactly like classic, plus
             "I have N followers." / "I have N follows."
        The ONLY non-upstream addition is a trailing output-format line, because
        classic names actions via TOOL SCHEMAS which JEV (logit-bias / guided
        JSON) does not have. whole_feed=False renders a single-post feed (per-post
        mode); True renders all posts (whole-feed mode).
        """
        import json as _json
        chars = [c for c in (allowed_chars or cls.VALID_ACTIONS)
                 if c in cls.ACTION_DESCRIPTIONS]
        if "S" not in chars:
            chars.append("S")
        # --- 1. system message (verbatim) ---
        name_string = f"Your name is {agent.user_name}."
        desc = f"{name_string}\nYour have profile: {agent.bio}."
        system_content = (
            "\n# OBJECTIVE\n"
            "You're a Twitter user, and I'll present you with some tweets. "
            "After you see the tweets, choose some actions from the following "
            "functions.\n\n# SELF-DESCRIPTION\n"
            "Your actions should be consistent with your self-description and "
            f"personality.\n{desc}\n\n# RESPONSE METHOD\n"
            "Please perform actions by tool calling.\n        "
        )
        # --- 3. env block (verbatim env_template, no groups) ---
        post_dicts = []
        for p in posts:
            post_dicts.append({
                "post_id": p.post_id,
                "user_id": getattr(p, "author_name", ""),
                "content": p.content,
                "num_likes": p.num_likes,
                "num_shares": p.num_shares,
            })
        posts_json = _json.dumps(post_dicts, indent=4)
        nf = agent.num_followers if agent.num_followers >= 0 else 0
        ng = agent.num_follows if agent.num_follows >= 0 else 0
        env_prompt = (
            f"I have {nf} followers. I have {ng} follows.\n"
            f"After refreshing, you see some posts {posts_json}\n"
            "pick one you want to perform action that best reflects your "
            "current inclination based on your profile and posts content. "
            "Do not limit your action in just `like` to like posts"
        )
        # --- 2. user message (verbatim) ---
        user_msg = (
            "Please perform social media actions after observing the platform "
            "environments. Notice that don't limit your actions for example to "
            f"just like the posts. Here is your social media environment: {env_prompt}"
        )
        # --- JEV-only output contract (classic uses tool schemas) ---
        menu = ", ".join(f"{c} ({cls.ACTION_DESCRIPTIONS[c][1]})" for c in chars)
        if whole_feed:
            out = (f'\n\nChoose ONE post and ONE action ({menu}). '
                   'Respond with JSON: {"post_id": <id>, "action": "<letter>"}.')
        else:
            letters = "/".join(chars)
            out = (f"\n\nChoose ONE action ({letters}): {menu}. "
                   "Output ONLY the letter.\nAction: ")
        return f"{system_content}\n{user_msg}{out}"

    @classmethod
    def parse_action_char(
        cls,
        raw_response: str,
        allowed_chars: Sequence[str] | None = None,
    ) -> str:
        """Parses and validates a single reaction character ('L', 'R', 'C', 'S') from model output.

        Dynamically resolves MCQ option letters ('A', 'B', 'C', 'D') to the corresponding action
        character for this experiment. Also handles raw action characters ('R', 'L') or words.

        Args:
            raw_response: Raw completion string emitted by LLM.
            allowed_chars: Optional enabled actions for dynamic MCQ resolution.

        Returns:
            One of 'L', 'R', 'Q', 'C', 'F', 'S'.
        """
        if not raw_response or not raw_response.strip():
            return "S"

        text = raw_response.strip()

        # Handle 'Action: ...' or '[Action]: ...' or 'Reaction: ...' or 'Decision: ...'
        if ":" in text:
            prefix_part, after_colon = text.split(":", 1)
            if any(
                k in prefix_part.lower()
                for k in ("action", "reaction", "decision", "choice", "response")
            ):
                text = after_colon.strip()

        if not text:
            return "S"

        # Resolve dynamic MCQ mapping for this experiment
        letter_to_action, _, option_letters = cls.get_mcq_mappings(allowed_chars)

        # Check bracketed or parenthesized token: (A), [A], (B), [B], etc.
        opt_str = "".join(option_letters)
        if opt_str:
            bracket_match = re.search(rf"[\[\(]([{opt_str}])[\]\)]", text, re.IGNORECASE)
            if bracket_match:
                letter = bracket_match.group(1).upper()
                if letter in letter_to_action:
                    return letter_to_action[letter][0]

        # Check immediate first character as MCQ option letter
        first_char = text[0].upper()
        if first_char in letter_to_action:
            return letter_to_action[first_char][0]

        # Check word boundary regex for isolated MCQ option letters
        for letter in option_letters:
            if re.search(rf"\b{letter}\b", text, re.IGNORECASE):
                return letter_to_action[letter][0]

        # Check full word matches
        upper_text = text.upper()
        if "REPOST" in upper_text:
            return "R"
        if "LIKE" in upper_text:
            return "L"
        if "QUOTE" in upper_text:
            return "Q"
        if "COMMENT" in upper_text:
            return "C"
        if "FOLLOW" in upper_text:
            return "F"
        if "SKIP" in upper_text or "DO_NOTHING" in upper_text or "NOTHING" in upper_text:
            return "S"

        # Check raw action characters: [R], (R), [L], etc.
        raw_bracket = re.search(r"[\[\(]([LRQCFS])[\]\)]", text, re.IGNORECASE)
        if raw_bracket:
            return raw_bracket.group(1).upper()
        if first_char in cls.VALID_ACTIONS:
            return first_char

        return "S"

    @classmethod
    def parse_semantic_action(
        cls,
        raw_response: str,
        allowed_chars: Sequence[str] | None = None,
    ) -> str:
        """Parses and validates a semantic action name ('like_post', 'repost', 'follow', 'do_nothing').

        Dynamically resolves MCQ option letters ('A', 'B', 'C', 'D') to the corresponding action
        name for this experiment. Also handles raw action names or characters.

        Args:
            raw_response: Raw completion or JSON string emitted by LLM.
            allowed_chars: Optional enabled actions for dynamic MCQ resolution.

        Returns:
            One of 'like_post', 'repost', 'quote_post', 'create_comment', 'follow', 'do_nothing'.
        """
        if not raw_response or not raw_response.strip():
            return "do_nothing"

        text = raw_response.strip().lower()

        # Handle 'Action: ...' or '[Action]: ...' or 'Decision: ...'
        if ":" in text:
            prefix_part, after_colon = text.split(":", 1)
            if any(
                k in prefix_part
                for k in ("action", "reaction", "decision", "choice", "response")
            ):
                text = after_colon.strip()

        if not text:
            return "do_nothing"

        # Resolve dynamic MCQ mapping for this experiment
        letter_to_action, _, option_letters = cls.get_mcq_mappings(allowed_chars)

        # Check bracketed or parenthesized token: (A), [A], (B), etc.
        opt_str = "".join(option_letters)
        if opt_str:
            bracket_match = re.search(rf"[\[\(]([{opt_str}])[\]\)]", text, re.IGNORECASE)
            if bracket_match:
                letter = bracket_match.group(1).upper()
                if letter in letter_to_action:
                    return letter_to_action[letter][1]

        # Check immediate first character as MCQ option letter
        first_char = text[0].upper()
        if first_char in letter_to_action:
            return letter_to_action[first_char][1]

        # Check word boundary regex for isolated MCQ option letters
        for letter in option_letters:
            if re.search(rf"\b{letter}\b", text, re.IGNORECASE):
                return letter_to_action[letter][1]

        # Direct exact or substring matches for canonical action names
        if "repost" in text:
            return "repost"
        if "like_post" in text or "like" in text:
            return "like_post"
        if "follow" in text:
            return "follow"
        if "do_nothing" in text or "nothing" in text or "skip" in text:
            return "do_nothing"
        if "quote_post" in text or "quote" in text:
            return "quote_post"
        if "create_comment" in text or "comment" in text:
            return "create_comment"

        # Fall back to single char parser mapped to action name
        ch = cls.parse_action_char(raw_response, allowed_chars=allowed_chars)
        return cls.ACTION_CHAR_TO_NAME.get(ch, "do_nothing")
