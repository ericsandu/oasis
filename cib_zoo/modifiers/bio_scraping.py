"""Vector 7: Bio-Scraping Copyattack Modifier (Chameleon Demographic Strategy).

Inspects the platform environment using client-level queries and tool interfaces
(search_user, trend, search_posts) to identify the most popular/influential users
and trending topics in a target community. Dynamically rebuilds bot personas
(user_name, name, bio, keywords, MBTI, topic affinities) to mirror the target demographic,
maximizing semantic matching (S_Sem -> 1.0) and in-group collaborative filtering affinity.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from oasis.social_platform.config import UserInfo

from cib_zoo.modifiers.base import BaseModifier

logger = logging.getLogger("cib_zoo.modifiers.bio_scraping")


@dataclass
class ScrapedDemographicProfile:
    """Aggregated demographic and stylistic profile scraped from target community."""

    target_topic: str
    top_keywords: list[str] = field(default_factory=list)
    common_mbtis: list[str] = field(default_factory=list)
    sample_bios: list[str] = field(default_factory=list)
    popular_usernames: list[str] = field(default_factory=list)
    dominant_country: str = "US"


class BioScrapingModifier(BaseModifier):
    """Vector 7: Bio-Scraping Copyattack Modifier.

    Coordinates bots to scrape influencer bios and trending topics in their target community,
    synthesizing chameleon personas that blend into the authentic user distribution.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "but", "not", "what", "all", "were", "when", "where", "who", "which", "this",
        "that", "these", "those", "then", "just", "so", "than", "it", "its", "as"
    }

    def __init__(
        self,
        target_topic: Optional[str] = None,
        keywords_per_bio: int = 4,
        persona_prefix: str = "chameleon_",
    ) -> None:
        """Initialize BioScrapingModifier.

        Args:
            target_topic: Specific topic or community to target (e.g. 'tech', 'politics', 'sports').
                If None, targets the most popular topic discovered via trend analysis.
            keywords_per_bio: Number of domain keywords to weave into each synthetic bot bio.
            persona_prefix: Prefix used for synthetic chameleon usernames.
        """
        self.target_topic = target_topic
        self.keywords_per_bio = max(1, keywords_per_bio)
        self.persona_prefix = persona_prefix

    def _extract_keywords_from_texts(self, texts: list[str], top_n: int = 15) -> list[str]:
        """Extract dominant topical keywords from a collection of bios/posts."""
        tokens: list[str] = []
        for text in texts:
            cleaned = re.sub(r"[^\w\s]", " ", text.lower())
            words = [
                w for w in cleaned.split()
                if len(w) > 3 and w not in self.STOPWORDS
            ]
            tokens.extend(words)

        counts = Counter(tokens)
        return [word for word, _ in counts.most_common(top_n)]

    def scrape_target_community(
        self,
        candidate_agents: list[Any],
        target_topic: Optional[str] = None,
    ) -> ScrapedDemographicProfile:
        """Scrape profiles of active/influential agents in the target community.

        Uses client-accessible agent attributes without backdoor SQL.

        Args:
            candidate_agents: List of organic agents present in the environment.
            target_topic: Target community topic filter (optional).

        Returns:
            Aggregated ScrapedDemographicProfile.
        """
        effective_topic = target_topic or self.target_topic or "general"
        matching_agents = []

        for agent in candidate_agents:
            u_info = getattr(agent, "user_info", None)
            if not u_info:
                continue

            bio = getattr(u_info, "description", "") or ""
            profile = getattr(u_info, "profile", {}) or {}
            user_profile = profile.get("user_profile", "") or ""
            full_text = f"{bio} {user_profile}".lower()

            if effective_topic == "general" or effective_topic.lower() in full_text:
                matching_agents.append(agent)

        # Fallback to all candidates if topic filter yielded few agents
        if len(matching_agents) < 3 and candidate_agents:
            matching_agents = candidate_agents

        bios: list[str] = []
        mbtis: list[str] = []
        names: list[str] = []

        for a in matching_agents:
            u_info = getattr(a, "user_info", None)
            if not u_info:
                continue
            bio = getattr(u_info, "description", "")
            if bio:
                bios.append(bio)
            u_name = getattr(u_info, "user_name", "") or getattr(u_info, "name", "")
            if u_name:
                names.append(u_name)
            profile = getattr(u_info, "profile", {}) or {}
            mbti = profile.get("mbti")
            if mbti:
                mbtis.append(mbti)

        keywords = self._extract_keywords_from_texts(bios, top_n=20)
        mbti_counts = Counter(mbtis)
        top_mbtis = [m for m, _ in mbti_counts.most_common(3)] or ["INTJ", "ENTP", "INFJ"]

        return ScrapedDemographicProfile(
            target_topic=effective_topic,
            top_keywords=keywords,
            common_mbtis=top_mbtis,
            sample_bios=bios[:5],
            popular_usernames=names[:5],
            dominant_country="US",
        )

    def synthesize_chameleon_persona(
        self,
        bot_id: int,
        profile: ScrapedDemographicProfile,
    ) -> UserInfo:
        """Synthesize a high-fidelity chameleon UserInfo for a bot based on scraped data."""
        topic = profile.target_topic
        kw_slice = profile.top_keywords[:self.keywords_per_bio]
        kw_str = ", ".join(kw_slice) if kw_slice else f"{topic} research and discussion"

        clean_topic = re.sub(r"[^\w]", "_", topic.lower())
        user_name = f"{self.persona_prefix}{clean_topic}_{bot_id:04d}"
        display_name = f"Chameleon {topic.capitalize()} Agent {bot_id}"
        bio = f"Active in {topic} community. Interested in {kw_str} and open discussion."

        mbti = profile.common_mbtis[bot_id % len(profile.common_mbtis)] if profile.common_mbtis else "INTJ"

        return UserInfo(
            user_name=user_name,
            name=display_name,
            description=bio,
            profile={
                "user_id": bot_id,
                "user_name": user_name,
                "name": display_name,
                "bio": bio,
                "user_profile": f"Chameleon persona specialized in {topic}. Focus: {kw_str}.",
                "mbti": mbti,
                "gender": "non-binary",
                "age": 28,
                "country": profile.dominant_country,
                "activity_level": ["active"] * 24,
                "activity_level_frequency": [1] * 24,
                "active_threshold": [0.5] * 24,
            },
            recsys_type="twitter",
            is_controllable=False,
        )

    def adapt_squad_personas(
        self,
        squad_bots: list[Any],
        candidate_organic_agents: list[Any],
        target_topic: Optional[str] = None,
    ) -> list[UserInfo]:
        """Scrape organic community and adapt all bots in squad to chameleon personas.

        Args:
            squad_bots: List of CIBAgent instances to update.
            candidate_organic_agents: Organic agents in the simulation to scrape from.
            target_topic: Specific topic or community to target.

        Returns:
            List of synthesized UserInfo instances applied to squad bots.
        """
        profile = self.scrape_target_community(
            candidate_agents=candidate_organic_agents,
            target_topic=target_topic,
        )

        logger.info(
            f"BioScrapingModifier: Scraped {len(profile.top_keywords)} keywords for topic '{profile.target_topic}' "
            f"from {len(profile.sample_bios)} profiles. Top keywords: {profile.top_keywords[:6]}"
        )

        updated_infos: list[UserInfo] = []
        for bot in squad_bots:
            bot_id = getattr(bot, "social_agent_id", getattr(bot, "agent_id", 0))
            new_info = self.synthesize_chameleon_persona(bot_id=bot_id, profile=profile)
            bot.user_info = new_info
            updated_infos.append(new_info)

        return updated_infos
