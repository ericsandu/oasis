"""Client-facing perception cache and environment for CIBAgent with zero SQLite queries."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import Environment
from oasis.social_platform.typing import ActionType
from cib_zoo.agent.budget import normalize_action_type

logger = logging.getLogger("cib_zoo.perception")


@dataclass
class CachedPost:
    """Represents a post observed through client-level platform actions."""

    post_id: int
    user_id: int
    content: str
    created_at: Optional[str] = None
    num_likes: int = 0
    num_dislikes: int = 0
    num_shares: int = 0
    original_post_id: Optional[int] = None
    quote_content: Optional[str] = None
    comments: list[dict[str, Any]] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)
    step_observed: int = 0
    timestamp_observed: float = 0.0
    source: str = "feed"  # "feed" | "search"

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard post dictionary matching OASIS platform format."""
        d = dict(self.raw_data) if self.raw_data else {}
        d.update(
            {
                "post_id": self.post_id,
                "user_id": self.user_id,
                "content": self.content,
                "created_at": self.created_at,
                "num_likes": self.num_likes,
                "num_dislikes": self.num_dislikes,
                "num_shares": self.num_shares,
                "original_post_id": self.original_post_id,
                "quote_content": self.quote_content,
                "comments": self.comments,
                "_source": self.source,
                "_step_observed": self.step_observed,
            }
        )
        return d

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        step: int = 0,
        timestamp: float = 0.0,
        source: str = "feed",
    ) -> CachedPost:
        return cls(
            post_id=int(data["post_id"]),
            user_id=int(data.get("user_id", 0)),
            content=str(data.get("content", "")),
            created_at=data.get("created_at"),
            num_likes=int(data.get("num_likes", 0)),
            num_dislikes=int(data.get("num_dislikes", 0)),
            num_shares=int(data.get("num_shares", 0)),
            original_post_id=data.get("original_post_id"),
            quote_content=data.get("quote_content"),
            comments=list(data.get("comments", []) or []),
            raw_data=dict(data),
            step_observed=step,
            timestamp_observed=timestamp or time.monotonic(),
            source=source,
        )


@dataclass
class CachedUser:
    """Represents a user observed through client-level platform actions."""

    user_id: int
    user_name: str
    name: Optional[str] = None
    bio: Optional[str] = None
    created_at: Optional[str] = None
    num_followers: int = 0
    num_followings: int = 0
    raw_data: dict[str, Any] = field(default_factory=dict)
    step_observed: int = 0
    timestamp_observed: float = 0.0
    source: str = "search"

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.raw_data) if self.raw_data else {}
        d.update(
            {
                "user_id": self.user_id,
                "user_name": self.user_name,
                "name": self.name,
                "bio": self.bio,
                "created_at": self.created_at,
                "num_followers": self.num_followers,
                "num_followings": self.num_followings,
                "_source": self.source,
                "_step_observed": self.step_observed,
            }
        )
        return d

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        step: int = 0,
        timestamp: float = 0.0,
        source: str = "search",
    ) -> CachedUser:
        return cls(
            user_id=int(data["user_id"]),
            user_name=str(data.get("user_name", "")),
            name=data.get("name"),
            bio=data.get("bio"),
            created_at=data.get("created_at"),
            num_followers=int(data.get("num_followers", 0)),
            num_followings=int(data.get("num_followings", 0)),
            raw_data=dict(data),
            step_observed=step,
            timestamp_observed=timestamp or time.monotonic(),
            source=source,
        )


class PerceptionCache:
    """Thread-safe, in-memory perception cache for CIBAgent.

    Adheres strictly to client-facing discovery: zero direct SQLite calls.
    """

    def __init__(
        self,
        ttl_steps: Optional[int] = None,
        ttl_seconds: Optional[float] = None,
        max_size: int = 1000,
        current_step: int = 0,
    ) -> None:
        self._ttl_steps = ttl_steps
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._current_step = current_step

        self._posts: OrderedDict[int, CachedPost] = OrderedDict()
        self._users: OrderedDict[int, CachedUser] = OrderedDict()
        self._search_posts_cache: dict[str, list[dict[str, Any]]] = {}
        self._search_users_cache: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @property
    def current_step(self) -> int:
        with self._lock:
            return self._current_step

    def advance_step(self, step: Optional[int] = None) -> None:
        """Advance internal step clock and invalidate expired entries."""
        with self._lock:
            if step is not None:
                self._current_step = step
            else:
                self._current_step += 1
            self.invalidate()

    def update_from_feed(
        self,
        posts: list[dict[str, Any]] | dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Cache posts returned by ActionType.REFRESH."""
        with self._lock:
            active_step = self._current_step if step is None else step
            now = time.monotonic()

            post_list: list[dict[str, Any]] = []
            if isinstance(posts, dict):
                post_list = posts.get("posts", [])
            elif isinstance(posts, list):
                post_list = posts

            for p in post_list:
                if not isinstance(p, dict) or "post_id" not in p:
                    continue
                cached = CachedPost.from_dict(p, step=active_step, timestamp=now, source="feed")
                post_id = cached.post_id
                if post_id in self._posts:
                    self._posts.move_to_end(post_id)
                self._posts[post_id] = cached

                while len(self._posts) > self._max_size:
                    self._posts.popitem(last=False)

    def update_from_search(
        self,
        posts_or_users: list[dict[str, Any]] | dict[str, Any],
        search_type: str = "posts",
        step: Optional[int] = None,
        query: Optional[str] = None,
    ) -> None:
        """Cache posts or users returned by SEARCH_POSTS or SEARCH_USER."""
        with self._lock:
            active_step = self._current_step if step is None else step
            now = time.monotonic()

            data_list: list[dict[str, Any]] = []
            if isinstance(posts_or_users, dict):
                if "posts" in posts_or_users:
                    search_type = "posts"
                    data_list = posts_or_users.get("posts", [])
                elif "users" in posts_or_users:
                    search_type = "users"
                    data_list = posts_or_users.get("users", [])
                else:
                    data_list = [posts_or_users]
            elif isinstance(posts_or_users, list):
                data_list = posts_or_users

            if search_type in ("posts", "post"):
                if query:
                    self._search_posts_cache[query] = [dict(p) for p in data_list]
                for p in data_list:
                    if not isinstance(p, dict) or "post_id" not in p:
                        continue
                    cached = CachedPost.from_dict(p, step=active_step, timestamp=now, source="search")
                    post_id = cached.post_id
                    if post_id in self._posts:
                        self._posts.move_to_end(post_id)
                    self._posts[post_id] = cached
                    while len(self._posts) > self._max_size:
                        self._posts.popitem(last=False)

            elif search_type in ("users", "user"):
                if query:
                    self._search_users_cache[query] = [dict(u) for u in data_list]
                for u in data_list:
                    if not isinstance(u, dict) or "user_id" not in u:
                        continue
                    cached_user = CachedUser.from_dict(u, step=active_step, timestamp=now, source="search")
                    user_id = cached_user.user_id
                    if user_id in self._users:
                        self._users.move_to_end(user_id)
                    self._users[user_id] = cached_user
                    while len(self._users) > self._max_size:
                        self._users.popitem(last=False)

    def update(
        self,
        action_type: Union[ActionType, str],
        data: dict[str, Any] | list[dict[str, Any]],
        query: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Convenience dispatcher for CIBAgent.update_perception."""
        canonical = normalize_action_type(action_type)
        if canonical == ActionType.REFRESH:
            self.update_from_feed(data, step=step)
        elif canonical == ActionType.SEARCH_POSTS:
            self.update_from_search(data, search_type="posts", step=step, query=query)
        elif canonical == ActionType.SEARCH_USER:
            self.update_from_search(data, search_type="users", step=step, query=query)

    def get_cached_posts(
        self,
        max_age_steps: Optional[int] = None,
        max_age_seconds: Optional[float] = None,
        source: Optional[str] = None,
        query: Optional[str] = None,
        as_dicts: bool = True,
    ) -> list[dict[str, Any]] | list[CachedPost]:
        """Query cached posts with freshness filtering."""
        with self._lock:
            now = time.monotonic()
            step_limit = max_age_steps if max_age_steps is not None else self._ttl_steps
            sec_limit = max_age_seconds if max_age_seconds is not None else self._ttl_seconds

            results: list[CachedPost] = []
            for post in reversed(self._posts.values()):  # newest first
                if step_limit is not None and (self._current_step - post.step_observed) > step_limit:
                    continue
                if sec_limit is not None and (now - post.timestamp_observed) > sec_limit:
                    continue
                if source is not None and post.source != source:
                    continue
                if query is not None and query.lower() not in post.content.lower():
                    continue
                results.append(post)

            if as_dicts:
                return [p.to_dict() for p in results]
            return results

    def get_cached_users(
        self,
        max_age_steps: Optional[int] = None,
        max_age_seconds: Optional[float] = None,
        query: Optional[str] = None,
        as_dicts: bool = True,
    ) -> list[dict[str, Any]] | list[CachedUser]:
        """Query cached users with freshness filtering."""
        with self._lock:
            now = time.monotonic()
            step_limit = max_age_steps if max_age_steps is not None else self._ttl_steps
            sec_limit = max_age_seconds if max_age_seconds is not None else self._ttl_seconds

            results: list[CachedUser] = []
            for user in reversed(self._users.values()):
                if step_limit is not None and (self._current_step - user.step_observed) > step_limit:
                    continue
                if sec_limit is not None and (now - user.timestamp_observed) > sec_limit:
                    continue
                if query is not None:
                    q = query.lower()
                    bio_match = user.bio and q in user.bio.lower()
                    name_match = user.name and q in user.name.lower()
                    uname_match = q in user.user_name.lower()
                    if not (uname_match or name_match or bio_match):
                        continue
                results.append(user)

            if as_dicts:
                return [u.to_dict() for u in results]
            return results

    def get_post_by_id(self, post_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            post = self._posts.get(int(post_id))
            return post.to_dict() if post else None

    get_post = get_post_by_id

    def get_user_by_id(self, user_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            user = self._users.get(int(user_id))
            return user.to_dict() if user else None

    get_user = get_user_by_id

    def get_feed(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        posts = self.get_cached_posts(source="feed")
        if limit is None:
            return posts
        return posts[:limit]

    def invalidate(
        self,
        current_step: Optional[int] = None,
        current_time: Optional[float] = None,
    ) -> int:
        """Purge expired entries from cache. Returns count of purged items."""
        with self._lock:
            c_step = self._current_step if current_step is None else current_step
            now = time.monotonic() if current_time is None else current_time
            purged = 0

            if self._ttl_steps is not None:
                expired_posts = [
                    pid for pid, p in self._posts.items()
                    if (c_step - p.step_observed) > self._ttl_steps
                ]
                for pid in expired_posts:
                    del self._posts[pid]
                    purged += 1

                expired_users = [
                    uid for uid, u in self._users.items()
                    if (c_step - u.step_observed) > self._ttl_steps
                ]
                for uid in expired_users:
                    del self._users[uid]
                    purged += 1

            if self._ttl_seconds is not None:
                expired_posts = [
                    pid for pid, p in self._posts.items()
                    if (now - p.timestamp_observed) > self._ttl_seconds
                ]
                for pid in expired_posts:
                    del self._posts[pid]
                    purged += 1

                expired_users = [
                    uid for uid, u in self._users.items()
                    if (now - u.timestamp_observed) > self._ttl_seconds
                ]
                for uid in expired_users:
                    del self._users[uid]
                    purged += 1

            return purged

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._posts.clear()
            self._users.clear()
            self._search_posts_cache.clear()
            self._search_users_cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._posts) + len(self._users)


class CIBEnvironment(Environment):
    """Client-facing environment for CIB agents eliminating all backdoor SQLite queries."""

    def __init__(self, action: SocialAction, perception: PerceptionCache) -> None:
        self.action: SocialAction = action
        self.perception: PerceptionCache = perception

    async def get_posts_env(self) -> str:
        res = await self.action.refresh()
        if isinstance(res, dict) and res.get("success"):
            self.perception.update(ActionType.REFRESH, res)
            return json.dumps(res.get("posts", []))
        return "After refreshing, there are no existing posts."

    async def get_followers_env(self) -> str:
        user = self.perception.get_user(self.action.agent_id)
        count = user.get("num_followers", 0) if user else 0
        return f"I have {count} followers."

    async def get_follows_env(self) -> str:
        user = self.perception.get_user(self.action.agent_id)
        count = user.get("num_followings", 0) if user else 0
        return f"I have {count} follows."

    async def to_text_prompt(self) -> str:
        return f"CIB Agent {self.action.agent_id} observation ready."
