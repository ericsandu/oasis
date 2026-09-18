"""Unit tests for atomic CIB action primitives."""

import pytest
from oasis.social_platform.typing import ActionType

from cib_zoo.primitives import (
    CommentPrimitive,
    FollowPrimitive,
    LikePrimitive,
    PostPrimitive,
    RepostPrimitive,
    SearchPrimitive,
)


class TestLikePrimitive:
    def test_like_primitive_generation(self) -> None:
        primitive = LikePrimitive(post_id=42)
        action = primitive.generate()
        assert action.action_type == ActionType.LIKE_POST
        assert action.action_args == {"post_id": 42}

    def test_unlike_primitive_generation(self) -> None:
        primitive = LikePrimitive(post_id=42, unlike=True)
        action = primitive.generate()
        assert action.action_type == ActionType.UNLIKE_POST
        assert action.action_args == {"post_id": 42}

    def test_override_at_generation(self) -> None:
        primitive = LikePrimitive(post_id=42)
        action = primitive.generate(post_id=99, unlike=True)
        assert action.action_type == ActionType.UNLIKE_POST
        assert action.action_args == {"post_id": 99}

    def test_invalid_post_id(self) -> None:
        primitive = LikePrimitive()
        with pytest.raises(ValueError, match="post_id is required"):
            primitive.generate()

        with pytest.raises(ValueError, match="positive integer"):
            primitive.generate(post_id=-5)


class TestCommentPrimitive:
    def test_comment_generation(self) -> None:
        primitive = CommentPrimitive(post_id=10, content="Great point!")
        action = primitive.generate()
        assert action.action_type == ActionType.CREATE_COMMENT
        assert action.action_args == {"post_id": 10, "content": "Great point!"}

    def test_template_formatting(self) -> None:
        primitive = CommentPrimitive(post_id=10, content="Hello bot {bot_id} at step {step}")
        action = primitive.generate(bot_id=7, step=3)
        assert action.action_args["content"] == "Hello bot 7 at step 3"

    def test_invalid_content(self) -> None:
        primitive = CommentPrimitive(post_id=10)
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            primitive.generate(content="")


class TestPostPrimitive:
    def test_post_generation_without_hashtags(self) -> None:
        primitive = PostPrimitive(content="Just setting up my synthetic persona.")
        action = primitive.generate()
        assert action.action_type == ActionType.CREATE_POST
        assert action.action_args == {"content": "Just setting up my synthetic persona."}

    def test_post_generation_with_hashtags(self) -> None:
        primitive = PostPrimitive(
            content="Check out the latest update!",
            hashtags=["tech", "#trending", "ai"],
        )
        action = primitive.generate()
        assert action.action_type == ActionType.CREATE_POST
        # Hashtags must be placed strictly within the content string
        assert action.action_args["content"] == "Check out the latest update! #tech #trending #ai"

    def test_empty_content_raises(self) -> None:
        primitive = PostPrimitive()
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            primitive.generate(content="   ")


class TestRepostPrimitive:
    def test_repost_without_quote(self) -> None:
        primitive = RepostPrimitive(post_id=101)
        action = primitive.generate()
        assert action.action_type == ActionType.REPOST
        assert action.action_args == {"post_id": 101}

    def test_repost_with_quote(self) -> None:
        primitive = RepostPrimitive(post_id=101, quote_content="Interesting analysis!")
        action = primitive.generate()
        assert action.action_type == ActionType.QUOTE_POST
        assert action.action_args == {"post_id": 101, "quote_content": "Interesting analysis!"}


class TestFollowPrimitive:
    def test_follow_generation(self) -> None:
        primitive = FollowPrimitive(followee_id=5)
        action = primitive.generate()
        assert action.action_type == ActionType.FOLLOW
        assert action.action_args == {"followee_id": 5}

    def test_unfollow_generation(self) -> None:
        primitive = FollowPrimitive(followee_id=5, unfollow=True)
        action = primitive.generate()
        assert action.action_type == ActionType.UNFOLLOW
        assert action.action_args == {"followee_id": 5}

    def test_invalid_followee_id(self) -> None:
        primitive = FollowPrimitive()
        with pytest.raises(ValueError, match="followee_id is required"):
            primitive.generate()


class TestSearchPrimitive:
    def test_search_posts(self) -> None:
        primitive = SearchPrimitive(query="election", search_type="posts")
        action = primitive.generate()
        assert action.action_type == ActionType.SEARCH_POSTS
        assert action.action_args == {"query": "election"}

    def test_search_user(self) -> None:
        primitive = SearchPrimitive(query="alice", search_type="user")
        action = primitive.generate()
        assert action.action_type == ActionType.SEARCH_USER
        assert action.action_args == {"query": "alice"}

    def test_refresh_feed(self) -> None:
        primitive = SearchPrimitive(search_type="refresh")
        action = primitive.generate()
        assert action.action_type == ActionType.REFRESH
        assert action.action_args == {}

    def test_invalid_search_type(self) -> None:
        primitive = SearchPrimitive(query="test", search_type="posts")
        with pytest.raises(ValueError, match="Unknown search_type"):
            primitive.generate(search_type="invalid_mode")
