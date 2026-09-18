"""Pytest fixtures and test harness configuration for cib_zoo."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import pytest

# Ensure repository root is on sys.path
_oasis_root = Path(__file__).resolve().parents[2]
if str(_oasis_root) not in sys.path:
    sys.path.insert(0, str(_oasis_root))

# Limit threads to max 4 to prevent host machine CPU overload
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(var, "4")

# Safe mock for heavy ML packages if running in lightweight CPU environment
for pkg in (
    "torch", "torch.nn", "sentence_transformers", "transformers",
    "sklearn", "sklearn.feature_extraction", "sklearn.feature_extraction.text",
    "sklearn.metrics", "sklearn.metrics.pairwise"
):
    try:
        __import__(pkg)
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules[pkg] = MagicMock()

from oasis.environment.env_action import ManualAction
from oasis.social_platform.channel import Channel
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.database import create_db
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType


@pytest.fixture
def tmp_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated temporary SQLite database path and override OASIS_DB_PATH."""
    db_file = tmp_path / "test_social_media.db"
    monkeypatch.setenv("OASIS_DB_PATH", str(db_file))
    create_db(str(db_file))
    return db_file


@pytest.fixture
def mock_channel() -> Channel:
    """Lightweight Channel fixture for isolated component testing without background loops."""
    return Channel()


@pytest.fixture
def mock_user_info() -> UserInfo:
    """Standardized UserInfo fixture for CIBAgent component testing."""
    return UserInfo(
        user_name="cib_test_bot",
        name="CIB Test Bot",
        description="Automated synthetic agent for CIB unit and component tests",
        profile={
            "nodes": [],
            "edges": [],
            "other_info": {
                "user_profile": "Synthetic CIB bot persona",
                "mbti": "INTJ",
                "gender": "non-binary",
                "age": 25,
                "country": "US",
                "activity_level": ["active"] * 24,
                "activity_level_frequency": [1] * 24,
                "active_threshold": [0.5] * 24,
            },
        },
        recsys_type="reddit",
        is_controllable=False,
    )


@pytest.fixture
def user_info_factory() -> Callable[..., UserInfo]:
    """Factory fixture to generate unique UserInfo instances."""

    def _create(
        agent_id: int = 1,
        user_name: str | None = None,
        name: str | None = None,
    ) -> UserInfo:
        uname = user_name or f"cib_bot_{agent_id:03d}"
        return UserInfo(
            user_name=uname,
            name=name or f"Bot {agent_id}",
            description=f"Synthetic bot {agent_id}",
            profile={
                "nodes": [],
                "edges": [],
                "other_info": {
                    "user_profile": f"Profile for {uname}",
                    "mbti": "INTP",
                    "gender": "non-binary",
                    "age": 25,
                    "country": "US",
                    "activity_level": ["active"] * 24,
                    "activity_level_frequency": [1] * 24,
                    "active_threshold": [0.5] * 24,
                },
            },
            recsys_type="reddit",
            is_controllable=False,
        )

    return _create


@pytest.fixture
def cib_agent_factory(
    mock_channel: Channel, mock_user_info: UserInfo
) -> Callable[..., Any]:
    """Factory fixture to instantiate CIBAgent with customized limits."""
    from cib_zoo.agent.cib_agent import CIBAgent

    def _create(
        agent_id: int = 1,
        max_actions_per_step: int = 1,
        total_budget: int | None = None,
        occurrence_thresholds: dict[ActionType | str, int] | None = None,
        user_info: UserInfo | None = None,
        channel: Channel | None = None,
    ) -> CIBAgent:
        return CIBAgent(
            agent_id=agent_id,
            user_info=user_info or mock_user_info,
            channel=channel or mock_channel,
            max_actions_per_step=max_actions_per_step,
            total_budget=total_budget,
            occurrence_thresholds=occurrence_thresholds,
            model=None,
        )

    return _create


@pytest.fixture
async def minimal_platform(
    tmp_db_path: Path,
    mock_channel: Channel,
) -> AsyncGenerator[Platform, None]:
    """Minimal running Platform instance backed by temporary SQLite DB."""
    platform = Platform(
        db_path=str(tmp_db_path),
        channel=mock_channel,
        show_score=False,
        recsys_type="reddit",
    )
    task = asyncio.create_task(platform.running())
    await asyncio.sleep(0.01)
    try:
        yield platform
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if hasattr(platform, "db") and platform.db:
            platform.db.close()
