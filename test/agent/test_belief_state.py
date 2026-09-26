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
"""Comprehensive unit tests for Track 2: BeliefState and ActionLogItem."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

from oasis.social_agent.belief_state import (
    LABEL_NEUTRAL,
    LABEL_SKEPTICAL,
    LABEL_STRONGLY_OPPOSED,
    LABEL_STRONGLY_SUPPORTIVE,
    LABEL_SUPPORTIVE,
    ActionLogItem,
    BeliefState,
)
from oasis.social_platform.typing import ActionType

# ============================================================================
# 1. ActionLogItem Tests
# ============================================================================


class TestActionLogItem:
    """Tests for ActionLogItem data representation and serialization."""

    def test_initialization(self) -> None:
        item = ActionLogItem(
            action_type="like",
            post_id=42,
            topic="tech",
            timestamp_iso="2026-09-26T12:00:00Z",
        )
        assert item.action_type == "like"
        assert item.post_id == 42
        assert item.topic == "tech"
        assert item.timestamp_iso == "2026-09-26T12:00:00Z"

    def test_serialization_roundtrip(self) -> None:
        item = ActionLogItem(
            action_type="comment",
            post_id=99,
            topic="#politics",
            timestamp_iso="2026-09-26T15:30:00Z",
        )
        data = item.to_dict()
        assert data == {
            "action_type": "comment",
            "post_id": 99,
            "topic": "#politics",
            "timestamp_iso": "2026-09-26T15:30:00Z",
        }
        reconstructed = ActionLogItem.from_dict(data)
        assert reconstructed == item


# ============================================================================
# 2. Stance Tracking & Label Tests
# ============================================================================


class TestBeliefStateStances:
    """Tests for stance initialization, bounds, clamping, and label mapping."""

    def test_default_stance_is_zero(self) -> None:
        state = BeliefState(user_id=1)
        assert state.get_stance("ai_safety") == 0.0
        assert state.get_stance_label("ai_safety") == LABEL_NEUTRAL
        assert state.get_stance_label("ai_safety") == "Neutral / Undecided"

    def test_clamping_on_initialization(self) -> None:
        state = BeliefState(
            user_id=1,
            stances={"extreme_pos": 2.5, "extreme_neg": -3.0, "normal": 0.5},
        )
        assert state.get_stance("extreme_pos") == 1.0
        assert state.get_stance("extreme_neg") == -1.0
        assert state.get_stance("normal") == 0.5

    def test_set_stance_clamping(self) -> None:
        state = BeliefState(user_id=1)
        assert state.set_stance("topic_a", 1.8) == 1.0
        assert state.get_stance("topic_a") == 1.0

        assert state.set_stance("topic_b", -2.4) == -1.0
        assert state.get_stance("topic_b") == -1.0

        assert state.set_stance("topic_c", 0.25) == 0.25
        assert state.get_stance("topic_c") == 0.25

    @pytest.mark.parametrize(
        ("score", "expected_label"),
        [
            # s > 0.3 -> Strongly Supportive
            (1.0, LABEL_STRONGLY_SUPPORTIVE),
            (0.8, LABEL_STRONGLY_SUPPORTIVE),
            (0.31, LABEL_STRONGLY_SUPPORTIVE),
            (0.30001, LABEL_STRONGLY_SUPPORTIVE),
            # 0.1 < s <= 0.3 -> Supportive
            (0.3, LABEL_SUPPORTIVE),
            (0.25, LABEL_SUPPORTIVE),
            (0.2, LABEL_SUPPORTIVE),
            (0.11, LABEL_SUPPORTIVE),
            (0.10001, LABEL_SUPPORTIVE),
            # -0.1 <= s <= 0.1 -> Neutral / Undecided
            (0.1, LABEL_NEUTRAL),
            (0.05, LABEL_NEUTRAL),
            (0.0, LABEL_NEUTRAL),
            (-0.05, LABEL_NEUTRAL),
            (-0.1, LABEL_NEUTRAL),
            # -0.3 <= s < -0.1 -> Skeptical
            (-0.10001, LABEL_SKEPTICAL),
            (-0.15, LABEL_SKEPTICAL),
            (-0.2, LABEL_SKEPTICAL),
            (-0.3, LABEL_SKEPTICAL),
            # s < -0.3 -> Strongly Opposed
            (-0.30001, LABEL_STRONGLY_OPPOSED),
            (-0.5, LABEL_STRONGLY_OPPOSED),
            (-0.8, LABEL_STRONGLY_OPPOSED),
            (-1.0, LABEL_STRONGLY_OPPOSED),
        ],
    )
    def test_stance_label_threshold_intervals(
        self, score: float, expected_label: str
    ) -> None:
        state = BeliefState(user_id=1, stances={"test_topic": score})
        assert state.get_stance_label("test_topic") == expected_label
        # Direct static/classmethod mapping verification
        assert BeliefState.label_from_score(score) == expected_label


# ============================================================================
# 3. Bounded Confidence Stance Update Tests
# ============================================================================


class TestBeliefStateUpdateStance:
    """Tests for bounded confidence opinion dynamics math:

    delta_s = alpha * sign(delta) * min(|delta|, delta_max) * omega_peer.
    """

    def test_update_math_zero_engagement(self) -> None:
        # Initial stance = 0.0, post_stance = 1.0, engagement = 0
        # delta = 1.0, sign = 1.0, |delta| = 1.0, min(|delta|, 1.0) = 1.0
        # omega_peer = sigmoid(0) = 0.5
        # alpha = 0.15
        # delta_s = 0.15 * 1.0 * 1.0 * 0.5 = 0.075
        # new_stance = 0.0 + 0.075 = 0.075
        state = BeliefState(user_id=1, persuasion_rate_alpha=0.15, delta_max=1.0)
        new_s = state.update_stance(topic="clean_energy", post_stance=1.0, peer_engagement=0)
        assert pytest.approx(new_s, rel=1e-5) == 0.075
        assert pytest.approx(state.get_stance("clean_energy"), rel=1e-5) == 0.075

    def test_social_proof_amplification(self) -> None:
        # Higher peer engagement should increase persuasion pull (omega_peer is larger)
        state_low = BeliefState(user_id=1, persuasion_rate_alpha=0.15)
        s_low = state_low.update_stance("clean_energy", post_stance=1.0, peer_engagement=0)

        state_high = BeliefState(user_id=2, persuasion_rate_alpha=0.15)
        s_high = state_high.update_stance("clean_energy", post_stance=1.0, peer_engagement=10)

        assert s_high > s_low
        # sigmoid(10) ~ 0.9999546, so delta_s ~ 0.15 * 1.0 * 1.0 * 0.9999546 ~ 0.14999
        expected_high = 0.15 * (1.0 / (1.0 + math.exp(-10)))
        assert pytest.approx(s_high, rel=1e-4) == expected_high

    def test_negative_stance_drift(self) -> None:
        # Starting at 0.5, exposed to post with -0.5 and explicit omega_peer = 1.0
        # delta = -0.5 - 0.5 = -1.0, sign = -1.0, min(1.0, 1.0) = 1.0
        # delta_s = 0.2 * (-1.0) * 1.0 * 1.0 = -0.2
        # new_s = 0.5 - 0.2 = 0.3
        state = BeliefState(
            user_id=1,
            stances={"cybersecurity": 0.5},
            persuasion_rate_alpha=0.2,
            delta_max=1.0,
        )
        new_s = state.update_stance(
            topic="cybersecurity",
            post_stance=-0.5,
            peer_engagement=5,
            omega_peer=1.0,
        )
        assert pytest.approx(new_s, rel=1e-5) == 0.3
        assert state.get_stance_label("cybersecurity") == LABEL_SUPPORTIVE

    def test_bounded_step_delta_max_constraint(self) -> None:
        # Test that delta_max caps extreme opinion jumps
        # User is at -0.8, post is at +0.8. Total delta = 1.6.
        # With delta_max = 0.5:
        # bounded_step = min(1.6, 0.5) = 0.5
        # delta_s = 0.2 * 1.0 * 0.5 * 1.0 = 0.1
        state = BeliefState(
            user_id=1,
            stances={"ai": -0.8},
            persuasion_rate_alpha=0.2,
            delta_max=0.5,
        )
        new_s = state.update_stance(
            topic="ai",
            post_stance=0.8,
            peer_engagement=10,
            omega_peer=1.0,
        )
        # -0.8 + 0.1 = -0.7
        assert pytest.approx(new_s, rel=1e-5) == -0.7

    def test_zero_delta_does_not_mutate_state(self) -> None:
        state = BeliefState(user_id=1, stances={"ai": 0.4})
        new_s = state.update_stance("ai", post_stance=0.4, peer_engagement=100)
        assert new_s == 0.4
        assert state.get_stance("ai") == 0.4

    def test_boundary_clamping_at_plus_minus_one(self) -> None:
        # Starting at 0.8, pull of delta_s = 0.4 should exceed 1.0 and clamp at 1.0
        # delta = 1.0 - 0.8 = 0.2; with alpha=3.0, delta_s = 3.0 * 0.2 = 0.6 -> 0.8 + 0.6 = 1.4 -> clamped to 1.0
        state = BeliefState(user_id=1, stances={"ai": 0.8}, persuasion_rate_alpha=3.0)
        new_s = state.update_stance("ai", post_stance=1.0, peer_engagement=10, omega_peer=1.0)
        assert new_s == 1.0
        assert state.get_stance("ai") == 1.0

        # Starting at -0.8, pull with alpha=3.0 should drop below -1.0 and clamp at -1.0
        # delta = -1.0 - (-0.8) = -0.2; delta_s = 3.0 * (-0.2) = -0.6 -> -0.8 - 0.6 = -1.4 -> clamped to -1.0
        state.set_stance("ai", -0.8)
        new_s = state.update_stance("ai", post_stance=-1.0, peer_engagement=10, omega_peer=1.0)
        assert new_s == -1.0
        assert state.get_stance("ai") == -1.0


# ============================================================================
# 4. Level 2 Episodic Action Log & Summary Tests
# ============================================================================


class TestBeliefStateEpisodicMemory:
    """Tests for Level 2 episodic action recording and single-line prompt summary."""

    def test_empty_memory_summary_is_empty_string(self) -> None:
        state = BeliefState(user_id=1)
        assert state.get_episodic_summary() == ""
        assert len(state.recent_actions) == 0

    def test_record_action_rolling_window(self) -> None:
        state = BeliefState(user_id=1, max_history_items=3)

        state.record_action("like", 1, "topic1", "2026-09-26T10:00:00Z")
        state.record_action("like", 2, "topic2", "2026-09-26T10:01:00Z")
        state.record_action("comment", 3, "topic3", "2026-09-26T10:02:00Z")

        assert len(state.recent_actions) == 3
        assert [a.post_id for a in state.recent_actions] == [1, 2, 3]

        # 4th action pushes out 1st action (FIFO)
        state.record_action("repost", 4, "topic4", "2026-09-26T10:03:00Z")
        assert len(state.recent_actions) == 3
        assert [a.post_id for a in state.recent_actions] == [2, 3, 4]

    def test_support_for_action_type_enum(self) -> None:
        state = BeliefState(user_id=1)
        state.record_action(ActionType.LIKE_POST, 10, "tech", "2026-09-26T10:00:00Z")
        state.record_action(ActionType.CREATE_COMMENT, 20, "", "2026-09-26T10:01:00Z")
        state.record_action(ActionType.REPOST, 30, "#target", "2026-09-26T10:02:00Z")

        summary = state.get_episodic_summary()
        assert summary == "Recent: Liked P10 (#tech), Commented on P20, Reposted P30 (#target)"

    def test_plan_spec_exact_episodic_summary_match(self) -> None:
        """Requirement R2/Section 4.4:

        'Recent: Liked P12 (#tech), Liked P42 (#target), Commented on P42'
        """
        state = BeliefState(user_id=1)
        state.record_action("like", 12, "tech", "2026-09-26T10:00:00Z")
        state.record_action("like", 42, "#target", "2026-09-26T10:01:00Z")
        state.record_action("comment", 42, "", "2026-09-26T10:02:00Z")

        summary = state.get_episodic_summary()
        expected = "Recent: Liked P12 (#tech), Liked P42 (#target), Commented on P42"
        assert summary == expected

    def test_clear_history(self) -> None:
        state = BeliefState(user_id=1)
        state.record_action("like", 1, "tech", "2026-09-26T10:00:00Z")
        assert len(state.recent_actions) == 1
        state.clear_history()
        assert len(state.recent_actions) == 0
        assert state.get_episodic_summary() == ""


# ============================================================================
# 5. Serialization & Roundtrip Tests
# ============================================================================


class TestBeliefStateSerialization:
    """Tests for to_dict, from_dict, and state reproduction."""

    def test_serialization_roundtrip(self) -> None:
        state = BeliefState(
            user_id=42,
            stances={"tech": 0.45, "politics": -0.6},
            max_history_items=5,
            persuasion_rate_alpha=0.18,
            delta_max=0.8,
        )
        state.record_action("like", 10, "tech", "2026-09-26T10:00:00Z")
        state.record_action("comment", 15, "politics", "2026-09-26T10:05:00Z")

        data = state.to_dict()
        reconstructed = BeliefState.from_dict(data)

        assert reconstructed.user_id == 42
        assert reconstructed.stances == {"tech": 0.45, "politics": -0.6}
        assert reconstructed.max_history_items == 5
        assert reconstructed.persuasion_rate_alpha == 0.18
        assert reconstructed.delta_max == 0.8
        assert len(reconstructed.recent_actions) == 2
        assert reconstructed.recent_actions[0].action_type == "like"
        assert reconstructed.recent_actions[0].post_id == 10
        assert reconstructed.get_episodic_summary() == state.get_episodic_summary()


# ============================================================================
# 6. AST & Zero SQLite Guardrails
# ============================================================================


class TestBeliefStateGuardrails:
    """Ensures zero sqlite3 imports, proper typing, and docstrings."""

    def test_zero_sqlite_imports(self) -> None:
        file_path = (
            Path(__file__).parents[2]
            / "oasis"
            / "social_agent"
            / "belief_state.py"
        )
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "sqlite" not in alias.name.lower(), (
                        f"Prohibited import found: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "sqlite" not in module.lower(), (
                    f"Prohibited from-import found: {module}"
                )

    def test_public_methods_have_docstrings_and_annotations(self) -> None:
        for method_name in [
            "get_stance",
            "set_stance",
            "get_stance_label",
            "update_stance",
            "record_action",
            "get_episodic_summary",
        ]:
            method = getattr(BeliefState, method_name)
            assert method.__doc__ is not None and len(method.__doc__.strip()) > 0, (
                f"Method {method_name} missing docstring"
            )
            annotations = inspect.get_annotations(method)
            assert "return" in annotations, (
                f"Method {method_name} missing return type annotation"
            )
