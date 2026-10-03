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
import pytest

from oasis.inference.jev_classifier import (
    ClassificationResult,
    apply_l0_debias,
    dump_action_confidence_gap,
)


def test_dump_action_confidence_gap_four_actions():
    results = [
        ClassificationResult(
            user_id=1,
            post_id=101,
            action_char="R",
            confidence=0.6,
            logits={"L": 0.5, "R": 1.2, "F": -0.8, "S": -1.5},
        ),
        ClassificationResult(
            user_id=2,
            post_id=101,
            action_char="F",
            confidence=0.7,
            logits={"L": -0.2, "R": -0.5, "F": 2.0, "S": -1.0},
        ),
    ]

    rows = dump_action_confidence_gap(results)
    assert len(rows) == 2

    r0 = rows[0]
    assert "p_like" in r0
    assert "p_repost" in r0
    assert "p_follow" in r0
    assert "p_skip" in r0
    assert "gap_r_minus_l" in r0
    assert r0["r_present"] == 1.0
    assert r0["l_present"] == 1.0
    assert r0["f_present"] == 1.0
    assert r0["s_present"] == 1.0
    assert r0["p_repost"] > r0["p_like"]

    r1 = rows[1]
    assert r1["p_follow"] > r1["p_like"]
    assert r1["p_follow"] > r1["p_repost"]


def test_apply_l0_debias_damping():
    # Batch where 'L' has an artificial positive letter bias of +2.0 across all items,
    # but item 1 genuinely has high Repost intent ('R').
    results = [
        ClassificationResult(
            user_id=1,
            post_id=100,
            action_char="L",
            confidence=0.5,
            logits={"L": 2.0, "R": 1.5, "S": 0.0},
        ),
        ClassificationResult(
            user_id=2,
            post_id=101,
            action_char="L",
            confidence=0.6,
            logits={"L": 2.2, "R": 0.5, "S": 0.0},
        ),
    ]

    # At prior_strength=0.0 (no debias), L dominates both
    raw = apply_l0_debias(results, prior_strength=0.0)
    assert raw[0].action_char == "L"
    assert raw[1].action_char == "L"

    # At prior_strength=1.0 (full L0 mean subtraction), L's mean ~2.1 is removed,
    # allowing R to win on item 1 (1.5 - 1.0 = +0.5 vs 2.0 - 2.1 = -0.1)
    full_l0 = apply_l0_debias(results, prior_strength=1.0)
    assert full_l0[0].action_char == "R"

    # At prior_strength=0.75 (damped L0), R also wins on item 1
    damped_l0 = apply_l0_debias(results, prior_strength=0.75)
    assert damped_l0[0].action_char == "R"

    # Verify that debiased logits reflect the damped profile
    # Profile for L is (2.0 + 2.2) / 2 = 2.1
    # Damped offset is 0.75 * 2.1 = 1.575
    expected_debiased_l = 2.0 - 0.75 * 2.1
    assert abs(damped_l0[0].logits["L"] - expected_debiased_l) < 1e-5
