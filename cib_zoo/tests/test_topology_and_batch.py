"""Tests for network topology generation and batch experiment analytics."""

import json

import pytest

from cib_zoo.runner.aggregate_batch_results import (
    generate_summary_data,
    load_batch_runs,
)
from cib_zoo.topology.network_builder import (
    build_multitopic_network,
    build_network,
    build_polarized_network,
)


def test_build_multitopic_network():
    """Verify multi-topic network construction with 150 agents partitioned across 3 communities."""
    _graph, agents, comm_map = build_multitopic_network(num_users=30)
    assert len(agents) == 30
    assert len(comm_map) == 30

    tech_count = sum(1 for c in comm_map.values() if c == "tech")
    pol_count = sum(1 for c in comm_map.values() if c == "politics")
    sports_count = sum(1 for c in comm_map.values() if c == "sports")

    assert tech_count == 10
    assert pol_count == 10
    assert sports_count == 10

    # Verify agent properties
    first_agent = agents[0]
    assert first_agent.social_agent_id == 1
    assert "tech" in first_agent.user_info.profile["other_info"]["community"]
    assert first_agent.user_info.recsys_type == "twitter"


def test_build_polarized_network():
    """Verify polarized network construction with progressive and conservative agents."""
    _graph, agents, comm_map = build_polarized_network(num_users=20)
    assert len(agents) == 20
    assert len(comm_map) == 20

    prog_count = sum(1 for c in comm_map.values() if c == "progressive")
    cons_count = sum(1 for c in comm_map.values() if c == "conservative")

    assert prog_count == 10
    assert cons_count == 10

    # Verify progressive vs conservative agent profiles
    first_prog = agents[0]
    first_cons = agents[10]
    assert comm_map[first_prog.social_agent_id] == "progressive"
    assert comm_map[first_cons.social_agent_id] == "conservative"
    assert "progressive" in first_prog.user_info.profile["other_info"]["community"]
    assert "conservative" in first_cons.user_info.profile["other_info"]["community"]


def test_build_network_factory_router():
    """Verify network factory router and exception on invalid topology."""
    _, a_multi, _ = build_network("multitopic", num_users=15)
    assert len(a_multi) == 15

    _, a_pol, _ = build_network("polarized", num_users=10)
    assert len(a_pol) == 10

    with pytest.raises(ValueError, match="Unknown topology 'invalid_topo'"):
        build_network("invalid_topo", num_users=10)


def test_aggregate_batch_results_summary(tmp_path):
    """Verify batch aggregator compilation on mock run JSONs."""
    mock_run_1 = {
        "status": "success",
        "preset": "s1",
        "campaign_name": "S1_MultiChannelRetrievalPoisoning",
        "topology": "multitopic",
        "topic_mode": "existing",
        "num_organic": 150,
        "num_bots": 15,
        "bot_ratio": 0.10,
        "exposure_baseline": 100.0,
        "exposure_payload": 250.0,
        "differential_amplification": 10.0,
        "telemetry": {
            "total_posts": 20,
            "total_likes": 80,
            "total_comments": 15,
            "total_bot_actions": 45,
        },
        "community_telemetry": {
            "tech": {"payload_impressions": 120, "payload_likes": 35},
            "politics": {"payload_impressions": 80, "payload_likes": 15},
            "sports": {"payload_impressions": 50, "payload_likes": 5},
        },
    }

    mock_run_2 = {
        "status": "success",
        "preset": "baseline",
        "campaign_name": "OrganicControlBaseline",
        "topology": "polarized",
        "topic_mode": "existing",
        "num_organic": 150,
        "num_bots": 0,
        "bot_ratio": 0.0,
        "exposure_baseline": 120.0,
        "exposure_payload": 115.0,
        "differential_amplification": 0.0,
        "telemetry": {
            "total_posts": 10,
            "total_likes": 40,
            "total_comments": 5,
            "total_bot_actions": 0,
        },
        "community_telemetry": {
            "progressive": {"payload_impressions": 60, "payload_likes": 10},
            "conservative": {"payload_impressions": 55, "payload_likes": 8},
        },
    }

    run1_file = tmp_path / "run_R01_results.json"
    run2_file = tmp_path / "run_R02_results.json"
    with open(run1_file, "w") as f:
        json.dump(mock_run_1, f)
    with open(run2_file, "w") as f:
        json.dump(mock_run_2, f)

    loaded = load_batch_runs(tmp_path)
    assert len(loaded) == 2

    summary = generate_summary_data(loaded)
    assert summary["total_runs"] == 2
    assert "s1" in summary["by_preset"]
    assert "baseline" in summary["by_preset"]
    assert "multitopic" in summary["by_topology"]
    assert "polarized" in summary["by_topology"]

    # Verify net lift calculation
    s1_summary = summary["by_preset"]["s1"][0]
    assert s1_summary["net_lift"] == 150.0
    assert pytest.approx(s1_summary["cost_efficiency_per_action"], 0.01) == 150.0 / 45
