"""Unit tests for temporal and adaptive modifiers."""

from cib_zoo.modifiers import PulsedWaveModifier, ThompsonSamplingBanditModifier


class TestPulsedWaveModifier:
    def test_duty_cycle_activation(self) -> None:
        # Period 10, duty cycle 0.4 -> active for steps 0, 1, 2, 3 in each cycle of 10
        modifier = PulsedWaveModifier(period=10, duty_cycle=0.4, phase_offset=0)
        assert modifier.active_steps_per_period == 4

        expected = [True, True, True, True, False, False, False, False, False, False]
        for step in range(10):
            assert modifier.is_active(step) == expected[step], f"Mismatch at step {step}"

    def test_phase_offset(self) -> None:
        # Period 5, duty 0.4 (2 steps active). Offset 2 -> active at 3, 4 (since (3+2)%5=0, (4+2)%5=1)
        modifier = PulsedWaveModifier(period=5, duty_cycle=0.4, phase_offset=2)
        # Step 0: (0+2)%5=2 >= 2 -> False
        assert not modifier.is_active(0)
        # Step 3: (3+2)%5=0 < 2 -> True
        assert modifier.is_active(3)
        # Step 4: (4+2)%5=1 < 2 -> True
        assert modifier.is_active(4)

    def test_filter_squad(self, cib_agent_factory) -> None:
        bots = [cib_agent_factory(agent_id=i) for i in range(1, 7)]
        modifier = PulsedWaveModifier(period=4, duty_cycle=0.5, rotation_cells=2)
        
        # Step 0 is active (first cell: 3 bots)
        active_step0 = modifier.filter_squad(step=0, squad_bots=bots)
        assert len(active_step0) == 3

        # Step 2 is dormant (duty 0.5 of period 4 = 2 active steps: 0, 1)
        active_step2 = modifier.filter_squad(step=2, squad_bots=bots)
        assert len(active_step2) == 0


class TestThompsonSamplingBanditModifier:
    def test_initial_state(self) -> None:
        bandit = ThompsonSamplingBanditModifier(arms=["arm_a", "arm_b"])
        stats = bandit.get_arm_stats()
        assert stats["arm_a"]["alpha"] == 1.0
        assert stats["arm_a"]["beta"] == 1.0
        assert stats["arm_a"]["expected_value"] == 0.5
        assert stats["arm_a"]["pulls"] == 0

    def test_posterior_updates(self) -> None:
        bandit = ThompsonSamplingBanditModifier(arms=["arm_a", "arm_b"])
        bandit.update("arm_a", reward=1.0)
        bandit.update("arm_a", reward=1.0)
        bandit.update("arm_b", reward=0.0)

        stats = bandit.get_arm_stats()
        assert stats["arm_a"]["alpha"] == 3.0
        assert stats["arm_a"]["beta"] == 1.0
        assert stats["arm_a"]["expected_value"] == 0.75

        assert stats["arm_b"]["alpha"] == 1.0
        assert stats["arm_b"]["beta"] == 2.0
        assert stats["arm_b"]["expected_value"] == 1.0 / 3.0

    def test_convergence_toward_optimal_arm(self) -> None:
        # Arm A has 90% true success, Arm B has 10%
        bandit = ThompsonSamplingBanditModifier(arms=["optimal", "suboptimal"], random_state=42)
        for _ in range(100):
            arm = bandit.sample_arm()
            if arm == "optimal":
                reward = 1.0 if bandit._rng.random() < 0.9 else 0.0
            else:
                reward = 1.0 if bandit._rng.random() < 0.1 else 0.0
            bandit.update(arm, reward)

        stats = bandit.get_arm_stats()
        # Optimal arm should have far more pulls and higher expected value
        assert stats["optimal"]["pulls"] > stats["suboptimal"]["pulls"]
        assert stats["optimal"]["expected_value"] > stats["suboptimal"]["expected_value"]
