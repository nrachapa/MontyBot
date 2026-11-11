"""Tests for sparse replay buffer and restart states (paper Sec. 6.3, Sec. 7)."""

import random

import numpy as np
import pytest

from src.training.replay import RestartStateGenerator, SparsePositionWrapper, SparseReplayBuffer


class TestSparsePositionWrapper:
    """Test sparse position wrapper creation and serialization."""

    def test_create_sparse_position(self):
        """Test creating a sparse position with visit mass."""
        pos = SparsePositionWrapper(
            board_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            sparse_policy={0: 0.5, 5: 0.3, 10: 0.2},
            outcome=1.0,
        )
        assert pos.board_fen.startswith("rnbq")
        assert pos.sparse_policy == {0: 0.5, 5: 0.3, 10: 0.2}
        assert pos.outcome == 1.0

    def test_unpack_tuple(self):
        """Test unpacking as (state, policy, value) tuple."""
        pos = SparsePositionWrapper(
            board_fen="test",
            sparse_policy={0: 0.5, 1: 0.5},
            outcome=0.0,
        )
        state, policy, value = pos
        assert state == pos
        assert len(policy) == 133
        assert policy[0] == pytest.approx(0.5)
        assert policy[1] == pytest.approx(0.5)
        assert value == 0.0


class TestSparseReplayBuffer:
    """Test sparse replay buffer operations."""

    def test_add_and_sample(self):
        """Test adding positions and sampling."""
        buffer = SparseReplayBuffer(capacity=1000)

        # Add 100 positions
        for i in range(100):
            pos = SparsePositionWrapper(
                board_fen=f"board_{i}",
                sparse_policy={i: 1.0},
                outcome=0.0,
            )
            buffer.add_position(pos)

        assert buffer.size() == 100

        # Sample batch
        batch = buffer.sample(10)
        assert len(batch) == 10

    def test_circular_buffer(self):
        """Test that buffer maintains capacity."""
        buffer = SparseReplayBuffer(capacity=50)

        # Add 100 positions
        for i in range(100):
            pos = SparsePositionWrapper(
                board_fen=f"board_{i}",
                sparse_policy={i: 1.0},
                outcome=0.0,
            )
            buffer.add_position(pos)

        assert buffer.size() == 50  # capacity enforced

    def test_priority_sampling(self):
        """Test that priority sampling works correctly."""
        buffer = SparseReplayBuffer(capacity=1000)

        # Add positions with varying visit counts
        for i in range(100):
            visit_count = i * 10  # higher index = more visits
            sparse_policy = {i: float(visit_count)}
            pos = SparsePositionWrapper(
                board_fen=f"board_{i}",
                sparse_policy=sparse_policy,
                outcome=0.0,
            )
            buffer.add_position(pos)

        # Sample with priority_weight=1.0 (all priority)
        batch = buffer.sample(10, priority_weight=1.0)
        assert len(batch) == 10

        # Positions with higher visits should be more likely to be sampled
        # (this is probabilistic, so we just check it works)
        for pos in batch:
            assert sum(pos.sparse_policy.values()) > 0

    def test_empty_buffer_sample(self):
        """Test sampling from empty buffer."""
        buffer = SparseReplayBuffer(capacity=1000)
        batch = buffer.sample(10)
        assert batch == []


class TestRestartStateGenerator:
    """Test restart state generation (paper Sec. 6.3)."""

    def test_restart_probability(self):
        """Test restart probability check."""
        buffer = SparseReplayBuffer(capacity=1000)
        generator = RestartStateGenerator(buffer, restart_probability=0.5)

        # Should return False at least once with p=0.5
        results = [generator.should_restart() for _ in range(100)]
        # Approximately half should be True
        true_count = sum(results)
        assert 20 < true_count < 80  # generous bounds for random

    def test_no_restart_when_disabled(self):
        """Test that restart is disabled when probability=0."""
        buffer = SparseReplayBuffer(capacity=1000)
        generator = RestartStateGenerator(buffer, restart_probability=0.0)

        results = [generator.should_restart() for _ in range(100)]
        assert all(not r for r in results)

    def test_restart_fen_when_empty(self):
        """Test restart FEN when buffer is empty."""
        buffer = SparseReplayBuffer(capacity=1000)
        generator = RestartStateGenerator(buffer, restart_probability=0.5)
        fen = generator.get_restart_fen()
        assert fen is None  # empty buffer returns None


class TestIntegration:
    """Integration tests for sparse replay + self-play pipeline."""

    def test_batch_add_and_sample(self):
        """Test adding trajectories and sampling for training."""
        buffer = SparseReplayBuffer(capacity=10000)

        # Simulate self-play trajectories
        for i in range(10):
            # Each trajectory has 50 positions
            for j in range(50):
                pos = SparsePositionWrapper(
                    board_fen=f"fen_{i}_{j}",
                    sparse_policy={j: 0.1},
                    outcome=0.0,
                )
                buffer.add_position(pos)

        # Verify buffer has positions
        assert buffer.size() == 500

        # Sample and verify we get valid positions
        batch = buffer.sample(50)
        assert len(batch) == 50

        # Verify each position is valid
        for pos in batch:
            assert len(pos.sparse_policy) > 0
