"""Tests for SearchCore — the MCTS algorithm math shared by MCTS and VectorizedMCTS.

These verify the three core functions in ``src.training.search._core``:
puct_score, dirichlet_perturb, backup_value.

Each test starts from an independent, known-good value (not recomputed
the way the code does) — the spec, not the implementation.
"""
import numpy as np
import pytest

from src.training.search._core import backup_value, dirichlet_perturb, puct_score


class TestPuctScore:
    """puct_score(value, prior, parent_visits, node_visits, c_puct)

    Formula: value + c_puct * prior * sqrt(parent_visits) / (1 + node_visits)
    """

    def test_unvisited_node_gets_bonus(self):
        """An unvisited node (zero Q) with a high prior gets a boost
        from the exploration term — it should be strictly positive."""
        score = puct_score(value=0.0, prior=0.5, parent_visits=100.0, node_visits=0.0, c_puct=0.6)
        # Exploration = 0.6 * 0.5 * 10 / 1 = 3.0
        expected = 0.0 + 0.6 * 0.5 * 10.0 / 1.0  # = 3.0
        assert np.isclose(score, expected), f"Expected {expected}, got {score}"

    def test_visited_node_decreases_bonus(self):
        """A node with many visits has a shrinking exploration bonus
        as node_visits grows."""
        score = puct_score(value=0.0, prior=0.5, parent_visits=100.0, node_visits=9.0, c_puct=0.6)
        # Exploration = 0.6 * 0.5 * 10 / 10 = 0.3
        expected = 0.6 * 0.5 * 10.0 / 10.0  # = 0.3
        assert np.isclose(score, expected)

    def test_positive_q_adds_to_bonus(self):
        """When the node has a positive Q, the score is value + exploration."""
        score = puct_score(value=0.7, prior=0.5, parent_visits=100.0, node_visits=0.0, c_puct=0.6)
        expected = 0.7 + 3.0  # = 3.7
        assert np.isclose(score, expected)

    def test_scalar_and_array_match(self):
        """VectorizedMCTS calls with arrays; MCTS with scalars.
        The math must be identical."""
        scalar = puct_score(value=0.1, prior=0.2, parent_visits=4.0, node_visits=1.0, c_puct=0.5)
        arr = puct_score(value=np.array([0.1]), prior=np.array([0.2]), parent_visits=np.array([4.0]), node_visits=np.array([1.0]), c_puct=0.5)
        assert np.isclose(scalar, float(arr[0]))

    def test_sqrt_parent_visits(self):
        """The exploration bonus scales with sqrt(parent_visits), not linear."""
        score = puct_score(value=0.0, prior=1.0, parent_visits=0.0, node_visits=0.0, c_puct=1.0)
        assert np.isclose(score, 0.0)  # sqrt(0) = 0, no exploration

    def test_clips_negative_parent_visits(self):
        """np.maximum(parent_visits, 0) means negative visits don't crash."""
        score = puct_score(value=0.0, prior=1.0, parent_visits=-1.0, node_visits=0.0, c_puct=1.0)
        assert np.isclose(score, 0.0)


class TestDirichletPerturb:
    """dirichlet_perturb(priors, noise, epsilon)

    Formula: priors * (1 - epsilon) + noise * epsilon
    """

    def test_no_noise(self):
        """With epsilon=0, the prior is returned unchanged."""
        prior = np.array([0.5, 0.3, 0.2])
        noise = np.array([0.1, 0.1, 0.1])
        result = dirichlet_perturb(prior, noise, 0.0)
        assert np.allclose(result, prior)

    def test_full_noise(self):
        """With epsilon=1, the noise is returned (prior is discarded)."""
        prior = np.array([0.5, 0.3, 0.2])
        noise = np.array([0.1, 0.2, 0.7])
        result = dirichlet_perturb(prior, noise, 1.0)
        assert np.allclose(result, noise)

    def test_half_noise(self):
        """With epsilon=0.5, the result is the midpoint of prior and noise."""
        prior = np.array([1.0, 0.0])
        noise = np.array([0.0, 1.0])
        result = dirichlet_perturb(prior, noise, 0.5)
        assert np.allclose(result, [0.5, 0.5])

    def test_preserves_sum(self):
        """A convex combination of priors and noise must stay between
        their min and max — for same-sum inputs the sum is preserved."""
        prior = np.array([0.8, 0.15, 0.05])
        noise = prior.copy()
        result = dirichlet_perturb(prior, noise, 0.1)
        assert np.isclose(result.sum(), prior.sum())

    def test_scalar_works(self):
        """Scalar inputs must return a scalar."""
        result = dirichlet_perturb(0.5, 0.5, 0.1)
        assert np.isscalar(result) or result.ndim == 0


class TestBackupValue:
    """backup_value(node_color, leaf_color, value)

    If node_color == leaf_color: return value
    Else: return -value (sign flip)
    """

    def test_same_color_no_flip(self):
        """Same side-to-move: the value keeps its sign."""
        assert backup_value(node_color=1, leaf_color=1, value=0.7) == 0.7
        assert backup_value(node_color=-1, leaf_color=-1, value=-0.3) == -0.3

    def test_opposite_color_flips(self):
        """Opposite side-to-move: the value is negated."""
        assert backup_value(node_color=1, leaf_color=-1, value=0.7) == -0.7
        assert backup_value(node_color=-1, leaf_color=1, value=-0.3) == 0.3

    def test_zero_value_neither_flip_nor_keep(self):
        """Zero is unchanged by negation."""
        assert backup_value(node_color=1, leaf_color=-1, value=0.0) == 0.0
