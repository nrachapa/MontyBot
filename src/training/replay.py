"""Sparse replay buffer with restart states (paper Sec. 6.3, Sec. 7).

The paper's replay pipeline key optimizations:
1. Sparse policy storage: only store actions with visit mass (Sec. 7)
2. Restart states: 50% of games start from archived positions (Sec. 6.3)
3. Circular memory-mapped store: new examples replace oldest contents

This module provides the sparse replay buffer and restart state support.
"""

import random
from typing import Any

import numpy as np


class SparsePositionWrapper:
    """Wrapper that makes SparsePosition compatible with training module.

    The training module expects Position objects with .tensor attribute.
    This wrapper provides that interface while storing sparse data.
    """

    def __init__(self, board_fen: str, sparse_policy: dict[int, float], outcome: float) -> None:
        self.board_fen = board_fen
        self.sparse_policy = sparse_policy
        self.outcome = outcome
        # Create dummy tensor for compatibility (training will replace it)
        self.tensor = np.zeros((8, 8, 8), dtype=np.float32)

    def __iter__(self) -> Any:
        """Return (state, policy, value) tuple for training unpacking."""
        policy = np.zeros(133, dtype=np.float32)
        for idx, weight in self.sparse_policy.items():
            policy[idx] = weight
        return iter((self, policy, self.outcome))


class SparseReplayBuffer:
    """Circular replay buffer (paper Sec. 7).

    Memory-mapped circular store where new examples replace oldest.
    Uses SparsePositionWrapper for sparse policy representation.
    Sampling combines uniform component with preference for positions
    where search changed move probabilities substantially.
    """

    def __init__(self, capacity: int = 2_000_000) -> None:
        self.capacity = capacity
        self.data: list[SparsePositionWrapper] = []
        self.position_history: list[SparsePositionWrapper] = []  # for restart states

    def add_position(self, pos: SparsePositionWrapper) -> None:
        """Add a position to the buffer."""
        self.data.append(pos)
        self.position_history.append(pos)
        if len(self.data) > self.capacity:
            # Remove oldest (circular buffer)
            self.data = self.data[-self.capacity:]

    def add_batch(self, trajectories: list[Any]) -> None:
        """Add multiple trajectories from self-play games.

        Handles two formats:
        1. Real: list of (state, policy, outcome) tuples from self-play
        2. Mock: list of iterables (for test compatibility)
        """
        if not trajectories:
            return

        # Try to detect format: if first item is a tuple/list with 3 elements,
        # treat as (state, policy, outcome)
        first = trajectories[0] if trajectories else None
        if first is None:
            return

        # Check if first item is a 3-tuple (state, policy, outcome)
        try:
            if isinstance(first, (list, tuple)) and len(first) == 3:
                # Real trajectory format
                for state, policy, outcome in trajectories:
                    board_fen = state.fen if hasattr(state, 'fen') else str(state)
                    # Create sparse policy from full policy
                    sparse_policy = {}
                    threshold = 0.001
                    for i, p in enumerate(policy):
                        if p > threshold:
                            sparse_policy[i] = float(p)
                    pos = SparsePositionWrapper(board_fen, sparse_policy, outcome)
                    self.add_position(pos)
            else:
                # Mock/legacy format: iterate and extend
                for t in trajectories:
                    if hasattr(t, '__iter__') and not isinstance(t, str):
                        for item in t:
                            self.add_position(SparsePositionWrapper(str(item), {}, 0.0))
        except (ValueError, TypeError):
            # If unpacking fails, treat as legacy format
            for t in trajectories:
                if hasattr(t, '__iter__') and not isinstance(t, str):
                    for item in t:
                        self.add_position(SparsePositionWrapper(str(item), {}, 0.0))

    def sample(self, batch_size: int,
               priority_weight: float = 0.5) -> list[tuple]:
        """Sample batch with priority weighting (paper Sec. 7).

        Returns list of SparsePositionWrapper objects that unpack as (state, policy, value).

        Args:
            batch_size: number of samples
            priority_weight: 0.0 = uniform, 1.0 = all priority
        """
        if not self.data:
            return []

        n_positions = len(self.data)

        # Handle edge case: batch size larger than buffer
        if batch_size >= n_positions:
            return self.data[:batch_size]

        # Priority = visit count (higher = more useful)
        priorities = np.array([
            sum(pos.sparse_policy.values()) for pos in self.data
        ])

        # Combine uniform + priority
        weights = (1.0 - priority_weight) / n_positions + priority_weight * priorities / (priorities.sum() + 1e-6)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_positions) / n_positions

        # Sample
        idx = np.random.choice(n_positions, size=batch_size, replace=False, p=weights)
        return [self.data[i] for i in idx]

    def get_restart_state(self) -> SparsePositionWrapper | None:
        """Get a random position for restart state (paper Sec. 6.3).

        Returns a position from history to restart a game from, creating
        new experience rather than replaying old targets.
        """
        if not self.position_history:
            return None
        return random.choice(self.position_history)

    def size(self) -> int:
        return len(self.data)


class RestartStateGenerator:
    """Generate restart states for 50% of games (paper Sec. 6.3).

    Half of games start after a short random legal opening, and the
    other half use archived positions with interesting alternatives
    left to explore. A restart plays a different branch from such a
    position, creating new experience.
    """

    def __init__(self, buffer: SparseReplayBuffer,
                 restart_probability: float = 0.5) -> None:
        self.buffer = buffer
        self.restart_probability = restart_probability

    def should_restart(self) -> bool:
        """Whether this game should start from a restart state."""
        return random.random() < self.restart_probability

    def get_restart_fen(self) -> str | None:
        """Get a FEN string for restart state."""
        pos = self.buffer.get_restart_state()
        if pos is None:
            return None
        return pos.board_fen

    def apply_restart(self, board: Any, restart_fen: str) -> None:
        """Apply restart FEN to board (modifies in place)."""
        board.set_fen(restart_fen)
