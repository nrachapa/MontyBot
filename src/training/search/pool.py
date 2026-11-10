"""NodePool — flat node storage with a view object for MCTS traversal.

ADR-0007 established a flat numpy pool (one contiguous node table) as the
node storage for ``VectorizedMCTS``.  The pool owns all the arrays and
lifecycle methods (allocation, growth, reset, serialization, node setup);
``FlatNode`` is a lightweight view that lets the search loop read individual
nodes through indexed property access.

The pool grows by doubling as a game's tree outgrows it. Slots are retained
for subtree reuse (the tree structure is never compacted).

``__init__.py`` re-exports ``FlatNode`` and ``NodePool`` so importers are
unaffected by the split.
"""

from typing import Any

import numpy as np

from ..game import MoveMapper


class NodePool:
    """Flat node pool backed by aligned numpy arrays.

    All arrays share a common capacity; the pool grows by doubling when
    needed.  Node layout:

        parent      – parent node index (−1 for root / orphan)
        first_child – first child slot index (−1 if leaf)
        n_children  – number of children (0 for leaf)
        actions     – action index at each child slot
        legal       – (capacity, action_size) float: legal action mask
        priors      – (capacity, action_size) float: root policy output
        terminal    – bool: terminal node flag
        winner      – int: outcome (+1/−1/0)
        colors      – int: side-to-move color (+1/−1)
        value_sum   – total backed-up value (for Q computation)
        visits      – visit count
        boards      – python-chess Board at each node (for expansion)
        fens        – FEN string at each node (for subtree reuse cache)
        tensors     – encoded 8-plane input tensor at each node
    """

    __slots__ = (
        "capacity", "action_size", "legal", "priors", "terminal", "winner",
        "colors", "parents", "first_child", "n_children", "actions",
        "value_sum", "visits", "boards", "fens", "tensors",
    )

    def __init__(self, capacity: int = 50_000) -> None:
        a = MoveMapper.ACTION_SIZE
        self.capacity = capacity
        self.action_size = a
        self.legal = np.zeros((capacity, a), dtype=np.float32)
        self.priors = np.zeros((capacity, a), dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=np.int8)
        self.winner = np.zeros(capacity, dtype=np.int8)
        self.colors = np.zeros(capacity, dtype=np.int8)
        self.parents = np.full(capacity, -1, dtype=np.int64)
        self.first_child = np.full(capacity, -1, dtype=np.int64)
        self.n_children = np.zeros(capacity, dtype=np.int64)
        self.actions = np.full(capacity, -1, dtype=np.int64)
        self.value_sum = np.zeros(capacity, dtype=np.float32)
        self.visits = np.zeros(capacity, dtype=np.int64)
        self.boards: list = [None] * capacity
        self.fens: list = [""] * capacity
        self.tensors: list = [None] * capacity

    # ---------------------------------------------------------- allocation

    @classmethod
    def allocate(cls, capacity: int) -> "NodePool":
        """Factory: create and return a pool with the given capacity."""
        return cls(capacity)

    def grow(self, need: int) -> None:
        """Grow the pool if *need* exceeds the current capacity (doubles)."""
        if need <= self.capacity:
            return
        cap = max(need, 2 * self.capacity)
        add = cap - self.capacity
        a = self.action_size
        self.legal = np.vstack([self.legal, np.zeros((add, a), dtype=np.float32)])
        self.priors = np.vstack([self.priors, np.zeros((add, a), dtype=np.float32)])
        self.terminal = np.concatenate([self.terminal, np.zeros(add, dtype=np.int8)])
        self.winner = np.concatenate([self.winner, np.zeros(add, dtype=np.int8)])
        self.colors = np.concatenate([self.colors, np.zeros(add, dtype=np.int8)])
        self.parents = np.concatenate([self.parents, np.full(add, -1, dtype=np.int64)])
        self.first_child = np.concatenate([self.first_child, np.full(add, -1, dtype=np.int64)])
        self.n_children = np.concatenate([self.n_children, np.zeros(add, dtype=np.int64)])
        self.actions = np.concatenate([self.actions, np.full(add, -1, dtype=np.int64)])
        self.value_sum = np.concatenate([self.value_sum, np.zeros(add, dtype=np.float32)])
        self.visits = np.concatenate([self.visits, np.zeros(add, dtype=np.int64)])
        self.boards += [None] * add
        self.fens += [""] * add
        self.tensors += [None] * add
        self.capacity = cap

    # ---------------------------------------------------------- node access

    def get(self, idx: int) -> "FlatNode":
        """Return the ``FlatNode`` view into node at *idx*."""
        return FlatNode(idx, self)

    def set_legal(self, idx: int, actions: np.ndarray) -> None:
        """Mark *actions* as legal at node *idx*.

        ``actions`` is a 1-d array of action indices (e.g. from ``np.nonzero``).
        """
        self.legal[idx, actions] = 1.0

    def set_terminal(self, idx: int, winner: int) -> None:
        """Mark node *idx* as terminal with outcome *winner* (+1/−1/0)."""
        self.terminal[idx] = 1
        self.winner[idx] = winner

    def children(self, idx: int) -> dict[int, "FlatNode"]:
        """Return children as {action_index -> FlatNode}."""
        lo = int(self.first_child[idx])
        n = int(self.n_children[idx])
        if n <= 0:
            return {}
        return {int(self.actions[lo + j]): FlatNode(lo + j, self) for j in range(n)}

    def first_child_idx(self, idx: int) -> int:
        """Index of the first child slot, or −1 if the node is a leaf."""
        return int(self.first_child[idx])

    def n_children_count(self, idx: int) -> int:
        """Number of children of node *idx*."""
        return int(self.n_children[idx])

    def node_actions(self, idx: int) -> np.ndarray:
        """Return the full legal mask for node *idx*.

        Note: the pool also exposes ``self.actions`` (the child-action
        mapping array) directly — this method returns the node's legal mask.
        """
        return self.legal[idx].copy()

    def node_visits(self, idx: int) -> int:
        """Visit count for node *idx*."""
        return int(self.visits[idx])

    def node_value_sum(self, idx: int) -> float:
        """Backed-up value sum for node *idx*."""
        return float(self.value_sum[idx])

    def node_colors(self, idx: int) -> int:
        """Side-to-move color (+1/−1) for node *idx*."""
        return int(self.colors[idx])

    def node_prior(self, idx: int) -> np.ndarray:
        """Prior distribution for node *idx* (root policy output)."""
        return self.priors[idx].copy()

    def node_tensor(self, idx: int) -> np.ndarray:
        """Return the 8-plane input tensor for node *idx* (for net calls)."""
        return self.tensors[idx]

    @property
    def size(self) -> int:
        """Pool capacity (the pool owns the capacity; VectorizedMCTS
        tracks the actual used count with its own ``size``)."""
        return self.capacity

    # ---------------------------------------------------------- node lifecycle

    def setup_node(
        self,
        idx: int,
        color: int,
        board: Any,
        tensor: np.ndarray,
        game: Any,
    ) -> None:
        """Fully set up a node slot.

        Sets up the board reference, tensor, color, legal mask, and
        terminal state.  Called by ``VectorizedMCTS._setup``.
        """
        self.boards[idx] = board
        self.fens[idx] = board.fen()
        self.tensors[idx] = tensor
        self.colors[idx] = color
        self.priors[idx] = 0.0
        self.legal[idx] = 0.0
        self.value_sum[idx] = 0.0
        self.visits[idx] = 0
        self.first_child[idx] = -1
        self.n_children[idx] = 0

        # Deferred draw (the proto's finding): claim_draw walks the whole
        # history per node; the 150-halfmove cap gives the same winner=0.
        if board.is_game_over(claim_draw=False) or board.halfmove_clock >= game.HALFMOVE_CAP:
            self.terminal[idx] = 1
            self.winner[idx] = self._compute_winner(board)
            self.value_sum[idx] = self.winner[idx] * color
            return

        self.winner[idx] = 0
        # Wrap board in Position so game.get_legal_moves can read .board
        from ..game import Position
        st = Position(board, color, board.fen(), tensor)
        acts = game.get_legal_moves(st)
        if acts:
            self.legal[idx, acts] = 1.0

    @staticmethod
    def _compute_winner(board: Any) -> int:
        """Compute the winner (+1/−1/0) from the board state."""
        if board.halfmove_clock >= 150:
            return 0
        r = board.result(claim_draw=False)
        if r == "1-0":
            return 1
        if r == "0-1":
            return -1
        return 0

    # ---------------------------------------------------------- serialization

    def __getstate__(self) -> dict:
        """Strip the pool arrays: workers allocate their own.

        The pool carries no state a fresh game needs (the first search
        always starts from a FEN-mismatching or absent root), so this is
        safe for the self-play pickling path.
        """
        state = self.__dict__.copy()
        for key in ("legal", "priors", "terminal", "winner", "colors", "parents",
                    "first_child", "n_children", "actions", "value_sum", "visits",
                    "boards", "fens", "tensors"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Re-allocate the arrays with the stored capacity
        a = self.action_size
        cap = self.capacity
        self.legal = np.zeros((cap, a), dtype=np.float32)
        self.priors = np.zeros((cap, a), dtype=np.float32)
        self.terminal = np.zeros(cap, dtype=np.int8)
        self.winner = np.zeros(cap, dtype=np.int8)
        self.colors = np.zeros(cap, dtype=np.int8)
        self.parents = np.full(cap, -1, dtype=np.int64)
        self.first_child = np.full(cap, -1, dtype=np.int64)
        self.n_children = np.zeros(cap, dtype=np.int64)
        self.actions = np.full(cap, -1, dtype=np.int64)
        self.value_sum = np.zeros(cap, dtype=np.float32)
        self.visits = np.zeros(cap, dtype=np.int64)
        self.boards = [None] * cap
        self.fens = [""] * cap
        self.tensors = [None] * cap


class FlatNode:
    """A lightweight view into a ``NodePool`` node.

    Implements the self-play root contract (``.children`` keyed by action
    index, ``.visit_count``) on top of the flat pool — the node data lives
    in the pool, the view is just an index (ADR-0007).
    """

    __slots__ = ("idx", "pool")

    def __init__(self, idx: int, pool: "NodePool") -> None:
        self.idx = idx
        self.pool = pool

    @property
    def children(self) -> dict:
        """Children as {action_index -> FlatNode}."""
        return self.pool.children(self.idx)

    @property
    def visit_count(self) -> int:
        """Visit count."""
        return self.pool.node_visits(self.idx)

    @property
    def prior(self) -> np.ndarray:
        """Prior distribution for this node."""
        return self.pool.node_prior(self.idx)

    @property
    def legal(self) -> np.ndarray:
        """Legal action mask for this node."""
        return self.pool.legal[self.idx].copy()

    @property
    def terminal(self) -> bool:
        """Whether this node is terminal."""
        return bool(self.pool.terminal[self.idx])

    @property
    def winner(self) -> int:
        """Winner outcome (+1/−1/0)."""
        return int(self.pool.winner[self.idx])

    @property
    def color(self) -> int:
        """Side-to-move color (+1/−1)."""
        return int(self.pool.colors[self.idx])

    @property
    def parent(self) -> int:
        """Parent node index (−1 for root)."""
        return int(self.pool.parents[self.idx])

    @property
    def value_sum(self) -> float:
        """Backed-up value sum."""
        return float(self.pool.value_sum[self.idx])
