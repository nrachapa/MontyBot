"""Monte Carlo + heuristic search and self-play for the training loop.

This is a package so the file can stay navigable as the strategies grow:

- ``mcts`` — the unbatched ``MCTS`` (legacy; ADR-0007 notes naive per-leaf
  batching is ~10x slower than ``VectorizedMCTS``; kept as the reference and
  for tests) plus the shared ``Node``.
- ``vectorized`` — ``VectorizedMCTS`` (the default neural search, ADR-0007)
  and its ``FlatNode`` pool view.
- ``heuristic`` — ``SunfishSearch`` and ``AlphaBetaSearch`` (heuristic, no net).
- ``selfplay`` — ``ParallelSelfPlay`` and the worker plumbing.

Public names are re-exported here so existing ``from ...search import ...``
importers are unaffected by the split.
"""

from .heuristic import AlphaBetaSearch, SunfishSearch
from .mcts import MCTS, Node
from .selfplay import ParallelSelfPlay, worker_init
from .vectorized import FlatNode, VectorizedMCTS

__all__ = [
    "AlphaBetaSearch",
    "MCTS",
    "Node",
    "ParallelSelfPlay",
    "SunfishSearch",
    "FlatNode",
    "VectorizedMCTS",
    "worker_init",
]
