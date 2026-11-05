"""SearchCore — the MCTS algorithm math shared by ``MCTS`` and ``VectorizedMCTS``.

The two searches store nodes differently (a tree of ``Node`` objects vs a flat
numpy pool) and evaluate the network differently (one call per leaf vs one
batched call per iteration — the ADR-0007 difference), so they cannot share a
single loop. What they *do* share, verbatim, is the AlphaZero algorithm:

- **PUCT selection** — ``puct_score``: ``Q + c·p·√N/(1+n)``.
- **Dirichlet root noise** — ``dirichlet_perturb``: ``p·(1-ε) + noise·ε``.
- **Sign-flip backup** — ``backup_value``: a leaf's value is added to an
  ancestor iff the ancestor is to move for the same side as the leaf (the
  zero-sum convention: White's gain is Black's loss).

Every function is numpy-compatible, so the unbatched ``MCTS`` calls it with
scalars (one child at a time) and ``VectorizedMCTS`` calls it with arrays
(the whole frontier at once) and gets identical math back.
"""

from typing import Any

import numpy as np


def puct_score(value: Any, prior: Any, parent_visits: Any, node_visits: Any, c_puct: float) -> Any:
    """PUCT selection score: ``value + c_puct * prior * sqrt(parent_visits) / (1 + node_visits)``.

    ``value`` is the node's mean backed-up Q (the search computes it; 0 when
    unvisited). Scalar or array.
    """
    return value + c_puct * prior * np.sqrt(np.maximum(parent_visits, 0)) / (1 + node_visits)


def dirichlet_perturb(priors: Any, noise: Any, epsilon: float) -> Any:
    """Dirichlet root-noise: ``priors * (1 - epsilon) + noise * epsilon``.

    ``noise`` is the (already-drawn) Dirichlet sample, aligned to ``priors``.
    Scalar or array.
    """
    return priors * (1 - epsilon) + noise * epsilon


def backup_value(node_color: int, leaf_color: int, value: float) -> float:
    """The sign-flipped contribution of a leaf's value to a node on its path.

    A leaf's value is from the leaf's side-to-move perspective; walking back
    up the tree it keeps its sign for nodes whose side to move matches the
    leaf and flips otherwise (zero-sum two-player game).
    """
    return value if node_color == leaf_color else -value
