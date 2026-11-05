"""Unbatched MCTS — LEGACY (kept as the reference/ablation path, not deleted).

Superseded for self-play by ``VectorizedMCTS`` (ADR-0007): this tree-of-``Node``
search makes one network call per leaf, which the ADR measured ~10x worse than
the vectorized flat pool. Still a valid search — the Factory's ``"mcts"`` name
and the tests use it — but new search work belongs in ``vectorized.py`` or the
shared ``_core`` (the PUCT / Dirichlet / backup math both MCTS share).
"""

from typing import Any

import numpy as np
import torch

from ...config import CONFIG
from ...core import Game, SearchStrategy
from ._core import backup_value, dirichlet_perturb, puct_score


class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.prior = prior
        self.state = None

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS(SearchStrategy):
    """
    Monte Carlo Tree Search with PUCT (Predictor + Upper Confidence Bound applied to Trees).
    """

    def __init__(self, game: Game, network: Any, simulations: int = None):
        self.game = game
        self.network = network
        self.simulations = simulations or CONFIG.simulations
        self.c_puct = 1.0  # Exploration constant
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25

    def search(self, state, root=None) -> tuple[np.ndarray, Any, float]:
        if root is None:
            root = Node(0)
            root.state = state

        # Add Dirichlet noise to root
        legal_moves = self.game.get_legal_moves(state)
        if not legal_moves:
            return np.zeros(CONFIG.action_size, dtype=np.float32), root, 0.0

        if root.children:
            # Root replacement (naive subtree reuse): the passed root's state IS
            # the current position, so its children are valid — only its policy
            # is stale (the previous player's, noise-applied). Refresh priors
            # from the current player's net (masked over existing children) and
            # let the noise draw below add a fresh Dirichlet perturbation. Do
            # NOT re-expand: that would destroy the inherited statistics.
            tensor = torch.from_numpy(root.state.tensor).float().unsqueeze(0).to(CONFIG.device)
            with torch.no_grad():
                probs = torch.softmax(self.network(tensor)[0], dim=1).cpu().numpy()[0]
            total = float(sum(probs[a] for a in root.children)) or 1.0
            for action, child in root.children.items():
                child.prior = probs[action] / total
        else:
            self._expand(root)

        # Apply noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
        for i, (_action, child) in enumerate(root.children.items()):
            child.prior = dirichlet_perturb(child.prior, noise[i], self.dirichlet_epsilon)

        for _ in range(self.simulations):
            node = root
            search_path = [node]

            # Selection
            while node.children:
                action, node = self._select_child(node)
                search_path.append(node)

            # Expansion and Evaluation
            # Reconstruct state for leaf node (inefficient but simple for now)
            # In optimized version, we would track state updates down the tree
            # For now, let's assume we can re-apply moves from root?
            # No, that's too slow.
            # Better: store state in node? Memory heavy.
            # Compromise: store state in node for now (Python MCTS is slow anyway).

            if node.state is None:
                # Should have been set during expansion of parent
                pass

            value = self._expand(node)

            # Backpropagation
            self._backpropagate(search_path, value, state.color)  # root player color

        # Return visit counts as policy
        policy = np.zeros(CONFIG.action_size, dtype=np.float32)
        for action, child in root.children.items():
            policy[action] = child.visit_count

        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum

        return policy, root, root.value()

    def _select_child(self, node: Node) -> tuple[int, Node]:
        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = puct_score(child.value(), child.prior, node.visit_count, child.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node: Node) -> float:
        # Check if terminal
        if self.game.is_terminal(node.state):
            # Value is from the leaf's side-to-move perspective (matches the
            # training target winner*color and _backpropagate's sign logic).
            return self.game.get_winner(node.state) * node.state.color

        # Evaluate with network
        # Network expects batch (N, C, H, W)
        tensor = torch.from_numpy(node.state.tensor).unsqueeze(0).to(torch.device(CONFIG.gen_device))
        with torch.no_grad():
            logits, value = self.network(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            v = value.item()

        # Mask illegal moves
        legal_moves = self.game.get_legal_moves(node.state)
        policy_sum = 0

        for action in legal_moves:
            p = probs[action]
            child = Node(p)
            # Compute next state for child
            child.state = self.game.apply_move(node.state, action)
            node.children[action] = child
            policy_sum += p

        # Re-normalize priors
        if policy_sum > 0:
            for child in node.children.values():
                child.prior /= policy_sum

        return v

    def _backpropagate(self, search_path: list[Node], value: float, root_color: int):
        for node in search_path:
            node.visit_count += 1
            # Value is from perspective of current player?
            # Network value is for the player whose turn it is in node.state
            # If node.state color == root_color, add value. Else subtract?
            # Let's simplify:
            # Value v is from perspective of player at leaf node.
            # If leaf node player is same as node player, add v.
            # Actually, standard AlphaZero:
            # v is for player at state s.
            # If we are at node s, and choose action a to get to s',
            # s' value is v'. v = -v'.
            # So we should flip value at each step up the tree.

            # Wait, simpler:
            # value passed in is from perspective of player at LEAF node.
            # Let's say leaf node is White's turn. v is White's advantage.
            # If node is White's turn, add v.
            # If node is Black's turn, subtract v.

            node_color = node.state.color
            leaf_color = search_path[-1].state.color
            # Zero-sum sign-flip by side to move (the leaf's value from the
            # leaf's perspective; the convention lives in _core.backup_value).
            node.value_sum += backup_value(node_color, leaf_color, value)
