import multiprocessing as mp
import os
from multiprocessing import Pool
from typing import Any

import chess
import numpy as np
import torch

from ..config import CONFIG
from ..core import Game, SearchStrategy, SelfPlayStrategy
from .batch_manager import BatchManager
from .game import MoveMapper

WORKER_CLIENT = None


def worker_init(clients):
    """Initializer: pick this worker's BatchClient. Clients are passed via
    initargs, not module globals — globals do not cross a spawn boundary
    (macOS/Windows), and an empty list here zero-division-killed every
    worker, hanging pool.map forever."""
    global WORKER_CLIENT
    # _identity is a tuple like (1,) for worker 1, (2,) for worker 2, etc.
    identity = mp.current_process()._identity
    worker_num = identity[0] - 1 if identity else 0
    WORKER_CLIENT = clients[worker_num % len(clients)]
    # Optimization: workers only build leaf tensors on the generation device;
    # inference happens in the main-process BatchManager.
    CONFIG.device = CONFIG.gen_device


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
            tensor = torch.from_numpy(root.state[0]).float().unsqueeze(0).to(CONFIG.device)
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
            child.prior = child.prior * (1 - self.dirichlet_epsilon) + noise[i] * self.dirichlet_epsilon

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
            self._backpropagate(search_path, value, state[1])  # state[1] is root player color

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
            ucb = self.c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.value() + ucb
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
            return self.game.get_winner(node.state) * node.state[1]

        # Evaluate with network
        # Network expects batch (N, C, H, W)
        tensor = torch.from_numpy(node.state[0]).unsqueeze(0).to(torch.device(CONFIG.gen_device))
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

            node_color = node.state[1]
            # We need to know the color of the LEAF node to interpret 'value'
            leaf_color = search_path[-1].state[1]

            if node_color == leaf_color:
                node.value_sum += value
            else:
                node.value_sum -= value


class SunfishSearch(SearchStrategy):
    """
    Heuristic search using Sunfish evaluation (PST + Material).
    """

    def __init__(self, game: Game, evaluator: Any):
        self.game = game
        self.action_size = CONFIG.action_size
        self.evaluator = evaluator
        self.network = None  # Explicitly set to None to avoid AttributeError

    def search(self, state, root=None) -> tuple[np.ndarray, Any]:
        # state is (planes, color, fen)
        fen = state[2]
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        p = np.zeros(self.action_size, dtype=np.float32)
        if not legal_moves:
            return p, None

        # Evaluate all next states
        scores = []
        indices = []

        turn_color = 1 if board.turn == chess.WHITE else -1

        for move in legal_moves:
            # Index from the PRE-push board: is_capture() is decided by the
            # destination occupancy before the move is made.
            idx = MoveMapper.to_index(move, board)
            board.push(move)

            # Evaluate from White's perspective
            # We pass the state tuple to evaluate, but our evaluator expects state?
            # Wait, SunfishEvaluation.evaluate expects state tuple?
            # Let's check SunfishEvaluation implementation.
            # It takes 'state'.
            # But here we are inside search loop, we have a board object.
            # We should construct a state tuple for the evaluator.
            # Or we should change evaluator interface to accept board?
            # No, interface says 'state: Any'.
            # So we must pass the state tuple.

            next_state = (None, 1 if board.turn == chess.WHITE else -1, board.fen())
            raw_score = self.evaluator.evaluate(next_state)

            # Convert to side-to-move perspective
            score = raw_score * turn_color

            scores.append(score)
            indices.append(idx)

            board.pop()

        # Softmax with temperature
        T = 200.0
        scores = np.array(scores)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / T)
        probs = exp_scores / np.sum(exp_scores)

        for idx, prob in zip(indices, probs):
            if idx < self.action_size:
                p[idx] += prob  # add: multiple moves can map to one action

        return p, None


class AlphaBetaSearch(SearchStrategy):
    """
    Alpha-Beta Pruning search using an evaluator.
    """

    def __init__(self, game: Game, evaluator: Any, depth: int = 3):
        self.game = game
        self.action_size = CONFIG.action_size
        self.evaluator = evaluator
        self.depth = depth
        self.network = None

    def search(self, state, root=None) -> tuple[np.ndarray, Any]:
        # state is (planes, color, fen, history, board)
        # We need the board object for move generation
        board = state[4] if len(state) > 4 else chess.Board(state[2])

        legal_moves = list(board.legal_moves)
        p = np.zeros(self.action_size, dtype=np.float32)
        if not legal_moves:
            return p, None

        best_score = -float("inf")
        best_move = None

        # Root search (maximize for current player)
        alpha = -float("inf")
        beta = float("inf")

        # Order moves? For now, random or simple ordering could help.
        # Let's just iterate.
        for move in legal_moves:
            board.push(move)
            # Next state is opponent's turn, so they will minimize our score.
            # We pass -beta, -alpha for negamax
            score = -self._alphabeta(board, self.depth - 1, -beta, -alpha, -1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

        if best_move:
            idx = MoveMapper.to_index(best_move, board)
            if idx < self.action_size:
                p[idx] = 1.0

        return p, None

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float, color: int) -> float:
        if depth == 0 or board.is_game_over():
            # Evaluate from perspective of side to move
            # Sunfish evaluator returns score from White's perspective?
            # Let's check SunfishEvaluation.evaluate
            # It returns score. If white pieces > black pieces, score > 0.
            # We need score from perspective of 'board.turn'.

            # Construct state for evaluator
            state = (None, 1 if board.turn == chess.WHITE else -1, board.fen())
            raw_score = self.evaluator.evaluate(state)

            # If board.turn is WHITE, raw_score is correct.
            # If board.turn is BLACK, raw_score is inverted (since raw_score is white advantage).
            # Wait, SunfishEvaluation returns "score".
            # If White is winning, score is positive.
            # If it's White's turn, we want positive score.
            # If it's Black's turn, we want positive score if Black is winning (so negative raw_score).

            perspective = 1 if board.turn == chess.WHITE else -1
            return raw_score * perspective

        legal_moves = list(board.legal_moves)
        best_score = -float("inf")

        for move in legal_moves:
            board.push(move)
            score = -self._alphabeta(board, depth - 1, -beta, -alpha, -color)
            board.pop()

            if score >= beta:
                return score  # Fail hard beta-cutoff
            if score > best_score:
                best_score = score
            alpha = max(alpha, score)

        return best_score


class ParallelSelfPlay(SelfPlayStrategy):
    def __init__(self, game: Game, search: SearchStrategy):
        self.game = game
        self.search = search

    def generate_games(self, n_games: int) -> list[Any]:
        if os.environ.get("PYTEST_RUNNING") == "1":
            return [self._play_game(i) for i in range(n_games)]

        # Check if search strategy uses a network
        if not getattr(self.search, "network", None):
            # Fallback for heuristic search (e.g. Sunfish)
            num_workers = min(os.cpu_count() or 1, 4)
            with Pool(num_workers) as pool:
                return pool.map(self._play_game, range(n_games))

        # Initialize BatchManager
        # Reserve cores for GPU manager and OS
        num_workers = max(1, (os.cpu_count() or 1) - 2)
        manager = BatchManager(self.search.network, torch.device(CONFIG.gen_device), batch_size=CONFIG.batch_size, timeout=0.01)
        manager.start()

        try:
            clients = [manager.create_client() for _ in range(num_workers)]
            with Pool(num_workers, initializer=worker_init, initargs=(clients,)) as pool:
                return pool.map(self._play_game, range(n_games))
        finally:
            manager.stop()

    def _play_game(self, seed: int):
        """Play one full game; trajectory values are the true game outcome
        from the perspective of the player who moved into each position."""
        # Patch network if running in worker with BatchClient
        if WORKER_CLIENT is not None:
            self.search.network = WORKER_CLIENT

        rng = np.random.default_rng(seed)
        state = self.game.get_initial_state()
        traj = []
        root = None
        while not self.game.is_terminal(state):
            # Subtree reuse (ticket 10 scope): after move m, the next player's
            # search root is the child node m led to (warm statistics); its
            # state IS the current position, search() refreshes its policy.
            policy, new_root, v = self.search.search(state, root)
            if policy.sum() == 0:
                break
            move = int(rng.choice(len(policy), p=policy / policy.sum()))
            traj.append((state, policy, v))
            root = new_root.children.get(move) if new_root is not None else None
            state = self.game.apply_move(state, move)

        winner = self.game.get_winner(state)
        # Value target = (1-2c)v + 2c*winner*color (AlphaZero convention):
        # the discounted search value bootstraps the value head toward a
        # positional read, anchored to the actual game outcome.
        c = CONFIG.c_discount
        return [(s, p, (1 - 2 * c) * v + 2 * c * winner * s[1]) for (s, p, v) in traj]
