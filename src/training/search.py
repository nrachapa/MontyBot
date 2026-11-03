import multiprocessing as mp
import os
from multiprocessing import Pool
from typing import Any

import chess
import numpy as np
import torch

from ..config import CONFIG
from ..core import Game, Network, SearchStrategy, SelfPlayStrategy
from .batch_manager import BatchManager
from .game import MoveMapper, encode_board

WORKER_CLIENT = None


def worker_init(clients, num_workers: int) -> None:
    """Initializer: pick this worker's BatchClient. Clients are passed via
    initargs, not module globals — globals do not cross a spawn boundary
    (macOS/Windows), and an empty list here zero-division-killed every
    worker, hanging pool.map forever. Vectorized searches pass no clients
    (they run the network in-worker, ADR-0007) and instead cap torch
    threads so the workers share the machine instead of oversubscribing it."""
    global WORKER_CLIENT
    if clients:
        # _identity is a tuple like (1,) for worker 1, (2,) for worker 2, etc.
        identity = mp.current_process()._identity
        worker_num = identity[0] - 1 if identity else 0
        WORKER_CLIENT = clients[worker_num % len(clients)]
    # Optimization: workers only build leaf tensors on the generation device.
    CONFIG.device = CONFIG.gen_device
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, num_workers)))


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
            return p, None, 0.0

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

        return p, None, 0.0


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
            return p, None, 0.0

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

        return p, None, 0.0

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

        num_workers = max(1, (os.cpu_count() or 1) - 2)

        # A vectorized search runs the network inside each worker (ADR-0007):
        # no cross-process BatchManager. Non-vectorized network searches keep
        # the BatchManager path; heuristic searches get plain workers.
        manager = None
        clients = []
        if not getattr(self.search, "vectorized", False) and getattr(self.search, "network", None):
            manager = BatchManager(self.search.network, torch.device(CONFIG.gen_device), batch_size=CONFIG.batch_size, timeout=0.01)
            manager.start()
            clients = [manager.create_client() for _ in range(num_workers)]

        try:
            with Pool(num_workers, initializer=worker_init, initargs=(clients, num_workers)) as pool:
                return pool.map(self._play_game, range(n_games))
        finally:
            if manager is not None:
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


class FlatNode:
    """A lightweight view into a VectorizedMCTS node pool.

    Implements the self-play root contract (``.children`` keyed by action
    index, ``.visit_count``) on top of the flat pool — the node data lives
    in the search, the view is just an index (ADR-0007).
    """

    __slots__ = ("idx", "search")

    def __init__(self, idx: int, search: "VectorizedMCTS") -> None:
        self.idx = idx
        self.search = search

    @property
    def children(self) -> dict:
        s = self.search
        lo = int(s.first_child[self.idx])
        n = int(s.n_children[self.idx])
        if n <= 0:
            return {}
        return {int(s.actions[lo + j]): FlatNode(lo + j, s) for j in range(n)}

    @property
    def visit_count(self) -> int:
        return int(self.search.visits[self.idx])


class VectorizedMCTS(SearchStrategy):
    """Fully vectorized MCTS (ADR-0007): one flat node pool, batched PUCT
    selection, and ONE batched net call per iteration (batch 32).

    ``simulations`` means leaf evaluations per move search, exactly as in
    MCTS; one iteration evaluates up to ``batch`` leaves. The pool is
    allocated lazily per process and stripped from the pickled state, so
    self-play workers each allocate their own pool and the per-task pickle
    stays small (the network is the bulk). Memory follows the searched
    subtree as in the branch MCTS (slots are retained for subtree reuse);
    the pool grows by doubling as a game's tree outgrows it.
    """

    vectorized = True

    def __init__(self, game: Game, network: Network, simulations: int | None = None,
                 batch: int | None = None, pool: int = 50_000) -> None:
        from ..config import CONFIG

        self.game = game
        self.network = network
        self.network.eval()
        self.simulations = simulations if simulations is not None else CONFIG.simulations
        self.batch = batch if batch is not None else CONFIG.net_eval_batch
        self.c_puct = 1.0  # matches MCTS
        self.dirichlet_alpha = 0.3
        self.dirichlet_eps = 0.25
        self.pool_cap = pool
        # Where the (pickled) net lives / will run: the configured
        # generation device. x.to(net_device) moves the input in either
        # direction, so a worker whose fresh CONFIG differs still lands on
        # the right device for the unpickled net.
        self.net_device = torch.device(CONFIG.gen_device)
        self._alloc_pool()
        self._reset()

    # ------------------------------------------------------------- pool

    def _alloc_pool(self) -> None:
        cap = self.pool_cap
        a = MoveMapper.ACTION_SIZE
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
        self.boards: list = [None] * cap
        self.fens: list = [""] * cap
        self.tensors: list = [None] * cap
        self.size = 0
        self.frontier: list[int] = []

    def _grow(self, need: int) -> None:
        if need <= self.pool_cap:
            return
        cap = max(need, 2 * self.pool_cap)
        add = cap - self.pool_cap
        a = MoveMapper.ACTION_SIZE
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
        self.pool_cap = cap

    def _reset(self) -> None:
        self.size = 0
        self.frontier = []

    def __getstate__(self) -> dict:
        """Strip the pool: workers allocate their own. The pool carries no
        state a fresh game needs (the first search always starts from a
        FEN-mismatching or absent root), so this is safe for the
        self-play pickling path."""
        state = self.__dict__.copy()
        for key in ("legal", "priors", "terminal", "winner", "colors", "parents",
                    "first_child", "n_children", "actions", "value_sum", "visits",
                    "boards", "fens", "tensors"):
            state.pop(key, None)
        state["size"] = 0
        state["frontier"] = []
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._alloc_pool()

    # -------------------------------------------------------------- nodes

    def _setup(self, k: int, color: int, fen: str, board: Any, tensor: np.ndarray) -> None:
        self.boards[k] = board
        self.fens[k] = fen
        self.tensors[k] = tensor
        self.colors[k] = color
        self.priors[k] = 0.0
        self.legal[k] = 0.0
        self.value_sum[k] = 0.0
        self.visits[k] = 0  # pool slots are reused after _reset — never trust inherited stats
        self.first_child[k] = -1
        self.n_children[k] = 0
        st = (tensor, color, fen, board)
        # Deferred draw (the proto's finding): claim_draw walks the whole
        # history per node; the 150-halfmove cap gives the same winner=0.
        if board.is_game_over(claim_draw=False) or board.halfmove_clock >= self.game.HALFMOVE_CAP:
            self.terminal[k] = 1
            self.winner[k] = self._winner(board)
            self.value_sum[k] = self.winner[k] * color  # own-color perspective
            return
        self.winner[k] = 0
        acts = self.game.get_legal_moves(st)
        if acts:
            self.legal[k, acts] = 1.0

    def _winner(self, board: Any) -> int:
        if board.halfmove_clock >= 150:
            return 0
        r = board.result(claim_draw=False)
        if r == "1-0":
            return 1
        if r == "0-1":
            return -1
        return 0

    def _new_root(self, state: Any, color: int) -> int:
        self._grow(self.size + 1)
        k = self.size
        self.size = k + 1
        self.parents[k] = -1
        self._setup(k, color, state[2], self.game._board(state), state[0])
        return k

    # -------------------------------------------------------------- net

    def _net_root(self, k: int) -> None:
        x = torch.from_numpy(self.tensors[k]).unsqueeze(0).to(next(self.network.parameters()).device)
        with torch.inference_mode():
            logits, _ = self.network(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        m = self.legal[k]
        pm = probs * m
        s = pm.sum()
        self.priors[k] = pm / s if s > 0 else m / m.sum()

    def _apply_dirichlet(self, k: int) -> None:
        acts = np.nonzero(self.legal[k])[0]
        if acts.size == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * int(acts.size))
        ep = np.zeros(MoveMapper.ACTION_SIZE, dtype=np.float32)
        ep[acts] = noise
        new = self.priors[k] * (1 - self.dirichlet_eps) + ep * self.dirichlet_eps
        if self.n_children[k] > 0:
            return  # warm root: children keep their edge priors
        for j, a in enumerate(acts):
            self.priors[k, int(a)] = new[int(a)]

    def _batch_net(self, batch: list[int]) -> None:
        if not batch:
            return
        x = torch.from_numpy(np.stack([self.tensors[i] for i in batch])).to(next(self.network.parameters()).device)
        with torch.inference_mode():
            _, value = self.network(x)
        for i, v in zip(batch, value.squeeze(1).cpu().numpy()):
            self.value_sum[i] = float(v)

    # ------------------------------------------------------------ loop

    def _pick_batch(self) -> list[int]:
        f = np.asarray(self.frontier)
        n = self.visits[f].astype(np.float64)
        pn = self.visits[self.parents[f]].astype(np.float64)
        scores = (self.value_sum[f] / np.maximum(1.0, n)
                  + self.c_puct * self.priors[f, self.actions[f]]
                  * np.sqrt(np.maximum(pn, 0.0)) / (1.0 + n))
        if f.size >= self.batch:
            top = np.argpartition(scores, -self.batch)[-self.batch:]
            take = set(int(f[i]) for i in top)
            self.frontier = [k for k in self.frontier if k not in take]
            return [int(f[i]) for i in top]
        self.frontier = []
        return [int(x) for x in f]

    def _expand(self, k: int) -> None:
        if self.terminal[k] or self.n_children[k] > 0:
            return
        acts = np.nonzero(self.legal[k])[0]
        if acts.size == 0:
            self.terminal[k] = 1  # net said "go" but no move is legal
            return
        self._grow(self.size + int(acts.size))
        start = self.size
        self.size += int(acts.size)
        self.first_child[k] = start
        self.n_children[k] = int(acts.size)
        for j, a in enumerate(acts):
            b = self.boards[k]
            b2 = b.copy()  # child gets its OWN board at the child position
            b2.push(MoveMapper.from_index(int(a), b))
            ck = start + int(j)
            self.parents[ck] = k
            self.actions[ck] = int(a)
            self._setup(ck, -int(self.colors[k]), b2.fen(), b2, encode_board(b2))
            self.priors[ck, int(a)] = self.priors[k, int(a)]  # edge prior = parent net
            if self.terminal[ck]:
                self._backup_path(ck)  # terminal outcome propagates now
            else:
                self.frontier.append(ck)

    def _backup_path(self, leaf: int) -> None:
        v = self.value_sum[leaf] / max(1, int(self.visits[leaf]))
        leaf_color = self.colors[leaf]
        k = leaf
        while k >= 0:
            self.visits[k] += 1
            self.value_sum[k] += v if self.colors[k] == leaf_color else -v
            k = self.parents[k]

    def search(self, state: Any, root: FlatNode | None = None) -> tuple[np.ndarray, Any, float]:
        root_idx = root.idx if isinstance(root, FlatNode) else None
        if root_idx is None or self.fens[root_idx] != state[2]:
            self._reset()
            root_idx = self._new_root(state, state[1])
            root = FlatNode(root_idx, self)
        if self.terminal[root_idx]:
            return (np.zeros(MoveMapper.ACTION_SIZE, dtype=np.float32), root,
                    float(self.winner[root_idx]) * int(state[1]))
        self._net_root(root_idx)
        self._apply_dirichlet(root_idx)
        if self.n_children[root_idx] == 0:
            self._expand(root_idx)
        for _ in range(max(1, (self.simulations + self.batch - 1) // self.batch)):
            if not self.frontier:
                break  # every reachable position is terminal or expanded
            batch = self._pick_batch()
            self._batch_net(batch)  # ONE net call per iteration (the ADR core)
            for k in batch:
                self._backup_path(k)
                self._expand(k)
        lo = int(self.first_child[root_idx])
        kids = np.arange(lo, lo + int(self.n_children[root_idx]))
        pol64 = np.bincount(self.actions[kids], weights=self.visits[kids].astype(np.float64),
                            minlength=MoveMapper.ACTION_SIZE)
        t = pol64.sum()
        if t > 0:
            policy = (pol64 / t).astype(np.float32)
        else:
            # Fallback (a random-weight net can mark every child terminal,
            # leaving no visits): never return a degenerate all-zero policy —
            # the unbatched MCTS always evaluates at least one leaf, so make
            # one final batched call over the frontier if it has anything.
            if self.frontier:
                batch = self._pick_batch()
                self._batch_net(batch)
                for k in batch:
                    self._backup_path(k)
                    self._expand(k)
                lo = int(self.first_child[root_idx])
                kids = np.arange(lo, lo + int(self.n_children[root_idx]))
                pol64 = np.bincount(self.actions[kids], weights=self.visits[kids].astype(np.float64),
                                    minlength=MoveMapper.ACTION_SIZE)
                t = pol64.sum()
            if t > 0:
                policy = (pol64 / t).astype(np.float32)
            else:
                ls = self.legal[root_idx].sum()
                policy = (self.legal[root_idx] / ls if ls > 0 else self.legal[root_idx]).astype(np.float32)
        value = float(self.value_sum[root_idx] / max(1, int(self.visits[root_idx])))
        return policy, root, value
