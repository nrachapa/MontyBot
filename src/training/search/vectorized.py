from typing import Any

import numpy as np
import torch

from ...core import Game, Network, SearchStrategy
from ..game import MoveMapper, Position, encode_board
from ._core import backup_value, dirichlet_perturb, puct_score


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

    def __init__(
        self,
        game: Game,
        network: Network,
        simulations: int,
        batch: int,
        gen_device: str,
        pool: int = 50_000,
    ) -> None:
        self.game = game
        self.network = network
        self.network.eval()
        self.simulations = simulations
        self.batch = batch
        self.c_puct = 1.0  # matches MCTS
        self.dirichlet_alpha = 0.3
        self.dirichlet_eps = 0.25
        self.pool_cap = pool
        # Where the (pickled) net lives / will run: the configured
        # generation device. x.to(net_device) moves the input in either
        # direction, so a worker whose fresh config differs still lands on
        # the right device for the unpickled net.
        self.net_device = torch.device(gen_device)
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
        st = Position(board, color, fen, tensor)
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
        self._setup(k, color, state.fen, state.board, state.tensor)
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
        new = dirichlet_perturb(self.priors[k], ep, self.dirichlet_eps)
        if self.n_children[k] > 0:
            return  # warm root: children keep their edge priors
        for _j, a in enumerate(acts):
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
        q = self.value_sum[f] / np.maximum(1.0, n)
        scores = puct_score(q, self.priors[f, self.actions[f]], pn, n, self.c_puct)
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
            self.value_sum[k] += backup_value(self.colors[k], leaf_color, v)
            k = self.parents[k]

    def search(self, state: Any, root: FlatNode | None = None) -> tuple[np.ndarray, Any, float]:
        root_idx = root.idx if isinstance(root, FlatNode) else None
        if root_idx is None or self.fens[root_idx] != state.fen:
            self._reset()
            root_idx = self._new_root(state, state.color)
            root = FlatNode(root_idx, self)
        if self.terminal[root_idx]:
            return (np.zeros(MoveMapper.ACTION_SIZE, dtype=np.float32), root,
                    float(self.winner[root_idx]) * int(state.color))
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
