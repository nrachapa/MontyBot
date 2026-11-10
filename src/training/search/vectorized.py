from typing import Any

import numpy as np
import torch

from ...core import Game, SearchStrategy
from ...training.game import MoveMapper
from ._core import backup_value, dirichlet_perturb, puct_score
from .pool import FlatNode, NodePool


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
        network: torch.nn.Module,
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
        self._alloc_pool(gen_device)
        self._reset()

    def _alloc_pool(self, gen_device: str) -> None:
        self.net_device = torch.device(gen_device)
        self.pool = NodePool.allocate(self.pool_cap)
        self.size = 0
        self.frontier: list[int] = []

    def _reset(self) -> None:
        self.size = 0
        self.frontier = []

    def __getstate__(self) -> dict:
        """Strip the pool: workers allocate their own.

        The pool carries no state a fresh game needs (the first search
        always starts from a FEN-mismatching or absent root), so this is
        safe for the self-play pickling path.
        """
        state = self.__dict__.copy()
        for key in ("pool",):
            state.pop(key, None)
        state["size"] = 0
        state["frontier"] = []
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.pool = NodePool.allocate(self.pool_cap)

    # -------------------------------------------------------------- nodes

    def _grow(self, need: int) -> None:
        self.pool.grow(need)

    def _setup(self, k: int, color: int, state: Any) -> None:
        self.pool.setup_node(k, color, state.board, state.tensor, self.game)

    def _new_root(self, state: Any, color: int) -> int:
        self._grow(self.size + 1)
        k = self.size
        self.size = k + 1
        self.pool.parents[k] = -1
        self._setup(k, color, state)
        return k

    # -------------------------------------------------------------- net

    def _net_root(self, k: int) -> None:
        x = torch.from_numpy(self.pool.node_tensor(k)).unsqueeze(0).to(next(self.network.parameters()).device)
        with torch.inference_mode():
            logits, _ = self.network(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        m = self.pool.legal[k]
        pm = probs * m
        s = pm.sum()
        self.pool.priors[k] = pm / s if s > 0 else m / m.sum()

    def _apply_dirichlet(self, k: int) -> None:
        acts = np.nonzero(self.pool.legal[k])[0]
        if acts.size == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * int(acts.size))
        ep = np.zeros(self.pool.action_size, dtype=np.float32)
        ep[acts] = noise
        new = dirichlet_perturb(self.pool.priors[k], ep, self.dirichlet_eps)
        if self.pool.n_children_count(k) > 0:
            return  # warm root: children keep their edge priors
        for _j, a in enumerate(acts):
            self.pool.priors[k, int(a)] = new[int(a)]

    def _batch_net(self, batch: list[int]) -> None:
        if not batch:
            return
        x = torch.from_numpy(np.stack([self.pool.tensors[i] for i in batch])).to(next(self.network.parameters()).device)
        with torch.inference_mode():
            _, value = self.network(x)
        for i, v in zip(batch, value.squeeze(1).cpu().numpy()):
            self.pool.value_sum[i] = float(v)

    # ------------------------------------------------------------ loop

    def _pick_batch(self) -> list[int]:
        f = np.asarray(self.frontier)
        n = self.pool.visits[f].astype(np.float64)
        pn = self.pool.visits[self.pool.parents[f]].astype(np.float64)
        q = self.pool.value_sum[f] / np.maximum(1.0, n)
        scores = puct_score(q, self.pool.priors[f, self.pool.actions[f]], pn, n, self.c_puct)
        if f.size >= self.batch:
            top = np.argpartition(scores, -self.batch)[-self.batch:]
            take = set(int(f[i]) for i in top)
            self.frontier = [k for k in self.frontier if k not in take]
            return [int(f[i]) for i in top]
        self.frontier = []
        return [int(x) for x in f]

    def _expand(self, k: int) -> None:
        if self.pool.terminal[k] or self.pool.n_children_count(k) > 0:
            return
        acts = np.nonzero(self.pool.legal[k])[0]
        if acts.size == 0:
            self.pool.terminal[k] = 1  # net said "go" but no move is legal
            return
        self._grow(self.size + int(acts.size))
        start = self.size
        self.size += int(acts.size)
        self.pool.first_child[k] = start
        self.pool.n_children[k] = int(acts.size)
        for j, a in enumerate(acts):
            b = self.pool.boards[k]
            b2 = b.copy()  # child gets its OWN board at the child position
            b2.push(MoveMapper.from_index(int(a), b))
            ck = start + int(j)
            self.pool.parents[ck] = k
            self.pool.actions[ck] = int(a)
            # Encode board for child node
            from ..game import encode_board
            tensor = encode_board(b2)
            self.pool.legal[ck] = 0.0  # filled by setup_node
            self.pool.visits[ck] = 0
            self.pool.value_sum[ck] = 0.0
            self.pool.priors[ck] = 0.0
            self.pool.first_child[ck] = -1
            self.pool.n_children[ck] = 0
            self.pool.setup_node(ck, -int(self.pool.colors[k]), b2, tensor, self.game)
            # Edge prior = parent net prior (for subtree reuse)
            self.pool.priors[ck, int(a)] = self.pool.priors[k, int(a)]
            if self.pool.terminal[ck]:
                self._backup_path(ck)  # terminal outcome propagates now
            else:
                self.frontier.append(ck)

    def _backup_path(self, leaf: int) -> None:
        v = self.pool.value_sum[leaf] / max(1, int(self.pool.visits[leaf]))
        leaf_color = self.pool.colors[leaf]
        k = leaf
        while k >= 0:
            self.pool.visits[k] += 1
            self.pool.value_sum[k] += backup_value(self.pool.colors[k], leaf_color, v)
            k = self.pool.parents[k]

    def search(self, state: Any, root: FlatNode | None = None) -> tuple[np.ndarray, Any, float]:
        root_idx = root.idx if isinstance(root, FlatNode) else None
        if root_idx is None or self.pool.fens[root_idx] != state.fen:
            self._reset()
            root_idx = self._new_root(state, state.color)
            root = FlatNode(root_idx, self.pool)
        if self.pool.terminal[root_idx]:
            return (np.zeros(self.pool.action_size, dtype=np.float32), root,
                    float(self.pool.winner[root_idx]) * int(state.color))
        self._net_root(root_idx)
        self._apply_dirichlet(root_idx)
        if self.pool.n_children_count(root_idx) == 0:
            self._expand(root_idx)
        for _ in range(max(1, (self.simulations + self.batch - 1) // self.batch)):
            if not self.frontier:
                break  # every reachable position is terminal or expanded
            batch = self._pick_batch()
            self._batch_net(batch)  # ONE net call per iteration (the ADR core)
            for k in batch:
                self._backup_path(k)
                self._expand(k)
        lo = int(self.pool.first_child[root_idx])
        kids = np.arange(lo, lo + int(self.pool.n_children[root_idx]))
        pol64 = np.bincount(self.pool.actions[kids], weights=self.pool.visits[kids].astype(np.float64),
                            minlength=self.pool.action_size)
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
                lo = int(self.pool.first_child[root_idx])
                kids = np.arange(lo, lo + int(self.pool.n_children[root_idx]))
                pol64 = np.bincount(self.pool.actions[kids], weights=self.pool.visits[kids].astype(np.float64),
                                    minlength=self.pool.action_size)
                t = pol64.sum()
            if t > 0:
                policy = (pol64 / t).astype(np.float32)
            else:
                ls = self.pool.legal[root_idx].sum()
                policy = (self.pool.legal[root_idx] / ls if ls > 0 else self.pool.legal[root_idx]).astype(np.float32)
        value = float(self.pool.value_sum[root_idx] / max(1, int(self.pool.visits[root_idx])))
        return policy, root, value
