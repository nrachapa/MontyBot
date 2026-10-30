"""Ticket 10 prototype: the ADR-0007 flat vectorized MCTS at real chess scale.

The ADR-0007 decision: fully vectorized MCTS — flat numpy node pool,
batched PUCT, one batched net call per iteration. This proto is the
design that replaces src/training/search.py's per-leaf python object
tree + per-leaf net calls:

  * Flat pool: one numpy array per node field; object slots hold the
    per-node chess.Board, FEN string, and (8,8,8) input tensor. No
    per-node python objects, no dict of children.
  * One batched net call per iteration (BATCH leaves): the iteration
    picks the top-BATCH PUCT-scored leaves of the frontier in ONE
    vectorized pass, calls the net once on all of them, then expands
    them all and backprops each. BATCH=32 = the ADR's own measurement
    protocol (the rejected naive batch-32 paid per-leaf python object
    cost; the flat pool is what ADR-0007 asks to validate).
  * Board-and-FEN consistency: every child gets its OWN board object
    (game.apply_move: copy + push); node k's board, fen, tensor,
    legal mask, and terminal flag all describe the same position.
    verify_flat.py asserts board.fen() == fen and
    tensor == encode_board(board) for every node.
  * python-chess cost is paid exactly once per NEW position, at expand
    time (legal mask / terminal / encode); the hot loop (PUCT + net +
    backup) touches numpy only — the "caching/deferral" ADR-0007
    requires of python-chess inside the vectorized loop.

Usage:  search(state, root=None) -> (policy[133], move, root_value)
        Passing a child node as `root` reuses its subtree (the branch
        MCTS's root-replacement rule); a FEN mismatch resets the pool.
Bench:  .venv/bin/python .scratch/elo-frontier/proto_flat_mcts.py
        (CPU here; pass device="mps" to FlatMCTS for the M6 run)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import chess
import numpy as np
import torch

from src.training.game import ChessGame, MoveMapper, encode_board
from src.training.network import AlphaZeroNet

POOL = 1 << 21  # 2,097,152 nodes; guard raises if a run exhausts it


class FlatMCTS:
    """Flat vectorized MCTS (ADR-0007). One batched net call / iteration."""

    def __init__(self, game: ChessGame, net: AlphaZeroNet, sims: int = 800,
                 device: str = "cpu", batch: int = 32):
        self.game = game
        self.net = net
        self.net.eval()
        self.sims = sims
        self.batch = batch
        self.device = torch.device(device)
        self.c_puct = 1.0            # matches src/training/search.py MCTS
        self.dirichlet_alpha = 0.3
        self.dirichlet_eps = 0.25

        # Flat pool: numeric numpy + object slots for the per-node state.
        self.boards: list = [None] * POOL
        self.fens: list = [""] * POOL
        self.tensors: list = [None] * POOL
        self.legal = np.zeros((POOL, 133), dtype=np.float32)   # one-hot legal mask
        self.priors = np.zeros((POOL, 133), dtype=np.float32)  # own net prior per action
        self.terminal = np.zeros(POOL, dtype=np.int8)
        self.winner = np.zeros(POOL, dtype=np.int8)            # white +1 / 0 / black -1
        self.colors = np.zeros(POOL, dtype=np.int8)            # +1 white / -1 black
        self.parents = np.full(POOL, -1, dtype=np.int64)
        self.first_child = np.full(POOL, -1, dtype=np.int64)
        self.n_children = np.zeros(POOL, dtype=np.int64)
        self.actions = np.full(POOL, -1, dtype=np.int64)       # edge action into this node
        self.value_sum = np.zeros(POOL, dtype=np.float32)      # own-color perspective
        self.visits = np.zeros(POOL, dtype=np.int64)
        self.size = 0
        self.frontier: list[int] = []  # unexpanded, non-terminal leaves
        self.net_calls = 0
        self.leaf_visits = 0  # net leaf evaluations = sum of batch sizes

    # ------------------------------------------------------------------ pool

    def _reset(self) -> None:
        self.size = 0
        self.frontier = []

    def _new_root(self, state, color: int) -> int:
        k = self.size
        self.size = k + 1
        if self.size > POOL:
            raise RuntimeError("flat MCTS pool exhausted")
        self.parents[k] = -1
        self.first_child[k] = -1  # pool slots are REUSED after _reset — clear child/visit state
        self.n_children[k] = 0
        self.visits[k] = 0
        self.value_sum[k] = 0.0
        self._setup(k, color, state[2], state[3], state[0])
        return k

    def _setup(self, k: int, color: int, fen: str, board, tensor) -> None:
        """Board/fen/tensor/terminal/winner/legal all describe the SAME
        position (the node's own) — verified by verify_flat.py."""
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
        # Deferred-draw terminal check (ADR-0007: manage python-chess cost
        # inside the vectorized loop). claim_draw=True walks the whole move
        # history per node (~113 us at 16 plies, grows with game length);
        # threefold/50-move draws are worth 1/2 = winner 0, exactly what the
        # 150-halfmove cap gives, so the search defers the claim to the cap.
        # The data engine (game.py, ticket 08) keeps the full claim_draw.
        b = st[3]
        if b.is_game_over(claim_draw=False) or b.halfmove_clock >= self.game.HALFMOVE_CAP:
            self.terminal[k] = 1
            self.winner[k] = self._winner(b)
            self.value_sum[k] = self.winner[k] * color  # own-color perspective
            return
        self.winner[k] = 0
        acts = self.game.get_legal_moves(st)
        if acts:
            self.legal[k, acts] = 1.0

    @staticmethod
    def _winner(b) -> int:
        """+1 white / -1 black / 0 draw — no claim_draw history walk."""
        if b.halfmove_clock >= 150:
            return 0
        r = b.result(claim_draw=False)
        if r == "1-0":
            return 1
        if r == "0-1":
            return -1
        return 0

    # -------------------------------------------------------------- net step

    def _net_root(self, root: int) -> None:
        """Net call for the root: 133 policy (masked, renormalized)."""
        x = torch.from_numpy(self.tensors[root]).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, _ = self.net(x)
        self.net_calls += 1
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        m = self.legal[root]
        pm = probs * m
        s = pm.sum()
        self.priors[root] = pm / s if s > 0 else m / m.sum()

    def _apply_dirichlet(self, root: int) -> None:
        """Fresh Dirichlet draw over the root's edge actions each search —
        the branch MCTS's root-replacement rule (children keep their stats)."""
        acts = np.nonzero(self.legal[root])[0]
        if acts.size == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * int(acts.size))
        ep = np.zeros(133, dtype=np.float32)
        ep[acts] = noise
        new = self.priors[root] * (1 - self.dirichlet_eps) + ep * self.dirichlet_eps
        if self.n_children[root] > 0:
            # Warm root: children keep their edge priors (noised when the
            # root was expanded) — re-drawing here would clobber them.
            return
        # Fresh root: search() expands the root next; _expand copies these
        # (noised) priors onto the children's edge slots.
        for j, a in enumerate(acts):
            self.priors[root, int(a)] = new[int(a)]

    def _batch_net(self, batch: list[int]) -> None:
        """ONE net call over every leaf of the iteration (the ADR-0007
        core). Internal leaves need only the value head."""
        if not batch:
            return
        x = torch.from_numpy(np.stack([self.tensors[i] for i in batch])).to(self.device)
        with torch.inference_mode():
            _, value = self.net(x)
        self.net_calls += 1
        vals = value.squeeze(1).cpu().numpy()
        self.leaf_visits += len(batch)
        for i, v in zip(batch, vals):
            self.value_sum[i] = float(v)

    # ------------------------------------------------------------ expansion

    def _expand(self, k: int) -> None:
        """Create the child block of k. Each child gets its OWN board
        object at the child position (game.apply_move = copy + push).
        The child's PUCT prior = k's net prior for the edge action."""
        if self.terminal[k] or self.n_children[k] > 0:
            return
        acts = np.nonzero(self.legal[k])[0]
        if acts.size == 0:
            self.terminal[k] = 1  # net said "go" but no move is legal
            return
        start = self.size
        self.size += int(acts.size)
        if self.size > POOL:
            raise RuntimeError("flat MCTS pool exhausted")
        self.first_child[k] = start
        self.n_children[k] = int(acts.size)
        for j, a in enumerate(acts):
            b = self.boards[k]
            b2 = b.copy()  # child gets its OWN board at the child position
            b2.push(MoveMapper.from_index(int(a), b))
            ck = start + int(j)
            self.parents[ck] = k
            self.actions[ck] = int(a)
            fen = b2.fen()  # child FEN from the child board (pushed)
            self._setup(ck, -int(self.colors[k]), fen, b2, encode_board(b2))
            self.priors[ck, int(a)] = self.priors[k, int(a)]  # edge prior (k's net)
            if self.terminal[ck]:
                self._backup_path(ck)  # terminal outcome propagates now
            else:
                self.frontier.append(ck)

    def _backup_path(self, leaf: int) -> None:
        """Backup the leaf's value (own-color perspective) up to the root,
        flipping sign on every color change."""
        v = self.value_sum[leaf] / max(1, int(self.visits[leaf]))
        leaf_color = self.colors[leaf]
        k = leaf
        while k >= 0:
            self.visits[k] += 1
            self.value_sum[k] += v if self.colors[k] == leaf_color else -v
            k = self.parents[k]

    # ---------------------------------------------------------- the loop

    def _pick_batch(self) -> list[int]:
        """Batched PUCT: top-BATCH leaves of the frontier, one vectorized
        pass. The PUCT prior of leaf k is k's edge prior (parent net)."""
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

    def search(self, state, root: int | None = None) -> tuple[np.ndarray, object, float]:
        if root is None or self.fens[root] != state[2]:
            self._reset()  # a passed root whose FEN no longer matches is dead weight
            root = self._new_root(state, state[1])
        if self.terminal[root]:
            return (np.zeros(133, dtype=np.float32), None,
                    float(self.winner[root]) * int(self.colors[root]))
        self._net_root(root)
        self._apply_dirichlet(root)
        if self.n_children[root] == 0:
            self._expand(root)

        for _ in range(self.sims):
            if not self.frontier:
                break  # every reachable position is terminal or expanded
            batch = self._pick_batch()
            self._batch_net(batch)          # ONE net call per iteration
            for k in batch:
                self._backup_path(k)
                self._expand(k)

        # Policy over 133 slots: sum root-child visits by their EDGE action
        # (the move made at the root) — the branch MCTS's same rule.
        lo = int(self.first_child[root])
        kids = np.arange(lo, lo + int(self.n_children[root]))
        pol64 = np.bincount(self.actions[kids], weights=self.visits[kids].astype(np.float64),
                            minlength=133)
        t = pol64.sum()
        pol = (pol64 / t if t > 0 else pol64).astype(np.float32)
        act = int(np.argmax(pol))
        move = MoveMapper.from_index(act, self.boards[root])  # raises if illegal — a silent fallback would poison self-play data
        rv = float(self.value_sum[root] / max(1, int(self.visits[root])))
        return pol, move, rv


# ------------------------------------------------------------------ bench

def _bench(sims: int, device: str, net, batch: int, n_warm: int = 3, n_timed: int = 5):
    """Median ms/move over n_timed timed moves (n_warm discarded).
    Per-search deltas: net calls, leaf visits (sum of batch sizes), nodes.
    sims = number of batched iterations (net calls) — NOT leaf visits."""
    game = ChessGame()
    fm = FlatMCTS(game, net, sims=sims, device=device, batch=batch)
    b = chess.Board()
    state = (encode_board(b), 1, b.fen(), b)
    ts, calls, leaves, sizes = [], 0, 0, []
    for i in range(n_warm + n_timed):
        c0, l0, s0 = fm.net_calls, fm.leaf_visits, fm.size
        t0 = time.perf_counter()
        pol, move, rv = fm.search(state)
        dt = (time.perf_counter() - t0) * 1000
        if i >= n_warm:
            ts.append(dt)
            calls += fm.net_calls - c0
            leaves += fm.leaf_visits - l0
            sizes.append(fm.size - s0)
        if move is None or b.is_game_over():
            b = chess.Board()
        else:
            b.push(move)
        state = (encode_board(b), 1 if b.turn == chess.WHITE else -1, b.fen(), b)
    n = max(1, len(ts))
    med = float(np.median(ts))
    calls //= n
    leaves //= n
    return med, calls, leaves, int(np.median(sizes)) if sizes else 0


def _bench_unbatched(sims: int, device: str, state_dict, n_warm: int = 3, n_timed: int = 5):
    """The current src MCTS (per-leaf net calls) — identical move-
    progression protocol, for a same-machine comparison."""
    from src.training.search import MCTS
    # the branch always generates on cpu (ADR-0007); a SEPARATE cpu net so the
    # flat bench's net (on `device`) is untouched
    gen_net = AlphaZeroNet().to("cpu")
    if state_dict is not None:
        gen_net.load_state_dict(state_dict)
    gen_net.eval()
    game = ChessGame()
    mcts = MCTS(game, gen_net, simulations=sims)
    b = chess.Board()
    state = (encode_board(b), 1, b.fen(), b)
    ts = []
    for i in range(n_warm + n_timed):
        t0 = time.perf_counter()
        pol, _root, _rv = mcts.search(state)
        dt = (time.perf_counter() - t0) * 1000
        if i >= n_warm:
            ts.append(dt)
        if b.is_game_over():
            b = chess.Board()
        else:
            b.push(MoveMapper.from_index(int(np.argmax(pol)), b))
        state = (encode_board(b), 1 if b.turn == chess.WHITE else -1, b.fen(), b)
    return float(np.median(ts))


def main():
    torch.manual_seed(0)
    os.environ.setdefault("PYTEST_RUNNING", "1")
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model.pt"))
    net = AlphaZeroNet().to(device)
    state_dict = None
    if os.path.exists(ckpt):
        try:
            state_dict = torch.load(ckpt, map_location=device, weights_only=False)["model_state"]
            net.load_state_dict(state_dict)
            print(f"loaded trained checkpoint {ckpt}")
        except RuntimeError as e:
            state_dict = None
            print(f"WARNING: checkpoint {ckpt} does not match the current net shape — "
                  f"using random weights (loop cost, not policy quality)")
    else:
        print("WARNING: no trained checkpoint — random weights (loop cost, not policy quality)")
    net.eval()
    print(f"[Ticket 10] device={device} (weights: {'trained' if state_dict else 'RANDOM'})")
    for sims in (32, 100):
        med, calls, leaves, size = _bench(sims, device, net, batch=32)
        unmed = _bench_unbatched(sims, device, state_dict)
        # flat 'sims' = net calls (batch 32) -> leaves = calls*32; unbatched leaves = sims.
        # Honest same-work comparison = ms per leaf.
        print(f"[Ticket 10] {sims:3d} sims | flat {med:8.1f} ms/move ({calls} calls, {leaves} leaves, "
              f"{med / max(leaves, 1):5.2f} ms/leaf, {size} nodes) | unbatched {unmed:8.1f} ms/move "
              f"({unmed / sims:5.2f} ms/leaf)")
    # Equal leaf-evaluation budget (800): the real self-play scale. Flat does
    # 800 leaves in 25 batched calls; unbatched in 800 calls.
    med, calls, leaves, size = _bench(25, device, net, batch=32)
    unmed = _bench_unbatched(800, device, state_dict)
    print(f"[Ticket 10] ~800 leaves | flat {med:8.1f} ms/move ({calls} calls, {leaves} leaves, "
          f"{med / max(leaves, 1):5.2f} ms/leaf) | unbatched {unmed:8.1f} ms/move "
          f"({unmed / 800:5.2f} ms/leaf) | same-work ratio {med / max(unmed, 1e-9):.2f}x")
    print("  ADR-0007 anchor (M6, ticket 09): unbatched 190 ms/move CPU / 216 ms/move MPS at 32 sims.")
    if device == "cpu":
        print("  NOTE: decision-grade needs MPS on the M6 — re-run with: python proto_flat_mcts.py mps")


if __name__ == "__main__":
    main()
