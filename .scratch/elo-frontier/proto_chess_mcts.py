"""Ticket 07 prototype: MCTS generation cost at REAL CHESS scale.

Validates the design decision with real numbers (per 01: never design on 3x3).
- Game: python-chess (real rules), 133 actions, 8-plane representation (ADR-0006)
- Net: small (f32/b1) so the *loop architecture*, not FLOPs, is what we measure
- Variants: unbatched (1 net call/leaf) vs batched-32 (one call/32 leaves)
- Devices: MPS and CPU, at 100/300 sims/move
Search quality is irrelevant here; only loop cost is measured.
"""
import time
import math
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_QUIET, ACT_CAPTURE, ACT_CASTLE = 0, 64, 128  # 64 + 64 + 5 = 133 (ADR-0006)
PIECE_PLANE = None  # pieces mapped via piece_type dict in planes()
PIECE_PLANE = {p: i for i, p in enumerate('pnbrqk')}


def _kings(board):
    wk = bq = None
    for s in range(64):
        p = board.piece_at(chess.SQUARES[s])
        if p is None:
            continue
        if p.piece_type == chess.KING:
            if p.color == chess.WHITE:
                wk = s
            else:
                bq = s
    return wk, bq


def _cast_flags(board):
    """[wO-O, wO-O-O, bO-O, bO-O-O] via legal-move membership (python-chess 1.11
    dropped the BCASTLING_* bit constants)."""
    leg = {(m.from_square, m.to_square) for m in board.legal_moves}
    wk, bk = _kings(board)
    return [
        wk is not None and (wk, wk + 1) in leg,
        wk is not None and (wk, wk - 2) in leg,
        bk is not None and (bk, bk + 1) in leg,
        bk is not None and (bk, bk - 2) in leg,
    ]


def planes(board: chess.Board, color_sign: int) -> np.ndarray:
    """8 planes (ADR-0006 layout): 6 piece-type planes signed by the PIECE's
    side (+1 white / -1 black), castling rights, en-passant target. (8x8 each.)"""
    x = np.zeros((8, 8, 8), dtype=np.float32)
    for sq in range(64):
        pc = board.piece_at(chess.SQUARES[sq])
        if pc is None:
            continue
        plane = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}[pc.piece_type]
        x[plane, sq // 8, sq % 8] = 1.0 if pc.color == chess.WHITE else -1.0
    if _cast_flags(board)[0] or _cast_flags(board)[1]:
        x[6, 0, 4] = 1
    elif _cast_flags(board)[2] or _cast_flags(board)[3]:
        x[6, 7, 4] = 1
    if board.ep_square is not None:
        x[7, board.ep_square // 8, board.ep_square % 8] = 1.0
    return x


def legal_actions(board: chess.Board):
    """Legal moves as (133-int, chess.Move) pairs; promotions fold into
    quiet/capture by destination square (piece chosen at execution)."""
    out = []
    for m in board.legal_moves:
        if board.is_capture(m):
            out.append((ACT_CAPTURE + m.to_square, m))
        else:
            out.append((m.to_square, m))
    out += castling_moves(board)
    return out


def castling_moves(board: chess.Board):
    flags = _cast_flags(board)
    wk, bk = _kings(board)
    flags = _cast_flags(board)
    wk, bk = _kings(board)
    res = []
    if flags[0]: res.append((ACT_CASTLE + 0, chess.Move(wk, wk + 1)))
    if flags[1]: res.append((ACT_CASTLE + 1, chess.Move(wk, wk - 2)))
    if flags[2]: res.append((ACT_CASTLE + 2, chess.Move(bk, bk + 1)))
    if flags[3]: res.append((ACT_CASTLE + 3, chess.Move(bk, bk - 2)))
    return res


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 32, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.block = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32), nn.ReLU())
        self.pfc = nn.Linear(32 * 8 * 8, 133)
        self.vfc = nn.Linear(8 * 8, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.block(F.relu(self.bn(self.conv(x))))
        return F.log_softmax(self.pfc(h.flatten(1)), 1), torch.tanh(self.vfc(h[:, -1].flatten(1)))


class Node:
    __slots__ = ('board', 'prior', 'parent', 'children', 'N', 'Q')
    def __init__(self, board, parent=None, prior=0.0):
        self.board, self.prior, self.parent = board, prior, parent
        self.children, self.N, self.Q = None, 0, 0.0


def select(node, c_puct):
    while node.children is not None and node.children and not node.board.is_game_over():
        Ns = math.sqrt(sum(c.N for c in node.children.values()))
        node = max(node.children.values(),
                   key=lambda c: c.Q + c_puct * c.prior * Ns / (1 + c.N))
    return node


def expand(leaf, pr, val):
    legal = legal_actions(leaf.board)
    if not legal:
        leaf.children = {}
        backup(leaf, val)
        return
    acts = np.array([a for a, _ in legal])
    pm = pr[acts]
    pm = pm / pm.sum() if pm.sum() > 0 else np.ones_like(pm) / len(pm)
    leaf.children = {}
    for (a, m), p in zip(legal, pm):
        nb = leaf.board.copy()
        nb.push(m)
        c = Node(nb, leaf, float(p))
        leaf.children[a] = c
    backup(leaf, val)


def backup(leaf, val):
    v, n = val, leaf
    while n is not None:
        n.N += 1
        n.Q += v
        v = -v
        n = n.parent


def mcts_run(board, net, dev, sims, batch=0):
    root = Node(board)
    ts = chess.WHITE  # turn sign helper
    for _ in range(sims):
        if batch:
            leaves = []
            for _ in range(batch):
                leaf = select(root, 1.5)
                if leaf.board.is_game_over():
                    r = proto_score(leaf.board)
                    backup(leaf, 0.0 if r is None else (r if leaf.board.turn == chess.WHITE else -r))
                else:
                    leaves.append(leaf)
            if leaves:
                xs = torch.from_numpy(np.stack(
                    [planes(l.board, 1 if l.board.turn == chess.WHITE else -1) for l in leaves]))
                xs = xs.permute(0, 3, 1, 2).to(dev)
                with torch.inference_mode():
                    lp, v = net(xs)
                priors, vals = torch.exp(lp).cpu().numpy(), v.squeeze(-1).cpu().numpy()
                for l, p, val in zip(leaves, priors, vals):
                    expand(l, p, float(val))
        else:
            leaf = select(root, 1.5)
            if leaf.board.is_game_over():
                r = proto_score(leaf.board)
                backup(leaf, 0.0 if r is None else (r if leaf.board.turn == chess.WHITE else -r))
                continue
            x = torch.from_numpy(planes(leaf.board, 1 if leaf.board.turn == chess.WHITE else -1))
            x = x.permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.inference_mode():
                lp, v = net(x)
            expand(leaf, torch.exp(lp)[0].cpu().numpy(), float(v.item()))

    visits = np.zeros(133, dtype=np.float32)
    if root.children:
        for a, c in root.children.items():
            visits[a] = c.N
    if visits.sum() > 0:
        visits /= visits.sum()
    return visits


def bench(dev_str, sims=100, batch=0, n_moves=8):
    net = Net().to(dev_str).eval()
    b = chess.Board()
    t0 = time.perf_counter()
    for _ in range(n_moves):
        mcts_run(b, net, dev_str, sims, batch=batch)
        lm = list(b.legal_moves)
        b.push(lm[np.random.randint(len(lm))])
        if b.is_game_over():
            b = chess.Board()
    dt = (time.perf_counter() - t0) / n_moves
    games_h = 3600 / (dt * 80)  # ~80 moves per chess game
    print(f"{dev_str:5s} sims={sims:4d} batch={batch:3d}: {dt*1000:9.1f} ms/move   ~{games_h:7.1f} games/h (solo process)")


if __name__ == '__main__':
    print("== chess-scale MCTS cost (f32/b1 net, 8 planes, 133 actions, ADR-0006) ==")
    for dev in ['mps', 'cpu']:
        bench(dev, 100, 0)
        bench(dev, 100, 32)
        bench(dev, 300, 32)


def proto_score(board):
    """+1/-1/0.0 for the game result; None if not over. (python-chess 1.11:
    result_score() was removed; outcome().result is the API.)"""
    if not board.is_game_over():
        return None
    r = board.outcome().result
    return {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}.get(r, 0.0)
