import sys, os
sys.path.insert(0, os.path.abspath("../.."))
os.environ.setdefault("PYTEST_RUNNING", "1")
import chess, numpy as np, torch
from src.training.game import ChessGame, MoveMapper, encode_board
from src.training.network import AlphaZeroNet
from proto_flat_mcts import FlatMCTS

torch.manual_seed(0); np.random.seed(0)
game = ChessGame()
net = AlphaZeroNet().eval().to("cpu")
b = chess.Board()
for u in ("e2e4","e7e5","g1f3","b8c6"):
    b.push(chess.Move.from_uci(u))
state = (encode_board(b), 1 if b.turn==chess.WHITE else -1, b.fen(), b)

fm = FlatMCTS(game, net, sims=10, device="cpu")
p, mv, rv = fm.search(state)
print(f"policy sum={p.sum():.3f} rv={rv:.3f} nodes={fm.size}")
mm = tt = lm = 0
for i in range(fm.size):
    bd, fen = fm.boards[i], fm.fens[i]
    if bd is None or fen is None: continue
    if bd.fen() != fen:
        mm += 1
        if mm <= 2: print(f"  BOARD/FEN MISMATCH node {i}: board.fen()={bd.fen()[:20]!r} stored={fen[:20]!r}")
    if not torch.equal(torch.from_numpy(encode_board(bd)), torch.from_numpy(np.asarray(fm.tensors[i]))):
        tt += 1
        if tt <= 2: print(f"  TENSOR MISMATCH node {i} (tensor != encode(board))")
    if fm.terminal[i] == 0:
        expect = np.zeros(133)
        for m in bd.legal_moves:
            expect[MoveMapper.to_index(m, bd)] = 1.0
        if not np.allclose(fm.legal[i], expect):
            lm += 1
            if lm <= 2: print(f"  LEGAL MISMATCH node {i}: {fen[:36]!r}")
print(f"nodes={fm.size}  board/fen mismatches={mm}  tensor mismatches={tt}  legal={lm}")
