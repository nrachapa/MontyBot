"""Ticket 09 bench: MPS vs CPU on the f128/b4 8-plane net, torch 2.2.
Median of >=3 timings (CODING_STANDARDS). Numbers feed ADR-0007/0010 and the map.

Usage: .venv/bin/python bench_ticket09.py
"""

import os
import statistics
import time

import numpy as np
import torch

os.environ["PYTEST_RUNNING"] = "1"

from src.config import CONFIG  # noqa: E402
from src.training.game import ChessGame, MoveMapper  # noqa: E402
from src.training.network import AlphaZeroNet  # noqa: E402
from src.training.search import MCTS  # noqa: E402
from src.training.training import AlphaZeroTraining  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
CONFIG.filters, CONFIG.blocks = 128, 4  # net A from ADR-0009
print(f"torch {torch.__version__}, mps={torch.backends.mps.is_available()}")


def median(f, n=5, warmup=2):
    for _ in range(warmup):
        f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def bench_train(device: str) -> None:
    dev = torch.device(device)
    net = AlphaZeroNet().to(dev)
    trainer = AlphaZeroTraining(net, dev)
    x = torch.randn(64, CONFIG.input_planes, 8, 8)
    y_p = torch.rand(64, CONFIG.action_size)
    y_p = (y_p / y_p.sum(1, keepdim=True)).numpy()
    y_v = torch.rand(64, 1).numpy()

    def _step():
        states = [(x[i].numpy(), 1, "", None) for i in range(64)]
        batch = [(s, y_p[i], y_v[i, 0]) for i, s in enumerate(states)]
        trainer.train_step(batch)

    ms = median(_step)
    print(f"train step (b64, f128/b4, {device:4s}): {ms:8.2f} ms/step  ({64 / max(ms, 1e-9) * 1000:7.0f} samples/s)")


def bench_move(device: str, compile_net: bool = False) -> None:
    """Unbatched MCTS move with the net on `device` (the ADR-0007 setup)."""
    dev = torch.device(device)
    CONFIG.device, CONFIG.gen_device = device, device  # root refresh + leaf inference follow the net
    net = AlphaZeroNet().to(dev)
    if compile_net:
        net = torch.compile(net)
    game = ChessGame()
    mcts = MCTS(game, net, simulations=32)
    state = game.get_initial_state()
    import chess as _chess

    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"]:
        b = _chess.Board(state[2])
        try:
            mv = b.parse_uci(uci)
        except _chess.IllegalMoveError:
            mv = next(iter(b.legal_moves))
        state = game.apply_move(state, MoveMapper.to_index(mv, b))
    ms = median(lambda: mcts.search(state), n=5, warmup=2)
    print(f"mcts move (32 sims, f128/b4, {device:4s}, compile={int(compile_net)}): {ms:8.2f} ms/move")


if __name__ == "__main__":
    for d in ("cpu", "mps"):
        bench_train(d)
    for d in ("cpu", "mps"):
        bench_move(d, compile_net=False)
    if torch.backends.mps.is_available():
        try:
            bench_move("mps", compile_net=True)
        except torch._dynamo.exc.BackendCompilerFailed:
            print("mcts move (32 sims, f128/b4, mps , compile=1): torch.compile unavailable (inductor: Device mps not supported)")
