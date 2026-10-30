"""Ticket 10 acceptance bench: games/h of VectorizedMCTS vs unbatched MCTS
on the PRODUCTION net (128f/4b, ~1.2M params) at equal sims, decision-grade.

Each form is measured on the device its ADR-0007 M6 measurement recommends
(flat → MPS, unbatched → CPU); flat-CPU for reference.
"""
import time

import numpy as np
import torch

from src.config import CONFIG
from src.training.game import ChessGame
from src.training.network import AlphaZeroNet
from src.training.search import MCTS, VectorizedMCTS

SIMS = 250
GAMES_PER = 3   # full games per form
MAX_PLIES = 400


def play_game(game, strat, rng):
    state = game.get_initial_state()
    root = None
    plies = 0
    t0 = time.perf_counter()
    while not game.is_terminal(state):
        policy, new_root, v = strat.search(state, root)
        s = float(policy.sum())
        if s <= 0:
            break
        move = int(rng.choice(len(policy), p=policy / s))
        root = new_root.children.get(move) if new_root is not None else None
        state = game.apply_move(state, move)
        plies += 1
        if plies > MAX_PLIES:
            break
    return time.perf_counter() - t0


def main():
    n_params = sum(p.numel() for p in AlphaZeroNet().parameters())
    print(f"net: {n_params/1e6:.2f}M params, sims={SIMS}")
    rng = np.random.default_rng(0)

    for name, dev, cls in [
        ("flat-MPS   (VectorizedMCTS)", "mps", VectorizedMCTS),
        ("flat-CPU   (VectorizedMCTS)", "cpu", VectorizedMCTS),
        ("unbatched-CPU (MCTS)",        "cpu", MCTS),
    ]:
        torch.manual_seed(0)
        # Unbatched MCTS moves leaves to gen_device; its root-refresh path to
        # CONFIG.device. The flat path uses gen_device for the batch. Point
        # both at the form's device so nothing lands off-net.
        CONFIG.device = dev
        CONFIG.gen_device = dev
        n = AlphaZeroNet().to(dev).eval()
        game = ChessGame()
        strat = cls(game, n, simulations=SIMS)
        strat.network.eval()
        times = [play_game(game, strat, rng) for _ in range(GAMES_PER)]
        med = sorted(times)[GAMES_PER // 2]
        print(f"{name}: median {med:.0f}s/game -> {3600/med:.1f} games/h", flush=True)
        del strat, n
    print("done")


if __name__ == "__main__":
    main()
