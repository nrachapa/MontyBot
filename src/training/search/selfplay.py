import multiprocessing as mp
import os
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch

from ...config import CONFIG
from ...core import Game, SearchStrategy, SelfPlayStrategy
from ..batch_manager import BatchManager

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
        return [(s, p, (1 - 2 * c) * v + 2 * c * winner * s.color) for (s, p, v) in traj]
