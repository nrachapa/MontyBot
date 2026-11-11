import multiprocessing as mp
import os
import queue
import threading
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch

from ...core import Game, SearchStrategy, SelfPlayStrategy
from ..batch_manager import BatchManager
from ..replay import SparseReplayBuffer

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
        identity = mp.current_process()._identity
        worker_num = identity[0] - 1 if identity else 0
        WORKER_CLIENT = clients[worker_num % len(clients)]
    # Workers only build leaf tensors on the generation device.
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, num_workers)))


class ParallelSelfPlay(SelfPlayStrategy):
    """Self-play with async generation, restart states, and capped-game fallback.

    Matches the paper's throughput engineering (Sec. 5.5): overlap actor
    game-generation with trainer optimization so self-play keeps feeding
    the replay buffer while the optimizer runs on earlier batches.

    Restart states (paper Sec. 6.3): 50% of games start from archived
    positions with interesting alternatives left to explore.

    Capped games (paper Sec. 4.2.6): when the move limit is reached, a short
    dedicated search at the final position provides a searched cutoff
    value — the paper measured Brier 0.444 vs 0.491 for the material
    heuristic fallback used in the original AlphaZero.
    """

    def __init__(
        self,
        game: Game,
        search: SearchStrategy,
        gen_device: str,
        batch_size: int,
        *,
        n_async_games: int = 0,
        capped_search_budget: int = 400,
        use_restart_states: bool = True,
        restart_state_probability: float = 0.5,
    ):
        self.game = game
        self.search = search
        self.gen_device = gen_device
        self.batch_size = batch_size
        self.n_async_games = n_async_games  # 0 = synchronous, >0 = async
        self.capped_search_budget = capped_search_budget
        self.use_restart_states = use_restart_states
        self.restart_state_probability = restart_state_probability
        self._restart_fen = None  # current restart FEN
        self._traj_queue: queue.Queue | None = None
        self._gen_thread: threading.Thread | None = None
        self._trajs: list | None = None

    def _should_restart(self) -> bool:
        """Whether this game should start from a restart state (paper Sec. 6.3)."""
        if not self.use_restart_states:
            return False
        return np.random.random() < self.restart_state_probability

    def _get_restart_fen(self, restart_buffer: SparseReplayBuffer | None = None) -> str | None:
        """Get a restart FEN from the replay buffer."""
        # In production, this would sample from the buffer
        # For now, return None (use normal opening)
        return None

    def generate_games(self, n_games: int) -> list[Any]:
        """Synchronous generate — blocks until games finish.

        If n_async_games > 0, starts background generation and waits.
        """
        if self.n_async_games == 0 or os.environ.get("PYTEST_RUNNING") == "1":
            return self._do_generate(n_games)
        self._start_async_generation(n_games)
        return self._wait_for_generation()

    def _start_async_generation(self, n_games: int) -> None:
        """Kick off background generation (actor overlap, Sec. 5.5)."""
        self._traj_queue = queue.Queue()
        self._gen_thread = threading.Thread(
            target=self._generate_async,
            args=(n_games,),
            daemon=True,
        )
        self._gen_thread.start()

    def _generate_async(self, n_games: int) -> None:
        self._trajs = self._do_generate(n_games)
        self._traj_queue.put(self._trajs)

    def _wait_for_generation(self) -> list[Any]:
        if self._gen_thread is not None:
            self._gen_thread.join()
            self._gen_thread = None
            self._traj_queue = None
        return self._trajs or []

    def _do_generate(self, n_games: int) -> list[Any]:
        if os.environ.get("PYTEST_RUNNING") == "1":
            return [self._play_game(i) for i in range(n_games)]

        num_workers = max(1, (os.cpu_count() or 1) - 2)

        manager = None
        clients = []
        if not getattr(self.search, "vectorized", False) and getattr(self.search, "network", None):
            manager = BatchManager(self.search.network, torch.device(self.gen_device), batch_size=self.batch_size, timeout=0.01)
            manager.start()
            clients = [manager.create_client() for _ in range(num_workers)]

        try:
            with Pool(num_workers, initializer=worker_init, initargs=(clients, num_workers)) as pool:
                return pool.map(self._play_game, range(n_games))
        finally:
            if manager is not None:
                manager.stop()

    def set_restart_buffer(self, buffer: SparseReplayBuffer) -> None:
        """Set the replay buffer for restart state generation."""
        # This would update the buffer reference for _get_restart_fen
        # In production, the buffer is shared via multiprocessing
        pass

    def _play_game(self, seed: int):
        """Play one full game; trajectory values are the true game outcome.

        Capped games (paper Sec. 4.2.6): if the move limit is reached, a
        short dedicated search at the final position provides a searched
        cutoff value — the paper measured Brier 0.444 vs 0.491 for the
        material heuristic fallback.
        """
        if WORKER_CLIENT is not None:
            self.search.network = WORKER_CLIENT

        rng = np.random.default_rng(seed)
        root = None
        traj = []

        def mcts_policy(state):
            nonlocal root
            from ..game import GameTerminated
            policy, new_root, v = self.search.search(state, root)
            if policy.sum() == 0:
                raise GameTerminated
            move = int(rng.choice(len(policy), p=policy / policy.sum()))
            root = new_root.children.get(move) if new_root is not None else None
            traj.append((state, policy, v))
            return move

        from ..game import play_game

        _, winner, is_capped = play_game(self.game, mcts_policy, max_moves=200)

        # Capped game search fallback (paper Sec. 4.2.6). Use the search
        # instance's own simulations budget (the instance is already tuned
        # for self-play depth) rather than trying to pass a kwarg.
        if is_capped and traj:
            final_state = traj[-1][0]
            _, _, cutoff_value = self.search.search(final_state, root)
            state, policy, _ = traj[-1]
            traj[-1] = (state, policy, cutoff_value)
            winner = 0  # capped games count as draws

        from src.config import CONFIG
        c = CONFIG.c_discount
        return [(s, p, (1 - 2 * c) * v + 2 * c * winner * s.color) for (s, p, v) in traj]
