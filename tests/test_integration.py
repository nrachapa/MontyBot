import os

import numpy as np
import pytest
import torch

from src.config import CONFIG
from src.training.game import ChessGame
from src.training.network import AlphaZeroNet
from src.training.search import MCTS, ParallelSelfPlay
from src.training.system import Factory


def test_parallel_selfplay_integration():
    # Setup
    original_filters = CONFIG.filters
    original_blocks = CONFIG.blocks
    CONFIG.filters = 4
    CONFIG.blocks = 1

    try:
        game = ChessGame()
        net = AlphaZeroNet()
        net.eval()

        # We need to ensure we are not in pytest mode for ParallelSelfPlay to use BatchManager
        # But we are running in pytest.
        # ParallelSelfPlay checks os.environ.get("PYTEST_RUNNING") == "1"
        # We need to unset it for this test.

        from unittest.mock import patch

        # We also need to make sure multiprocessing works.
        # On Mac, default is 'spawn'. Objects must be pickleable.
        # Net is pickleable. Game is pickleable.

        search = MCTS(game, net, simulations=2)
        sp = ParallelSelfPlay(game, search)

        # Run generation
        # We use a small number of games
        with patch.dict(os.environ, {"PYTEST_RUNNING": "0"}):
            games = sp.generate_games(n_games=2)

        assert len(games) == 2
        assert len(games[0]) > 0

    finally:
        CONFIG.filters = original_filters
        CONFIG.blocks = original_blocks
