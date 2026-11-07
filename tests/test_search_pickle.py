"""Regression guard: every class in the `search/` package must pickle round-trip.

Self-play pickles searches across a spawn boundary (the multiprocessing Pool),
but the rest of the suite runs with PYTEST_RUNNING=1, which makes
``ParallelSelfPlay`` use in-process workers — so nothing else would catch a
class that stops pickling, e.g. after a module move like the ``search/`` split.
"""
import pickle

import pytest

from src.config import CONFIG
from src.training.evaluation import SunfishEvaluation
from src.training.game import ChessGame
from src.training.network import AlphaZeroNet
from src.training.search import (
    AlphaBetaSearch,
    MCTS,
    ParallelSelfPlay,
    SunfishSearch,
    VectorizedMCTS,
)


@pytest.fixture
def tiny_net():
    """A 4-filter / 1-block net (a few KB) so the pickles stay small + fast."""
    original = (CONFIG.filters, CONFIG.blocks)
    CONFIG.filters, CONFIG.blocks = 4, 1
    net = AlphaZeroNet()
    net.eval()
    yield net
    CONFIG.filters, CONFIG.blocks = original


def _make(name, net):
    game = ChessGame()
    if name == "mcts":
        return MCTS(game, net, simulations=2, action_size=CONFIG.action_size, gen_device="cpu")
    if name == "vectorized_mcts":
        return VectorizedMCTS(game, net, simulations=2, batch=4, gen_device="cpu", pool=1024)
    if name == "sunfish":
        return SunfishSearch(game, SunfishEvaluation(), CONFIG.action_size)
    if name == "alphabeta":
        return AlphaBetaSearch(game, SunfishEvaluation(), action_size=CONFIG.action_size)
    return ParallelSelfPlay(game, MCTS(game, net, simulations=2, action_size=CONFIG.action_size, gen_device="cpu"), gen_device="cpu", batch_size=CONFIG.batch_size)


@pytest.mark.parametrize("name", ["mcts", "vectorized_mcts", "sunfish", "alphabeta", "selfplay"])
def test_search_classes_pickle(name, tiny_net):
    obj = _make(name, tiny_net)
    clone = pickle.loads(pickle.dumps(obj))
    assert clone is not None


def test_vectorized_pool_survives_pickle(tiny_net):
    """The pool is stripped from the pickled state and re-allocated on load
    (workers each allocate their own); the shape must come back intact."""
    obj = VectorizedMCTS(ChessGame(), tiny_net, simulations=2, batch=4, gen_device="cpu", pool=1024)
    clone = pickle.loads(pickle.dumps(obj))
    assert clone.legal.shape == (1024, 133)
