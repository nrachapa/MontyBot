import numpy as np

from ..core import Network, SearchStrategy, SelfPlayStrategy, TrainingStrategy
from .evaluation import SunfishEvaluation, ZeroEvaluation
from .game import ChessGame
from .network import AlphaZeroNet
from .search import MCTS, AlphaBetaSearch, ParallelSelfPlay, SunfishSearch, VectorizedMCTS
from .training import AlphaZeroTraining


class Factory:
    _networks = {"alphazero": AlphaZeroNet}
    _games = {"chess": ChessGame}
    _search = {"mcts": MCTS, "vectorized_mcts": VectorizedMCTS, "sunfish": SunfishSearch, "alphabeta": AlphaBetaSearch}
    _selfplay = {"parallel": ParallelSelfPlay}
    _training = {"alphazero": AlphaZeroTraining}
    _evaluation = {"sunfish": SunfishEvaluation, "zero": ZeroEvaluation}

    @classmethod
    def create_network(cls, name: str, config=None) -> Network:
        if config is None:
            from src.config import CONFIG
            config = CONFIG
        return cls._networks[name](
            filters=config.filters,
            blocks=config.blocks,
            input_planes=config.input_planes,
            action_size=config.action_size,
        )

    @classmethod
    def create_game(cls, name):
        return cls._games[name]()

    @classmethod
    def create_search(cls, name: str, game, evaluator=None, network=None, config=None) -> SearchStrategy:
        if config is None:
            from src.config import CONFIG
            config = CONFIG
        if name == "sunfish":
            return cls._search[name](game, evaluator, config.action_size)
        if name == "alphabeta":
            return cls._search[name](game, evaluator, depth=3, action_size=config.action_size)
        if name == "mcts":
            return cls._search[name](game, network,
                                     simulations=config.simulations,
                                     action_size=config.action_size,
                                     gen_device=config.gen_device)
        if name == "vectorized_mcts":
            return cls._search[name](game, network,
                                     simulations=config.simulations,
                                     batch=config.net_eval_batch,
                                     gen_device=config.gen_device)
        return cls._search[name](game)

    @classmethod
    def create_selfplay(cls, name: str, game, search, config=None) -> SelfPlayStrategy:
        if config is None:
            from src.config import CONFIG
            config = CONFIG
        return cls._selfplay[name](game, search, config.gen_device, config.batch_size)

    @classmethod
    def create_training(cls, name: str, net, device, config=None) -> TrainingStrategy:
        if config is None:
            from src.config import CONFIG
            config = CONFIG
        return cls._training[name](net, device, config.learning_rate, config.mixed_precision)

    @classmethod
    def create_evaluation(cls, name):
        return cls._evaluation[name]()


class Buffer:
    def __init__(self, capacity: int = 250_000):
        self.data = []
        self.capacity = capacity

    def add_batch(self, trajectories):
        for t in trajectories:
            self.data.extend(t)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity :]

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.data))
        if n == 0:
            return []
        idx = np.random.choice(len(self.data), n, replace=False)
        return [self.data[i] for i in idx]
