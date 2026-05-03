import numpy as np

from .evaluation import SunfishEvaluation, ZeroEvaluation
from .game import ChessGame
from .network import AlphaZeroNet
from .search import MCTS, AlphaBetaSearch, ParallelSelfPlay, SunfishSearch
from .training import AlphaZeroTraining


class Factory:
    _networks = {"alphazero": AlphaZeroNet}
    _games = {"chess": ChessGame}
    _search = {"mcts": MCTS, "sunfish": SunfishSearch, "alphabeta": AlphaBetaSearch}
    _selfplay = {"parallel": ParallelSelfPlay}
    _training = {"alphazero": AlphaZeroTraining}
    _evaluation = {"sunfish": SunfishEvaluation, "zero": ZeroEvaluation}

    @classmethod
    def create_network(cls, name):
        return cls._networks[name]()

    @classmethod
    def create_game(cls, name):
        return cls._games[name]()

    @classmethod
    def create_search(cls, name, game, evaluator=None, network=None):
        if name == "sunfish":
            return cls._search[name](game, evaluator)
        if name == "alphabeta":
            return cls._search[name](game, evaluator)
        if name == "mcts":
            return cls._search[name](game, network)
        return cls._search[name](game)

    @classmethod
    def create_selfplay(cls, name, game, search):
        return cls._selfplay[name](game, search)

    @classmethod
    def create_training(cls, name, net, device):
        return cls._training[name](net, device)

    @classmethod
    def create_evaluation(cls, name):
        return cls._evaluation[name]()


class Buffer:
    def __init__(self, capacity: int = 1024):
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
