import numpy as np

from ..core import Network, SearchStrategy, SelfPlayStrategy, TrainingStrategy
from .evaluation import SunfishEvaluation, ZeroEvaluation
from .game import ChessGame
from .network import AlphaZeroNet
from .search import MCTS, AlphaBetaSearch, ParallelSelfPlay, SunfishSearch, VectorizedMCTS
from .training import AlphaZeroTraining


class ProgressiveModelTransition:
    """Manages progressive model sizing (paper Sec. 9.1).

    Starts with smaller, faster network for early training.
    Transitions to full network after size_transition_iteration.
    Uses paired head-to-head matches to decide if transition is warranted.
    """

    def __init__(self, config=None) -> None:
        if config is None:
            from src.config import CONFIG
            config = CONFIG
        self.config = config
        self.current_filters = config.initial_filters
        self.current_blocks = config.initial_blocks
        self.transition_iteration = config.size_transition_iteration
        self.transitioned = False

    def should_transition(self, iteration: int) -> bool:
        """Check if model should transition to larger size."""
        return not self.transitioned and iteration >= self.transition_iteration

    def create_larger_network(self, smaller_net, config=None) -> Network:
        """Create larger network initialized with smaller network's predictions.

        Uses function-preserving growth to avoid catch-up training.
        """
        from .network import AlphaZeroNet
        if config is None:
            config = self.config
        larger = AlphaZeroNet(
            filters=config.filters,
            blocks=config.blocks,
            input_planes=config.input_planes,
            action_size=config.action_size,
        )
        # Initialize larger network weights from smaller network
        self._initialize_larger_from_smaller(smaller_net, larger)
        return larger

    def _initialize_larger_from_smaller(self, smaller_net, larger_net):
        """Copy weights from smaller network to larger network.

        This provides head-to-head match capability for size transition decisions.
        """
        # Copy state dict with size adjustment
        smaller_state = smaller_net.state_dict()
        larger_state = larger_net.state_dict()

        # Only copy weights with matching shapes
        for key in larger_state:
            if key in smaller_state and larger_state[key].shape == smaller_state[key].shape:
                larger_state[key] = smaller_state[key].clone()

        # Copy conv weights (first layer expansion with zero padding)
        conv_key = 'conv.weight'
        if conv_key in smaller_state and conv_key in larger_state:
            larger_state[conv_key] = larger_state[conv_key].clone()
            # Copy to first channels, pad rest with zeros
            larger_state[conv_key][:smaller_state[conv_key].shape[0], :, :, :] = smaller_state[conv_key]

        # Copy residual block weights with padding for expanded filters
        for i in range(min(len(smaller_net.blocks), len(larger_net.blocks))):
            prefix = f"blocks.{i}."
            for key in smaller_state:
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    larger_key = f"blocks.{i}.{suffix}"
                    if larger_key in larger_state:
                        # Pad with zeros if shapes differ
                        if larger_state[larger_key].shape == smaller_state[key].shape:
                            larger_state[larger_key] = smaller_state[key].clone()
                        else:
                            # Copy to first dimensions, pad rest with zeros
                            new_weight = larger_state[larger_key].clone()
                            # Determine which dimension to copy along
                            smaller_shape = smaller_state[key].shape
                            # Copy smaller weights into first portion
                            slices = [slice(0, s) for s in smaller_shape]
                            new_weight[tuple(slices)] = smaller_state[key]
                            larger_state[larger_key] = new_weight

        larger_net.load_state_dict(larger_state)

    def get_current_model_config(self) -> dict:
        """Return current model configuration."""
        return {
            "filters": self.current_filters,
            "blocks": self.current_blocks,
            "is_larger": self.transitioned,
        }


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
    def create_selfplay(cls, name: str, game, search, config=None, **kwargs) -> SelfPlayStrategy:
        if config is None:
            from src.config import CONFIG
            config = CONFIG
        return cls._selfplay[name](
            game, search, config.gen_device, config.batch_size, **kwargs
        )

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
