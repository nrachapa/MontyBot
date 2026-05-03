from unittest.mock import MagicMock, Mock, patch

import chess
import numpy as np
import pytest
import torch

from src.core import Game, Network, SearchStrategy, SelfPlayStrategy, TrainingStrategy
from src.training.game import ChessGame, MoveMapper, encode_board
from src.training.network import AlphaZeroNet
from src.training.search import MCTS, ParallelSelfPlay
from src.training.system import Buffer, Factory
from src.training.trainer import Trainer
from src.training.training import AlphaZeroTraining


class TestAbstractions:
    def test_abstracts(self):
        with pytest.raises(TypeError):
            Network()
        with pytest.raises(TypeError):
            SearchStrategy()
        with pytest.raises(TypeError):
            SelfPlayStrategy()
        with pytest.raises(TypeError):
            TrainingStrategy()
        with pytest.raises(TypeError):
            Game()


class TestImplementations:
    @pytest.fixture
    def net(self):
        from src.config import CONFIG

        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        CONFIG.filters = 16
        CONFIG.blocks = 1
        net = AlphaZeroNet()
        CONFIG.filters = original_filters
        CONFIG.blocks = original_blocks
        return net

    def test_forward_shapes(self, net):
        from src.config import CONFIG

        x = torch.randn(1, CONFIG.input_planes, 8, 8)
        logits, v = net(x)
        assert logits.shape == (1, CONFIG.action_size)
        assert v.shape == (1, 1)

    def test_mcts_uniform(self):
        from src.config import CONFIG

        game = ChessGame()
        # Mock network for MCTS
        network = Mock()
        network.return_value = (torch.randn(1, CONFIG.action_size), torch.randn(1, 1))
        mcts = MCTS(game, network)

        # Mock game state
        state = game.get_initial_state()
        policy = mcts.search(state)

        assert isinstance(policy, np.ndarray)
        assert policy.shape == (CONFIG.action_size,)
        assert np.isclose(policy.sum(), 1.0)

    @patch("src.training.search.Pool")
    def test_parallel_selfplay_mocked(self, mock_pool):
        from src.config import CONFIG

        mock_pool.return_value.__enter__.return_value = MagicMock()
        game = ChessGame()
        search = Mock()
        search.search.return_value = np.ones(CONFIG.action_size) / CONFIG.action_size
        sp = ParallelSelfPlay(game, search)
        trajs = sp.generate_games(1)
        assert isinstance(trajs, list) and len(trajs) == 1

    def test_training_step_tiny(self, net):
        from src.config import CONFIG

        trainer = AlphaZeroTraining(net, torch.device("cpu"))
        # Fake batch of 2 chess states
        s = np.zeros((CONFIG.input_planes, 8, 8), dtype=np.float32)
        p = np.zeros(CONFIG.action_size, dtype=np.float32)
        p[0] = 1.0
        batch = [((s, 1), p, 0.0), ((s, -1), p, 0.0)]
        loss = trainer.train_step(batch)
        assert isinstance(loss, float) and loss >= 0.0

    def test_mcts_no_legal_moves_edge(self):
        """Force MCTS path where there are no legal moves."""
        game = ChessGame()
        # Use a checkmated state (Fool's Mate)
        # 1. f3 e5 2. g4 Qh4#
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()

        # State: (planes, color, fen, history)
        state = (encode_board(board), 1, board.fen(), [board.fen()])

        network = Mock()
        mcts = MCTS(game, network)
        policy = mcts.search(state)

        assert policy.sum() == 0

    @patch("src.training.search.Pool")
    def test_parallel_selfplay_pool_branch(self, mock_pool):
        """Cover the multiprocessing pool branch by ensuring PYTEST_RUNNING is not set and Pool is used."""
        from src.config import CONFIG

        mock_context = MagicMock()
        mock_context.map.return_value = [[("s", "p", 0.0)]]
        mock_pool.return_value.__enter__.return_value = mock_context
        game = ChessGame()
        search = Mock()
        search.search.return_value = np.ones(CONFIG.action_size) / CONFIG.action_size
        sp = ParallelSelfPlay(game, search)
        with patch.dict("os.environ", {}, clear=True):
            trajs = sp.generate_games(1)
        assert isinstance(trajs, list) and len(trajs) == 1

    # test_chess_game_clone removed as _clone is no longer used

    def test_parallel_selfplay_zero_policy_break(self):
        from src.config import CONFIG

        game = ChessGame()
        search = Mock()
        search.search.return_value = np.zeros(CONFIG.action_size, dtype=np.float32)
        sp = ParallelSelfPlay(game, search)
        # Call private method directly to hit the break branch at line 169
        res = sp._play_game(0)
        assert res == []

    def test_move_mapper_promotions(self):
        import chess

        from src.training.game import MoveMapper

        # Test promotion moves
        move = chess.Move(0, 63, promotion=chess.QUEEN)
        idx = MoveMapper.to_index(move)
        assert idx >= 4096

        # Test from_index with promotion
        move_back = MoveMapper.from_index(4200)
        assert move_back.promotion is not None

        # Test invalid promotion
        move = chess.Move(0, 63, promotion=5)
        idx = MoveMapper.to_index(move)
        assert idx < MoveMapper.ACTION_SIZE

    def test_chess_game_edge_cases(self):
        game = ChessGame()
        state = game.get_initial_state()

        # Test terminal and winner
        assert not game.is_terminal(state)
        winner = game.get_winner(state)
        assert winner in [-1, 0, 1]

        # Test invalid move fallback
        new_state = game.apply_move(state, 9999)
        assert new_state is not None
        assert new_state[1] == -1

    def test_parallel_selfplay_pytest_env(self):
        from src.config import CONFIG

        game = ChessGame()
        network = Mock()
        network.return_value = (torch.randn(1, CONFIG.action_size), torch.randn(1, 1))
        search = MCTS(game, network)
        selfplay = ParallelSelfPlay(game, search)

        # Should run synchronously in main process
        games = selfplay.generate_games(1)
        assert len(games) == 1
        assert len(games[0]) > 0

    def test_encode_board_coverage(self):
        """Test encode_board function."""
        import chess

        board = chess.Board()
        encoded = encode_board(board)
        assert encoded.shape == (104, 8, 8)


class TestSystem:
    def test_config_defaults(self):
        from src.config import CONFIG

        assert CONFIG.input_planes == 104 and CONFIG.action_size == MoveMapper.ACTION_SIZE

    def test_factory_and_buffer(self):
        from src.config import CONFIG

        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        CONFIG.filters = 8
        CONFIG.blocks = 1

        try:
            net = Factory.create_network("alphazero")
            game = Factory.create_game("chess")
            assert isinstance(net, AlphaZeroNet) and isinstance(game, ChessGame)
            buf = Buffer(capacity=4)
            buf.add_batch(
                [[((np.zeros((12, 8, 8), np.float32), 1), np.ones(CONFIG.action_size) / CONFIG.action_size, 0.0)]]
            )
            assert len(buf.data) >= 1
            sample = buf.sample(2)
            assert len(sample) >= 1
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks

    def test_config_gpu_scaling(self):
        """Test GPU memory-based batch size scaling."""
        from src.config import MontyBotConfig

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            # Test small GPU memory
            mock_props.return_value.total_memory = 4e9  # 4GB
            config = MontyBotConfig(device="cuda")
            assert config.batch_size == 16

            # Test large GPU memory
            mock_props.return_value.total_memory = 20e9  # 20GB
            config = MontyBotConfig(device="cuda")
            assert config.batch_size == 64

    def test_buffer_edge_cases(self):
        """Test buffer capacity and empty sampling."""
        buf = Buffer(capacity=2)

        # Test empty buffer
        assert buf.sample(5) == []

        # Test capacity overflow
        buf.add_batch([[("state1", "policy1", 0.1)]])
        buf.add_batch([[("state2", "policy2", 0.2)]])
        buf.add_batch([[("state3", "policy3", 0.3)]])  # Should overflow

        assert len(buf.data) == 2  # Capacity maintained

        # Test sampling
        sample = buf.sample(1)
        assert len(sample) == 1

    def test_factory_coverage(self):
        """Test all factory creation methods."""
        from src.config import CONFIG

        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        CONFIG.filters = 8
        CONFIG.blocks = 1

        try:
            # Test all factory methods
            net = Factory.create_network("alphazero")
            game = Factory.create_game("chess")
            search = Factory.create_search("mcts", game)
            selfplay = Factory.create_selfplay("parallel", game, search)
            training = Factory.create_training("alphazero", net, torch.device("cpu"))

            # Verify types
            assert net is not None
            assert game is not None
            assert search is not None
            assert selfplay is not None
            assert training is not None
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks


class TestTrainer:
    @patch("src.training.search.Pool")
    def test_train_iteration_fast(self, mock_pool):
        from src.config import CONFIG

        mock_pool.return_value.__enter__.return_value = MagicMock()
        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        original_games = CONFIG.games_per_iteration
        original_steps = CONFIG.train_steps

        CONFIG.filters = 8
        CONFIG.blocks = 1
        CONFIG.games_per_iteration = 1
        CONFIG.train_steps = 1

        try:
            tr = Trainer()
            # Ensure search has network attribute for MCTS
            tr.search.network = tr.network
            res = tr.train_iteration(0)
            assert "loss" in res and isinstance(res["loss"], float)
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks
            CONFIG.games_per_iteration = original_games
            CONFIG.train_steps = original_steps

    @patch("src.training.search.Pool")
    def test_trainer_full_workflow(self, mock_pool):
        from src.config import CONFIG

        mock_pool.return_value.__enter__.return_value = MagicMock()
        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        original_mixed = CONFIG.mixed_precision

        CONFIG.filters = 8
        CONFIG.blocks = 1
        CONFIG.mixed_precision = False

        try:
            trainer = Trainer()

            # Mock components
            trainer.search.network = trainer.network  # Ensure MCTS has network
            trainer.selfplay.generate_games = Mock(return_value=[[("state", "policy", 0.1)]])
            trainer.training.train_step = Mock(return_value=1.5)

            # Test training iteration
            result = trainer.train_iteration(0)
            assert "loss" in result
            assert isinstance(result["loss"], float)

            # Test full training loop
            trainer.train(1)
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks
            CONFIG.mixed_precision = original_mixed

    @patch("src.training.search.Pool")
    def test_trainer_empty_buffer_path(self, mock_pool):
        """Covers the branch where buffer.sample returns empty and loop breaks."""
        from src.config import CONFIG

        mock_pool.return_value.__enter__.return_value = MagicMock()
        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        original_mixed = CONFIG.mixed_precision
        original_games = CONFIG.games_per_iteration
        original_steps = CONFIG.train_steps

        CONFIG.filters = 8
        CONFIG.blocks = 1
        CONFIG.mixed_precision = False
        CONFIG.games_per_iteration = 1
        CONFIG.train_steps = 1

        try:
            trainer = Trainer()

            # Ensure MCTS has network attribute
            trainer.search.network = trainer.network
            # Ensure buffer.sample returns [] to trigger the break at line 24
            trainer.selfplay.generate_games = Mock(return_value=[[("s", "p", 0.0)]])
            trainer.buffer.sample = Mock(return_value=[])
            trainer.training.train_step = Mock(return_value=0.0)

            res = trainer.train_iteration(0)
            assert res["loss"] == 0.0
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks
            CONFIG.mixed_precision = original_mixed
            CONFIG.games_per_iteration = original_games
            CONFIG.train_steps = original_steps


class TestExtensibility:
    def test_register_custom_network(self):
        from src.config import CONFIG

        class CustomNet(Network, torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(12 * 8 * 8, CONFIG.action_size)

            def forward(self, x):
                z = x.view(x.size(0), -1)
                return self.l(z), torch.tanh(torch.ones((x.size(0), 1)))

        Factory._networks["custom"] = CustomNet
        net = Factory.create_network("custom")
        x = torch.randn(1, 12, 8, 8)
        logits, v = net(x)
        assert logits.shape == (1, CONFIG.action_size) and v.shape == (1, 1)
