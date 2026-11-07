from unittest.mock import MagicMock, Mock, patch

import chess
import numpy as np
import pytest
import torch

from src.core import Game, Network, SearchStrategy, SelfPlayStrategy, TrainingStrategy
from src.training.game import ChessGame, MoveMapper, Position, encode_board
from src.training.network import AlphaZeroNet
from src.training.search import MCTS, Node, ParallelSelfPlay, VectorizedMCTS
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
        mcts = MCTS(game, network, simulations=10, action_size=CONFIG.action_size, gen_device="cpu")

        # Mock game state
        state = game.get_initial_state()
        policy, root, value = mcts.search(state)

        assert isinstance(policy, np.ndarray)
        assert policy.shape == (CONFIG.action_size,)
        assert np.isclose(policy.sum(), 1.0)
        assert isinstance(root, Node)
        assert -1.0 <= value <= 1.0

    def test_mcts_search_reuses_warm_root(self):
        """Root replacement (naive subtree reuse): a passed-in root that
        already has children must NOT be re-expanded (the branch's old
        behavior destroyed inherited statistics); its children survive with
        their visit counts, and new visits are added on top.
        """
        from src.config import CONFIG

        saved = CONFIG.simulations
        CONFIG.simulations = 50  # enough for pigeonhole: some root child is revisited
        try:
            game = ChessGame()
            network = Mock()
            network.return_value = (torch.zeros(1, CONFIG.action_size), torch.zeros(1, 1))
            mcts = MCTS(game, network, simulations=50, action_size=CONFIG.action_size, gen_device="cpu")

            state = game.get_initial_state()
            policy, root, _ = mcts.search(state)
            assert len(root.children) > 0
            # A root child that was expanded and whose children accumulated visits.
            move = max(root.children, key=lambda a: root.children[a].visit_count)
            child = root.children[move]
            inherited = {a: n.visit_count for a, n in child.children.items()}
            assert any(v > 0 for v in inherited.values())  # test validity guard

            root_visits_before = child.visit_count
            next_state = game.apply_move(state, move)
            policy_2, root_2, _ = mcts.search(next_state, root=child)
            assert root_2 is child
            # Inherited statistics survive (old code re-expanded: fresh leaves at 0).
            for a, visits in inherited.items():
                assert root_2.children[a].visit_count >= visits
            # New visits were added on top of the inherited ones.
            assert root_2.visit_count >= root_visits_before + CONFIG.simulations
        finally:
            CONFIG.simulations = saved

    @patch("src.training.search.selfplay.Pool")
    def test_parallel_selfplay_mocked(self, mock_pool):
        from src.config import CONFIG

        mock_pool.return_value.__enter__.return_value = MagicMock()
        game = ChessGame()
        search = Mock()
        # All-zero policy: exercises the game-loop exit (a real MCTS policy is
        # zero off the legal actions, so from_index must never see illegal ones).
        search.search.return_value = (np.zeros(CONFIG.action_size, dtype=np.float32), None, 0.0)
        sp = ParallelSelfPlay(game, search, gen_device="cpu", batch_size=CONFIG.batch_size)
        trajs = sp.generate_games(1)
        assert isinstance(trajs, list) and len(trajs) == 1

    def test_play_game_stores_discounted_value_target(self):
        """Value target = (1−2c)·v + 2c·winner·color (AlphaZero convention).

        Mock search returns a constant v=0.5 with a uniform-over-legal policy
        (so the game stays legal); the stored trajectory values must follow
        the exact formula. Outcome-independent checks: for the first two
        entries (opposite colors, same v): v0+v1 = 1−2c and
        v0−v1 = 2c·winner·(s0.color − s1.color) ∈ {−4c, 0, +4c}.
        """
        from src.config import CONFIG

        game = ChessGame()
        c = CONFIG.c_discount

        def fake_search(state, root=None):
            actions = game.get_legal_moves(state)
            p = np.zeros(CONFIG.action_size, dtype=np.float32)
            if actions:
                p[actions] = 1.0 / len(actions)
            return p, None, 0.5

        search = Mock()
        search.search.side_effect = fake_search
        sp = ParallelSelfPlay(game, search, gen_device="cpu", batch_size=CONFIG.batch_size)

        trajs = sp.generate_games(1)
        traj = trajs[0]
        assert len(traj) >= 2
        (s0, _, v0), (s1, _, v1) = traj[0], traj[1]
        assert s0.color == 1 and s1.color == -1  # alternating colors
        assert np.isclose(v0 + v1, 1 - 2 * c)
        assert abs(v0 - v1) < 1e-6 or np.isclose(abs(v0 - v1), 4 * c)

    def test_play_game_reuses_subtree_between_moves(self):
        """After move m, the next search root is the child node m led to
        (warm statistics inherited), not a fresh empty root."""
        from src.config import CONFIG

        game = ChessGame()
        network = Mock()
        network.return_value = (torch.zeros(1, CONFIG.action_size), torch.zeros(1, 1))
        mcts = MCTS(game, network, simulations=10, action_size=CONFIG.action_size, gen_device="cpu")
        calls = []

        def spy(state, root=None):
            calls.append(root)
            return mcts.search(state, root)

        wrapper = Mock()
        wrapper.search.side_effect = spy
        sp = ParallelSelfPlay(game, wrapper, gen_device="cpu", batch_size=CONFIG.batch_size)
        trajs = sp.generate_games(1)
        assert len(trajs[0]) >= 2  # game lasted > 1 ply
        # First move: fresh root. Every subsequent move: the previous child.
        assert calls[0] is None
        for root in calls[1:]:
            assert isinstance(root, Node)
            assert root.visit_count >= 1  # inherited statistics

    def test_training_step_tiny(self, net):
        from src.config import CONFIG

        trainer = AlphaZeroTraining(net, torch.device("cpu"), CONFIG.learning_rate, CONFIG.mixed_precision)
        # Fake batch of 2 chess states
        s = np.zeros((CONFIG.input_planes, 8, 8), dtype=np.float32)
        p = np.zeros(CONFIG.action_size, dtype=np.float32)
        p[0] = 1.0
        board = chess.Board()
        fen = board.fen()
        batch = [
            (Position(board, 1, fen, s), p, 0.0),
            (Position(board, -1, fen, s), p, 0.0),
        ]
        loss = trainer.train_step(batch)
        assert isinstance(loss, float) and loss >= 0.0

    def test_mcts_no_legal_moves_edge(self):
        """Force MCTS path where there are no legal moves."""
        from src.config import CONFIG

        game = ChessGame()
        # Use a checkmated state (Fool's Mate)
        # 1. f3 e5 2. g4 Qh4#
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()

        # State: a Position (board is authoritative)
        state = Position(board, 1, board.fen(), encode_board(board))

        network = Mock()
        mcts = MCTS(game, network, simulations=10, action_size=CONFIG.action_size, gen_device="cpu")
        policy, _, _ = mcts.search(state)

        assert policy.sum() == 0

    @patch("src.training.search.selfplay.Pool")
    def test_parallel_selfplay_pool_branch(self, mock_pool):
        """Cover the multiprocessing pool branch by ensuring PYTEST_RUNNING is not set and Pool is used."""
        from src.config import CONFIG

        mock_context = MagicMock()
        mock_context.map.return_value = [[("s", "p", 0.0)]]
        mock_pool.return_value.__enter__.return_value = mock_context
        game = ChessGame()
        search = Mock()
        search.search.return_value = (np.ones(CONFIG.action_size) / CONFIG.action_size, None)
        sp = ParallelSelfPlay(game, search, gen_device="cpu", batch_size=CONFIG.batch_size)
        with patch.dict("os.environ", {}, clear=True):
            trajs = sp.generate_games(1)
        assert isinstance(trajs, list) and len(trajs) == 1

    # test_chess_game_clone removed as _clone is no longer used

    def test_parallel_selfplay_zero_policy_break(self):
        from src.config import CONFIG

        game = ChessGame()
        search = Mock()
        search.search.return_value = (np.zeros(CONFIG.action_size, dtype=np.float32), None, 0.0)
        sp = ParallelSelfPlay(game, search, gen_device="cpu", batch_size=CONFIG.batch_size)
        # Call private method directly to hit the break branch at line 169
        res = sp._play_game(0)
        assert res == []

    def test_move_mapper_promotions(self):
        from src.training.game import MoveMapper

        # Quiet promotion: a7a8=Q with a8 empty folds into the quiet
        # destination-square action; the resolved move keeps the promotion.
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        move = chess.Move(48, 56, promotion=chess.QUEEN)
        assert move in list(board.legal_moves)
        idx = MoveMapper.to_index(move, board)
        assert idx < MoveMapper.CAPTURE_OFFSET
        resolved = MoveMapper.from_index(idx, board)
        assert resolved.to_square == 56 and resolved.promotion is not None

        # Capture promotion: a7xb8 occupies the capture half of the space.
        board2 = chess.Board("1p6/P7/8/8/8/8/8/k6K w - - 0 1")
        move2 = chess.Move(48, 57, promotion=chess.QUEEN)
        assert move2 in list(board2.legal_moves)
        idx2 = MoveMapper.to_index(move2, board2)
        assert MoveMapper.CAPTURE_OFFSET <= idx2 < MoveMapper.CASTLE_OFFSET
        resolved2 = MoveMapper.from_index(idx2, board2)
        assert board2.is_capture(resolved2)

        # Round-trip: every legal move of every position maps to an in-range
        # action that resolves to a legal move (the data-poisoning regression).
        for start in [
            chess.Board(),
            chess.Board("8/Pp6/8/8/8/8/8/K6k w - - 0 1"),
            chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            # All four castles legal (startpos has none: f1/c1 occupied).
            chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        ]:
            for m in start.legal_moves:
                a = MoveMapper.to_index(m, start)
                assert 0 <= a < MoveMapper.ACTION_SIZE
                back = MoveMapper.from_index(a, start)
                assert back in list(start.legal_moves)
                assert back.to_square == m.to_square
                assert start.is_capture(back) == start.is_capture(m)
                if a >= MoveMapper.CASTLE_OFFSET:
                    # Castle actions cannot collide: resolution must be exact.
                    assert back == m, (m, a, back)

    def test_chess_game_edge_cases(self):
        game = ChessGame()
        state = game.get_initial_state()

        # Test terminal and winner
        assert not game.is_terminal(state)
        winner = game.get_winner(state)
        assert winner in [-1, 0, 1]

        # Illegal actions must raise, never silently substitute (ticket 08).
        with pytest.raises(ValueError):
            game.apply_move(state, 9999)
        with pytest.raises(ValueError):
            game.apply_move(state, 48)  # in range, but no white piece can go quietly to a7

    def test_parallel_selfplay_pytest_env(self):
        from src.config import CONFIG

        game = ChessGame()
        network = Mock()
        network.return_value = (torch.randn(1, CONFIG.action_size), torch.randn(1, 1))
        search = MCTS(game, network, simulations=10, action_size=CONFIG.action_size, gen_device="cpu")
        selfplay = ParallelSelfPlay(game, search, gen_device="cpu", batch_size=CONFIG.batch_size)

        # Should run synchronously in main process
        games = selfplay.generate_games(1)
        assert len(games) == 1
        assert len(games[0]) > 0

    def test_encode_board_coverage(self):
        """Test encode_board function."""
        import chess

        board = chess.Board()
        encoded = encode_board(board)
        assert encoded.shape == (8, 8, 8)


class TestSystem:
    def test_config_defaults(self):
        from src.config import CONFIG

        assert CONFIG.input_planes == 8 and CONFIG.action_size == MoveMapper.ACTION_SIZE

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
    @patch("src.training.search.selfplay.Pool")
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

    @patch("src.training.search.selfplay.Pool")
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

    @patch("src.training.search.selfplay.Pool")
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
            def __init__(self, filters=128, blocks=4, input_planes=8, action_size=133):
                super().__init__()
                self.l = torch.nn.Linear(input_planes * 8 * 8, action_size)

            def forward(self, x):
                z = x.view(x.size(0), -1)
                return self.l(z), torch.tanh(torch.ones((x.size(0), 1)))

        Factory._networks["custom"] = CustomNet
        net = Factory.create_network("custom", CONFIG)
        x = torch.randn(1, CONFIG.input_planes, 8, 8)
        logits, v = net(x)
        assert logits.shape == (1, CONFIG.action_size) and v.shape == (1, 1)


class TestVectorizedMCTS:
    """The ADR-0007 flat vectorized MCTS, at the search + self-play seams."""

    @pytest.fixture
    def vnet(self):
        from src.config import CONFIG

        saved = (CONFIG.filters, CONFIG.blocks)
        CONFIG.filters, CONFIG.blocks = 16, 1
        net = AlphaZeroNet()
        CONFIG.filters, CONFIG.blocks = saved
        return net

    def test_search_returns_3tuple_with_masked_policy(self, vnet):
        from src.config import CONFIG

        game = ChessGame()
        search = VectorizedMCTS(game, vnet, simulations=10, batch=32, gen_device="cpu")
        state = game.get_initial_state()
        policy, root, value = search.search(state)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (CONFIG.action_size,)
        assert np.isclose(policy.sum(), 1.0)
        # Legal-move coverage identical to the game: zero off-legal actions.
        legal = set(game.get_legal_moves(state))
        illegal = [i for i in range(CONFIG.action_size) if i not in legal]
        assert policy[illegal].sum() == 0.0
        assert len(root.children) > 0
        assert -1.0 <= value <= 1.0

    def test_warm_root_reuse_keeps_statistics(self, vnet):
        """Root-replacement (subtree reuse): the passed child comes back
        as the new root (identity), inherited visits survive, and new
        visits are added on top — the branch MCTS's same contract."""
        from src.config import CONFIG

        saved = CONFIG.simulations
        CONFIG.simulations = 128  # several batched iterations at batch 32
        try:
            game = ChessGame()
            search = VectorizedMCTS(game, vnet, simulations=128, batch=32, gen_device="cpu")
            state = game.get_initial_state()
            _, root, _ = search.search(state)
            move = max(root.children, key=lambda a: root.children[a].visit_count)
            child = root.children[move]
            inherited = {a: n.visit_count for a, n in child.children.items()}
            child_visits = child.visit_count
            next_state = game.apply_move(state, move)
            _, root_2, _ = search.search(next_state, root=child)
            assert root_2 is child
            for a, visits in inherited.items():
                assert root_2.children[a].visit_count >= visits
            assert root_2.visit_count >= child_visits
        finally:
            CONFIG.simulations = saved

    def test_fen_mismatch_starts_fresh_root(self, vnet):
        game = ChessGame()
        search = VectorizedMCTS(game, vnet, simulations=10, batch=32, gen_device="cpu")
        s0 = game.get_initial_state()
        _, r0, _ = search.search(s0)
        a0 = game.get_legal_moves(s0)[0]  # a guaranteed-legal action
        s1 = game.apply_move(s0, a0)
        s1 = game.apply_move(s1, game.get_legal_moves(s1)[0])  # never expanded by the s0 search
        _, r1, _ = search.search(s1, root=r0)  # stale root
        assert r1 is not r0
        assert len(r1.children) > 0

    def test_terminal_state_zero_policy(self, vnet):
        # Fool's Mate: rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()
        game = ChessGame()
        state = Position(board, 1, board.fen(), encode_board(board))
        search = VectorizedMCTS(game, vnet, simulations=10, batch=32, gen_device="cpu")
        policy, _, value = search.search(state)
        assert policy.sum() == 0
        assert value != 0  # a real winner, own-color perspective

    def test_selfplay_game_trajectory(self, vnet):
        """One real self-play game through ParallelSelfPlay: legal moves
        only, alternating colors, value targets in [-1, 1]."""
        from src.config import CONFIG

        game = ChessGame()
        search = VectorizedMCTS(game, vnet, simulations=10, batch=32, gen_device="cpu")
        sp = ParallelSelfPlay(game, search, gen_device="cpu", batch_size=CONFIG.batch_size)
        trajs = sp.generate_games(1)
        assert len(trajs) == 1
        traj = trajs[0]
        assert len(traj) >= 2
        (s0, p0, v0), (s1, p1, v1) = traj[0], traj[1]
        assert s0.color == 1 and s1.color == -1
        assert -1.0 <= v0 <= 1.0 and -1.0 <= v1 <= 1.0
        legal = set(game.get_legal_moves(s0))
        assert p0[[i for i in range(CONFIG.action_size) if i not in legal]].sum() == 0.0
