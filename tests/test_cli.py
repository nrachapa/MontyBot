from unittest.mock import MagicMock, Mock, patch

import chess
import numpy as np
import pytest
from click.testing import CliRunner

from src.cli import cli
from src.config import CONFIG


class TestClickCLI:
    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()
        # Store original CONFIG values
        self.original_config = {
            "filters": CONFIG.filters,
            "blocks": CONFIG.blocks,
            "iterations": CONFIG.iterations,
            "batch_size": CONFIG.batch_size,
            "eval_games": CONFIG.eval_games,
            "eval_threshold": CONFIG.eval_threshold,
        }

    def teardown_method(self):
        """Restore CONFIG after each test"""
        for key, value in self.original_config.items():
            setattr(CONFIG, key, value)

    def test_cli_help(self):
        """Test CLI help output"""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MontyBot Chess Engine CLI" in result.output
        assert "train" in result.output
        assert "evaluate" in result.output
        assert "config" in result.output

    def test_train_help(self):
        """Test train command help"""
        result = self.runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "Launch model training" in result.output
        assert "--iterations" in result.output
        assert "--filters" in result.output

    def test_evaluate_help(self):
        """Test evaluate command help"""
        result = self.runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Evaluate trained model" in result.output
        assert "--games" in result.output
        assert "--threshold" in result.output
        assert "--is-local" in result.output

    @patch("src.training.trainer.Trainer")
    def test_train_local_with_params(self, mock_trainer):
        """Test local training with parameter overrides"""
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        result = self.runner.invoke(
            cli, ["train", "--local", "--iterations", "100", "--filters", "64", "--blocks", "2"]
        )

        assert result.exit_code == 0
        assert "🏠 Starting local training..." in result.output
        assert CONFIG.iterations == 100
        assert CONFIG.filters == 64
        assert CONFIG.blocks == 2
        mock_trainer_instance.train.assert_called_once_with(100)

    @patch("src.infra.sagemaker_manager.launch_training")
    def test_train_cloud(self, mock_launcher):
        """Test cloud training"""
        result = self.runner.invoke(cli, ["train", "--cloud"])

        assert result.exit_code == 0
        assert "🚀 Launching cloud training..." in result.output
        mock_launcher.assert_called_once()

    @patch("src.cli._evaluate_impl")
    def test_evaluate_with_params(self, mock_evaluate):
        """Test evaluate with parameter overrides"""
        result = self.runner.invoke(cli, ["evaluate", "--games", "50", "--threshold", "0.8"])

        assert result.exit_code == 0
        assert "Starting evaluation" in result.output
        assert CONFIG.eval_games == 50
        assert CONFIG.eval_threshold == 0.8
        mock_evaluate.assert_called_once()

    def test_config_show(self):
        """Test config show command"""
        result = self.runner.invoke(cli, ["config", "show"])

        assert result.exit_code == 0
        assert "📋 Current Configuration:" in result.output
        assert f"Filters: {CONFIG.filters}" in result.output
        assert f"Blocks: {CONFIG.blocks}" in result.output
        assert f"Device: {CONFIG.device}" in result.output

    def test_config_validate_valid(self):
        """Test config validate with valid config"""
        result = self.runner.invoke(cli, ["config", "validate"])

        assert result.exit_code == 0
        assert "✅ Configuration is valid" in result.output

    def test_config_validate_invalid(self):
        """Test config validate with invalid config"""
        # Test by creating invalid config data and validating
        with patch("src.cli.CONFIG") as mock_config:
            mock_config.model_validate.side_effect = Exception("Invalid")
            mock_config.model_dump.return_value = {}

            result = self.runner.invoke(cli, ["config", "validate"])

            assert result.exit_code == 1
            assert "❌ Configuration is invalid" in result.output

    def test_config_reset(self):
        """Test config reset command"""
        # Modify config
        CONFIG.filters = 999
        CONFIG.blocks = 999

        result = self.runner.invoke(cli, ["config", "reset"], input="y\n")

        assert result.exit_code == 0
        assert "✅ Configuration reset to defaults" in result.output

    def test_verbose_flag(self):
        """Test verbose flag"""
        result = self.runner.invoke(cli, ["--verbose", "config", "show"])

        assert result.exit_code == 0
        assert "Verbose mode enabled" in result.output

    def test_parameter_validation(self):
        """Test parameter type validation"""
        # Test invalid integer
        result = self.runner.invoke(cli, ["train", "--iterations", "invalid"])
        assert result.exit_code == 2
        assert "Invalid value" in result.output

        # Test invalid float
        result = self.runner.invoke(cli, ["evaluate", "--threshold", "invalid"])
        assert result.exit_code == 2
        assert "Invalid value" in result.output


class TestCLIIntegration:
    """Integration tests with actual components"""

    @patch("src.cli.S3Manager")
    @patch("os.path.exists", return_value=False)
    def test_evaluate_no_model(self, mock_exists, mock_s3_manager):
        """Test evaluate when no model exists"""
        runner = CliRunner()

        # Mock S3Manager to return None (no checkpoint)
        mock_s3_instance = Mock()
        mock_s3_instance.download_checkpoint.return_value = None
        mock_s3_manager.return_value = mock_s3_instance

        result = runner.invoke(cli, ["evaluate"])

        assert result.exit_code == 0
        assert "❌ No trained model found in S3" in result.output

    @patch("src.cli._evaluate_impl")
    def test_evaluate_with_model_success(self, mock_evaluate_impl):
        """Test evaluate command with successful model evaluation"""
        runner = CliRunner()

        # Mock successful evaluation
        def mock_eval(*args, **kwargs):
            import click

            click.echo("Win Rate: 75.0% (15-0-5)")
            click.echo("✅ Model OK!")

        mock_evaluate_impl.side_effect = mock_eval

        result = runner.invoke(cli, ["evaluate"])

        assert result.exit_code == 0
        assert "Starting evaluation" in result.output
        assert "Win Rate: 75.0%" in result.output
        assert "✅ Model OK!" in result.output
        mock_evaluate_impl.assert_called_once()

    @patch("src.cli._evaluate_impl")
    def test_evaluate_with_model_failure(self, mock_evaluate_impl):
        """Test evaluate command with model below threshold"""
        runner = CliRunner()

        # Mock failed evaluation
        def mock_eval(*args, **kwargs):
            import click

            click.echo("Win Rate: 45.0% (9-0-11)")
            click.echo("⚠️ Need training (< 90.0%)")

        mock_evaluate_impl.side_effect = mock_eval

        result = runner.invoke(cli, ["evaluate"])

        assert result.exit_code == 0
        assert "Starting evaluation" in result.output
        assert "Win Rate: 45.0%" in result.output
        assert "⚠️ Need training" in result.output
        mock_evaluate_impl.assert_called_once()

    @patch("src.cli.S3Manager")
    @patch("src.cli.AlphaZeroNet")
    @patch("os.path.exists", return_value=True)
    @patch("torch.load")
    def test_evaluate_local_model_exists(self, mock_load, mock_exists, mock_alpha_zero_net, mock_s3_manager):
        """Test evaluate when local model exists"""
        runner = CliRunner()

        # Mock torch load
        mock_load.return_value = {"model_state": {}}

        # Mock network
        mock_network = Mock()
        mock_network.eval.return_value = mock_network
        mock_alpha_zero_net.return_value = mock_network

        # Mock S3Manager (should not be called for download)
        mock_s3_instance = Mock()
        mock_s3_manager.return_value = mock_s3_instance

        # Mock play_game so the CLI doesn't run a real game loop.
        # The CLI calls play_game(game, policy, max_moves=200);
        # patching at the import ensures the mock is used.
        with patch("src.cli.play_game") as mock_play:
            mock_play.return_value = ([Mock()], 0)  # (traj, winner)

            result = runner.invoke(cli, ["evaluate", "--is-local"])

            assert result.exit_code == 0
            assert "📂 Loading local model.pt..." in result.output
            mock_s3_instance.download_checkpoint.assert_not_called()


class TestEvaluateImpl:
    """Direct tests of _evaluate_impl function"""

    @patch("os.path.exists", return_value=False)
    @patch("src.cli.S3Manager")
    @patch("src.cli.AlphaZeroNet")
    @patch("src.cli.play_game")
    def test_evaluate_impl_with_model(
        self,
        mock_play,
        mock_alpha_zero_net,
        mock_s3_manager,
        mock_exists,
    ):
        """Test _evaluate_impl function directly.

        Note: mock decorators are applied bottom-to-top, so the first
        argument gets the TOP decorator's mock.
        """
        from src.cli import _evaluate_impl

        # mock_play = top decorator = "src.cli.play_game"
        # mock_exists = bottom decorator = "os.path.exists"
        mock_play.return_value = ([], 0)  # play_game returns (traj, winner)
        mock_exists.return_value = False  # os.path.exists returns False

        # mock_alpha_zero_net = 2nd decorator = "src.training.network.AlphaZeroNet"
        mock_network = Mock()
        mock_network.eval.return_value = mock_network
        mock_network.load_state_dict = Mock()
        mock_network.return_value = (Mock(), Mock())
        mock_alpha_zero_net.return_value = mock_network

        # mock_s3_manager = 3rd decorator = "src.cli.S3Manager"
        mock_s3_instance = Mock()
        mock_s3_instance.download_checkpoint.return_value = {"model_state": {}}
        mock_s3_manager.return_value = mock_s3_instance

        # Set config
        original_games = CONFIG.eval_games
        original_device = CONFIG.device
        CONFIG.eval_games = 1
        CONFIG.device = "auto"

        try:
            _evaluate_impl()
            # Should complete without error
            mock_s3_manager.assert_called_once()
            mock_alpha_zero_net.assert_called_once()
            # play_game was called once (for 1 game)
            mock_play.assert_called_once()
        finally:
            CONFIG.eval_games = original_games
            CONFIG.device = original_device

    @patch("os.path.exists", return_value=False)
    @patch("src.cli.S3Manager")
    @patch("src.cli.AlphaZeroNet")
    @patch("torch.device")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("src.cli.play_game")
    def test_evaluate_impl_game_loop(
        self,
        mock_play,
        mock_cuda,
        mock_device,
        mock_alpha_zero_net,
        mock_s3_manager,
        mock_exists,
    ):
        """Test _evaluate_impl calls play_game with the right parameters.

        Note: mock_decorator order is reversed — first arg = top decorator.
        """
        from src.cli import _evaluate_impl

        # mock_play = top = "src.cli.play_game"
        # mock_exists = bottom = "os.path.exists"
        mock_play.return_value = ([], 0)  # (traj, winner)
        mock_exists.return_value = False

        # mock_s3_manager = "src.cli.S3Manager"
        mock_s3_instance = Mock()
        mock_s3_instance.download_checkpoint.return_value = {"model_state": {}}
        mock_s3_manager.return_value = mock_s3_instance

        # mock_alpha_zero_net = "src.training.network.AlphaZeroNet"
        mock_network = MagicMock()
        mock_network.eval.return_value = mock_network
        mock_alpha_zero_net.return_value = mock_network
        mock_network.return_value = (MagicMock(), MagicMock())

        # mock_device / mock_cuda = torch mocks
        mock_device.return_value = Mock()

        # Set config: 3 games so play_game is called 3 times
        original_games = CONFIG.eval_games
        CONFIG.eval_games = 3

        try:
            _evaluate_impl()
            # play_game should be called once per game
            assert mock_play.call_count == 3
            # Verify each call has the right parameters
            for call in mock_play.call_args_list:
                args, kwargs = call
                # args: (game, policy_fn); kwarg: max_moves=200
                assert len(args) >= 2
                assert kwargs.get("max_moves") == 200
        finally:
            CONFIG.eval_games = original_games


class TestOpponentSelection:
    """The evaluate --opponent seam: random (existing) vs the repo heuristic
    (alpha-beta + piece counts). A trained model's true strength signal comes
    from beating the heuristic, not random play (any net beats random)."""

    def test_heuristic_opponent_is_legal(self):
        import src.cli as cli_mod
        from src.training.game import Position, encode_board

        board = chess.Board()
        state = Position(board, -1, board.fen(), encode_board(board))
        move = cli_mod._opponent_policy(state, "heuristic")
        # Return value should be a valid action index
        assert 0 <= move < 133

    def test_random_opponent_is_legal(self):
        import src.cli as cli_mod
        from src.training.game import Position, encode_board

        board = chess.Board()
        for _ in range(10):
            state = Position(board, 1, board.fen(), encode_board(board))
            move = cli_mod._opponent_policy(state, "random")
            # Should be a valid action index
            assert 0 <= move < 133

    def test_heuristic_takes_hanging_rook(self):
        import src.cli as cli_mod
        from src.training.game import Position, encode_board

        # A white rook is dropped on c6 (the c7 pawn's square, vacated first);
        # c6 is undefended, so a capture is a net material gain. The old g2
        # version was untestable: nothing from black's start position can reach
        # g2, so declining was *correct*. The heuristic must play for the
        # capture, not a quiet developing move — this is the check that the
        # negamax sign convention maximizes its own side, not inverts the tree.
        board = chess.Board()
        board.remove_piece_at(chess.C6)
        board.set_piece_at(chess.C6, chess.Piece(chess.ROOK, chess.WHITE))
        board.push(chess.Move.from_uci("a1a2"))  # black to move
        state = Position(board, -1, board.fen(), encode_board(board))
        move = cli_mod._opponent_policy(state, "heuristic")
        # The returned move index should correspond to capturing the rook on c6
        move = cli_mod._opponent_policy(state, "heuristic")
        # The heuristic captures the rook on c6.
        resolved = cli_mod.MoveMapper.from_index(move, board)
        assert resolved.to_square == chess.C6
        assert board.is_capture(resolved)
