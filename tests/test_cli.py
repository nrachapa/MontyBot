from unittest.mock import MagicMock, Mock, patch

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
        assert "🎲 Starting evaluation..." in result.output
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
    @patch("src.cli.Factory.create_network")
    @patch("os.path.exists", return_value=False)
    def test_evaluate_no_model(self, mock_exists, mock_factory, mock_s3_manager):
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
        assert "🎲 Starting evaluation..." in result.output
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
        assert "🎲 Starting evaluation..." in result.output
        assert "Win Rate: 45.0%" in result.output
        assert "⚠️ Need training" in result.output
        mock_evaluate_impl.assert_called_once()

    @patch("src.cli.S3Manager")
    @patch("src.cli.Factory.create_network")
    @patch("os.path.exists", return_value=True)
    @patch("torch.load")
    def test_evaluate_local_model_exists(self, mock_load, mock_exists, mock_factory, mock_s3_manager):
        """Test evaluate when local model exists"""
        runner = CliRunner()

        # Mock torch load
        mock_load.return_value = {"model_state": {}}

        # Mock network
        mock_network = Mock()
        mock_network.eval.return_value = mock_network
        mock_factory.return_value = mock_network

        # Mock S3Manager (should not be called for download)
        mock_s3_instance = Mock()
        mock_s3_manager.return_value = mock_s3_instance

        # We need to mock the game loop to avoid infinite loop or errors
        with patch("src.cli.chess.Board") as mock_board_class:
            mock_board = Mock()
            mock_board.is_game_over.return_value = True
            mock_board.result.return_value = "1-0"
            mock_board_class.return_value = mock_board

            result = runner.invoke(cli, ["evaluate", "--is-local"])

            assert result.exit_code == 0
            assert "📂 Loading local model.pt..." in result.output
            mock_s3_instance.download_checkpoint.assert_not_called()


class TestEvaluateImpl:
    """Direct tests of _evaluate_impl function"""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.device")
    @patch("random.choice")
    @patch("numpy.argmax", return_value=0)
    @patch("torch.softmax")
    @patch("torch.no_grad")
    @patch("torch.from_numpy")
    @patch("src.cli.chess.Board")
    @patch("src.cli.Factory.create_network")
    @patch("src.cli.S3Manager")
    @patch("os.path.exists", return_value=False)
    def test_evaluate_impl_with_model(
        self,
        mock_exists,
        mock_s3_manager,
        mock_factory,
        mock_board_class,
        mock_from_numpy,
        mock_no_grad,
        mock_softmax,
        mock_argmax,
        mock_choice,
        mock_device,
        mock_cuda,
    ):
        """Test _evaluate_impl function directly"""
        from src.cli import _evaluate_impl

        # Mock S3Manager with checkpoint
        mock_s3_instance = Mock()
        mock_s3_instance.download_checkpoint.return_value = {"model_state": {}}
        mock_s3_manager.return_value = mock_s3_instance

        # Mock network
        mock_network = Mock()
        mock_network.eval.return_value = mock_network
        mock_network.load_state_dict = Mock()
        mock_network.return_value = (Mock(), Mock())
        mock_factory.return_value = mock_network

        # Mock device
        mock_device.return_value = Mock()

        # Mock board - game ends immediately
        mock_board = Mock()
        mock_board.is_game_over.return_value = True
        mock_board.result.return_value = "1-0"  # White wins
        mock_board_class.return_value = mock_board

        # Mock torch operations
        mock_tensor = Mock()
        mock_tensor.unsqueeze.return_value.to.return_value = Mock()
        mock_from_numpy.return_value = mock_tensor

        mock_probs = Mock()
        mock_probs.cpu.return_value.numpy.return_value = [[0.5, 0.3, 0.2]]
        mock_softmax.return_value = mock_probs

        # Set eval_games to 1 for quick test
        original_games = CONFIG.eval_games
        original_device = CONFIG.device
        CONFIG.eval_games = 1
        CONFIG.device = "auto"

        try:
            _evaluate_impl()
            # Should complete without error
            mock_s3_manager.assert_called_once()
            mock_factory.assert_called_once_with("alphazero")
        finally:
            CONFIG.eval_games = original_games
            CONFIG.device = original_device

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.device")
    @patch("random.choice")
    @patch("numpy.argmax", return_value=0)
    @patch("torch.softmax")
    @patch("torch.no_grad")
    @patch("torch.from_numpy")
    @patch("src.cli.chess.Board")
    @patch("src.cli.Factory.create_network")
    @patch("src.cli.S3Manager")
    @patch("os.path.exists", return_value=False)
    def test_evaluate_impl_game_loop(
        self,
        mock_exists,
        mock_s3_manager,
        mock_factory,
        mock_board_class,
        mock_from_numpy,
        mock_no_grad,
        mock_softmax,
        mock_argmax,
        mock_choice,
        mock_device,
        mock_cuda,
    ):
        """Test _evaluate_impl game loop with moves"""
        from src.cli import _evaluate_impl

        # Mock S3Manager
        mock_s3_instance = Mock()
        mock_s3_instance.download_checkpoint.return_value = {"model_state": {}}
        mock_s3_manager.return_value = mock_s3_instance

        # Mock network
        mock_network = MagicMock()
        mock_network.eval.return_value = mock_network
        mock_factory.return_value = mock_network
        # Ensure the network call returns a tuple of (logits, values)
        mock_network.return_value = (MagicMock(), MagicMock())

        # Mock device
        mock_device.return_value = Mock()

        # Mock board behavior
        # We want to simulate a few moves
        import chess

        mock_board = MagicMock()
        # is_game_over returns False twice then True. Add extra True just in case.
        mock_board.is_game_over.side_effect = [False, False] + [True] * 100
        mock_board.move_stack = []  # empty
        mock_board.turn = True  # White
        # Use real chess moves
        mock_board.legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        mock_board.result.return_value = "1-0"
        mock_board.ep_square = None
        mock_board.has_kingside_castling_rights.return_value = False
        mock_board.has_kingside_castling_rights.return_value = False
        mock_board.has_queenside_castling_rights.return_value = False
        mock_board.is_repetition.return_value = False
        mock_board.pieces.return_value = set()
        mock_board.halfmove_clock = 0
        mock_board.turn = chess.WHITE
        mock_board_class.return_value = mock_board

        # Mock torch
        mock_tensor = Mock()
        mock_tensor.unsqueeze.return_value.to.return_value = Mock()
        mock_from_numpy.return_value = mock_tensor

        mock_probs = Mock()
        # Create a full-sized probability array
        full_probs = np.zeros((1, 4672), dtype=np.float32)
        # Set some probabilities for legal moves (indices ~700-800 for e2e4/d2d4)
        # We don't need exact indices, just need array to be large enough
        # But we do access it with legal_indices.
        # Let's just make it large enough.
        mock_probs.cpu.return_value.numpy.return_value = full_probs
        mock_softmax.return_value = mock_probs

        # Set config
        original_games = CONFIG.eval_games
        CONFIG.eval_games = 1

        try:
            _evaluate_impl()
            # Verify game loop interactions
            assert mock_board.push.call_count == 2
        finally:
            CONFIG.eval_games = original_games
