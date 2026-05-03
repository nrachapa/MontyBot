import chess
import numpy as np
import pytest

from src.training.evaluation import SunfishEvaluation
from src.training.game import ChessGame, MoveMapper
from src.training.search import SunfishSearch


class TestSunfishEval:
    def test_initial_board_eval(self):
        """Initial board should be roughly equal."""
        board = chess.Board()
        state = (None, 1, board.fen())
        evaluator = SunfishEvaluation()
        score = evaluator.evaluate(state)
        # Sunfish PSTs might not be perfectly zero-sum for initial position
        assert abs(score) < 50

    def test_material_advantage(self):
        """White material advantage should be positive."""
        # Remove Black Queen
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        state = (None, 1, board.fen())
        evaluator = SunfishEvaluation()
        score = evaluator.evaluate(state)
        # Queen value is 929
        assert score > 800

    def test_positional_advantage(self):
        """Test PST: Knight in center vs corner."""
        # Add pawns to avoid insufficient material draw
        # White Knight on e4 (center)
        b1 = chess.Board("7k/8/8/8/4N3/8/8/K1P5 w - - 0 1")
        s1 = SunfishEvaluation().evaluate((None, 1, b1.fen()))

        # White Knight on h1 (corner)
        b2 = chess.Board("7k/8/8/8/8/8/8/K1P4N w - - 0 1")
        s2 = SunfishEvaluation().evaluate((None, 1, b2.fen()))

        assert s1 > s2


class TestChessGameIntegration:
    def test_fen_persistence(self):
        """Test that ChessGame maintains state via FEN."""
        game = ChessGame()
        state = game.get_initial_state()
        # state is (planes, color, fen)
        assert len(state) == 4
        initial_fen = state[2]

        # Apply move e2e4
        _ = chess.Board(initial_fen)  # Create board to test FEN parsing
        move = chess.Move.from_uci("e2e4")
        idx = MoveMapper.to_index(move)

        next_state = game.apply_move(state, idx)
        next_fen = next_state[2]

        assert initial_fen != next_fen
        # FEN string: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
        # Check for the pawn on e4 (4th rank, 5th file -> 4P3)
        assert "4P3" in next_fen

    def test_turn_switching(self):
        game = ChessGame()
        state = game.get_initial_state()
        # White to move (1)
        assert state[1] == 1

        # Apply move
        moves = game.get_legal_moves(state)
        next_state = game.apply_move(state, moves[0])

        # Black to move (-1)
        assert next_state[1] == -1


class TestMCTSSunfish:
    def test_search_returns_valid_policy(self):
        game = ChessGame()
        evaluator = SunfishEvaluation()
        search = SunfishSearch(game, evaluator)
        state = game.get_initial_state()

        policy = search.search(state)

        assert len(policy) == MoveMapper.ACTION_SIZE
        assert np.isclose(policy.sum(), 1.0)

    def test_search_is_not_uniform(self):
        """Sunfish search should favor good moves over bad ones."""
        game = ChessGame()
        evaluator = SunfishEvaluation()
        search = SunfishSearch(game, evaluator)

        # Position where taking the queen is obvious
        # White to move. Black Queen on d5 hanging.
        # White Rook on d1.
        board = chess.Board("7k/8/8/3q4/8/8/8/3R3K w - - 0 1")
        state = (None, 1, board.fen())

        policy = search.search(state)

        # Find move Rxd5
        move = chess.Move.from_uci("d1d5")
        idx = MoveMapper.to_index(move)

        # Probability of taking queen should be high
        assert policy[idx] > 0.5
