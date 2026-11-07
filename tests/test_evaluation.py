"""Tests for the evaluation module through the evaluate(Position) interface."""

import chess
import pytest

from src.training.evaluation import SunfishEvaluation, ZeroEvaluation
from src.training.game import Position


class TestSunfishEvaluation:
    """Verify SunfishEvaluation.evaluate works through the Position interface."""

    def _make(self, fen: str, turn: chess.Color = chess.WHITE) -> Position:
        board = chess.Board(fen)
        return Position(board, 1 if board.turn == chess.WHITE else -1, board.fen(), None)

    # ------------------------------------------------------------------
    # Checkmate detection

    def test_white_checkmate_negative_score(self):
        """Fool's Mate: White checkmated — score far negative."""
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()
        state = Position(board, 1, board.fen(), None)
        evaluator = SunfishEvaluation()
        score = evaluator.evaluate(state)
        assert score == -99999

    def test_black_checkmate_positive_score(self):
        """Black wins by checkmate — score far positive (Black perspective)."""
        # Fool's mate: White checkmated on move 2 (Black wins)
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()
        state = Position(board, -1, board.fen(), None)
        evaluator = SunfishEvaluation()
        score = evaluator.evaluate(state)
        # From Black's perspective (-1), White checkmate = +99999 (Black wins)
        assert score == 99999

    # ------------------------------------------------------------------
    # Draw detection

    def test_stalemate_zero(self):
        """Stalemate returns 0."""
        board = chess.Board("8/8/8/8/8/8/K7/K7 b - - 0 1")
        assert board.is_stalemate()
        state = Position(board, -1, board.fen(), None)
        assert SunfishEvaluation().evaluate(state) == 0.0

    def test_insufficient_material_zero(self):
        """King vs King — insufficient material returns 0."""
        board = chess.Board("8/8/8/8/8/8/8/Kk6 b - - 0 1")
        assert board.is_insufficient_material()
        state = Position(board, -1, board.fen(), None)
        assert SunfishEvaluation().evaluate(state) == 0.0

    # ------------------------------------------------------------------
    # Material counting

    def test_single_white_piece(self):
        """A lone white queen (with kings) should score close to its material value."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
        state = Position(board, 1, board.fen(), None)
        score = SunfishEvaluation().evaluate(state)
        # Material: Q = 929, plus PST bonus
        assert score > 900

    def test_material_sum_white_and_black(self):
        """White queen vs black queen — should score near zero."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A1, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.H8, chess.Piece(chess.QUEEN, chess.BLACK))
        state = Position(board, 1, board.fen(), None)
        score = SunfishEvaluation().evaluate(state)
        # Two equal material pieces — roughly cancels
        assert abs(score) < 100

    # ------------------------------------------------------------------
    # PST bonus detection

    def test_pawn_center_advantage(self):
        """A pawn on e4 should score more than a pawn on a4 due to PST."""
        board_a = chess.Board()
        board_a.clear()
        board_a.set_piece_at(chess.A4, chess.Piece(chess.PAWN, chess.WHITE))
        board_a.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board_a.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        state_a = Position(board_a, 1, board_a.fen(), None)
        score_a = SunfishEvaluation().evaluate(state_a)

        board_e = chess.Board()
        board_e.clear()
        board_e.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board_e.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board_e.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        state_e = Position(board_e, 1, board_e.fen(), None)
        score_e = SunfishEvaluation().evaluate(state_e)

        assert score_e > score_a  # Center pawn > edge pawn

    def test_knight_center_advantage(self):
        """A knight on e4 should score more than on a1."""
        board_a = chess.Board()
        board_a.clear()
        board_a.set_piece_at(chess.A1, chess.Piece(chess.KNIGHT, chess.WHITE))
        board_a.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board_a.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board_a.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))  # add to avoid stalemate
        state_a = Position(board_a, 1, board_a.fen(), None)
        score_a = SunfishEvaluation().evaluate(state_a)

        board_e = chess.Board()
        board_e.clear()
        board_e.set_piece_at(chess.E4, chess.Piece(chess.KNIGHT, chess.WHITE))
        board_e.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board_e.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board_e.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))  # add to avoid stalemate
        state_e = Position(board_e, 1, board_e.fen(), None)
        score_e = SunfishEvaluation().evaluate(state_e)

        assert score_e > score_a  # Center knight > edge knight

    # ------------------------------------------------------------------
    # Black-side mirroring

    def test_black_piece_score_inverted(self):
        """A black queen on e4 should score negative relative to white."""
        board_white = chess.Board()
        board_white.clear()
        board_white.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
        board_white.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board_white.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        state_white = Position(board_white, 1, board_white.fen(), None)
        score_white = SunfishEvaluation().evaluate(state_white)

        board_black = chess.Board()
        board_black.clear()
        board_black.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.BLACK))
        board_black.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board_black.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        state_black = Position(board_black, 1, board_black.fen(), None)
        score_black = SunfishEvaluation().evaluate(state_black)

        assert score_white > score_black  # White queen > Black queen
        # White queen contributes positive, black queen contributes negative
        # (king values partially cancel)

    # ------------------------------------------------------------------
    # Zero evaluation

    def test_zero_evaluation_always_zero(self):
        """ZeroEvaluation always returns 0 regardless of position."""
        board = chess.Board()  # starting position
        state = Position(board, 1, board.fen(), None)
        assert ZeroEvaluation().evaluate(state) == 0.0

    def test_zero_evaluation_on_checkmate(self):
        """Even checkmate returns 0 for ZeroEvaluation."""
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        state = Position(board, 1, board.fen(), None)
        assert ZeroEvaluation().evaluate(state) == 0.0
