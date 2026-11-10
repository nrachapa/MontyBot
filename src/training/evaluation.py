import chess

from src.core import EvaluationStrategy
from .evaluation_data import piece_val, pst
from .game import Position  # noqa: F401  # forward-ref target for type annotations


class SunfishEvaluation(EvaluationStrategy):
    """Sunfish-style material + piece-square evaluation.

    The evaluator takes a ``Position`` and reads ``board`` directly from it
    — no FEN re-parse, no tuple unpacking.
    """

    def evaluate(self, state: "Position") -> float:
        board = state.board

        if board.is_checkmate():
            # Checkmate score from the perspective of state.color (+1=White, -1=Black)
            # If the side-to-move (board.turn) is checkmated, the score is
            # negative from their perspective, which means positive from state.color's.
            turn_is_white = board.turn == chess.WHITE
            if turn_is_white and state.color == 1:
                return -99999  # White to move, White is checkmated, White perspective
            if turn_is_white and state.color == -1:
                return 99999   # White to move, White is checkmated, Black perspective
            if not turn_is_white and state.color == -1:
                return -99999  # Black to move, Black is checkmated, Black perspective
            if not turn_is_white and state.color == 1:
                return 99999   # Black to move, Black is checkmated, White perspective
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue

            symbol = piece.symbol().upper()
            if symbol not in piece_val:
                continue

            val = piece_val[symbol]
            r, f = divmod(sq, 8)

            if piece.color == chess.WHITE:
                pst_idx = (7 - r) * 8 + f
                score += val + pst[symbol][pst_idx]
            else:
                sq_mirrored = 63 - sq
                r_m, f_m = divmod(sq_mirrored, 8)
                pst_idx = (7 - r_m) * 8 + f_m
                score -= val + pst[symbol][pst_idx]

        return score


class ZeroEvaluation(EvaluationStrategy):
    """A zero-opponent evaluator that always returns 0."""

    def evaluate(self, state: "Position") -> float:
        return 0.0
