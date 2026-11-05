from typing import Any

import chess
import numpy as np

from ...config import CONFIG
from ...core import Game, SearchStrategy
from ..game import MoveMapper, Position


class SunfishSearch(SearchStrategy):
    """
    Heuristic search using Sunfish evaluation (PST + Material).
    """

    def __init__(self, game: Game, evaluator: Any):
        self.game = game
        self.action_size = CONFIG.action_size
        self.evaluator = evaluator
        self.network = None  # Explicitly set to None to avoid AttributeError

    def search(self, state, root=None) -> tuple[np.ndarray, Any]:
        # state is (planes, color, fen)
        board = chess.Board(state.fen)
        legal_moves = list(board.legal_moves)

        p = np.zeros(self.action_size, dtype=np.float32)
        if not legal_moves:
            return p, None, 0.0

        # Evaluate all next states
        scores = []
        indices = []

        turn_color = 1 if board.turn == chess.WHITE else -1

        for move in legal_moves:
            # Index from the PRE-push board: is_capture() is decided by the
            # destination occupancy before the move is made.
            idx = MoveMapper.to_index(move, board)
            board.push(move)

            # Evaluate from White's perspective
            # We pass the state tuple to evaluate, but our evaluator expects state?
            # Wait, SunfishEvaluation.evaluate expects state tuple?
            # Let's check SunfishEvaluation implementation.
            # It takes 'state'.
            # But here we are inside search loop, we have a board object.
            # We should construct a state tuple for the evaluator.
            # Or we should change evaluator interface to accept board?
            # No, interface says 'state: Any'.
            # So we must pass the state tuple.

            next_state = Position(board, 1 if board.turn == chess.WHITE else -1, board.fen(), None)
            raw_score = self.evaluator.evaluate(next_state)

            # Convert to side-to-move perspective
            score = raw_score * turn_color

            scores.append(score)
            indices.append(idx)

            board.pop()

        # Softmax with temperature
        T = 200.0
        scores = np.array(scores)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / T)
        probs = exp_scores / np.sum(exp_scores)

        for idx, prob in zip(indices, probs):
            if idx < self.action_size:
                p[idx] += prob  # add: multiple moves can map to one action

        return p, None, 0.0


class AlphaBetaSearch(SearchStrategy):
    """
    Alpha-Beta Pruning search using an evaluator.
    """

    def __init__(self, game: Game, evaluator: Any, depth: int = 3):
        self.game = game
        self.action_size = CONFIG.action_size
        self.evaluator = evaluator
        self.depth = depth
        self.network = None

    def search(self, state, root=None) -> tuple[np.ndarray, Any]:
        # state is (planes, color, fen, history, board)
        # We need the board object for move generation
        board = state.board

        legal_moves = list(board.legal_moves)
        p = np.zeros(self.action_size, dtype=np.float32)
        if not legal_moves:
            return p, None, 0.0

        best_score = -float("inf")
        best_move = None

        # Root search (maximize for current player)
        alpha = -float("inf")
        beta = float("inf")

        # Order moves? For now, random or simple ordering could help.
        # Let's just iterate.
        for move in legal_moves:
            board.push(move)
            # Next state is opponent's turn, so they will minimize our score.
            # We pass -beta, -alpha for negamax
            score = -self._alphabeta(board, self.depth - 1, -beta, -alpha, -1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

        if best_move:
            idx = MoveMapper.to_index(best_move, board)
            if idx < self.action_size:
                p[idx] = 1.0

        return p, None, 0.0

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float, color: int) -> float:
        if depth == 0 or board.is_game_over():
            # Evaluate from perspective of side to move
            # Sunfish evaluator returns score from White's perspective?
            # Let's check SunfishEvaluation.evaluate
            # It returns score. If white pieces > black pieces, score > 0.
            # We need score from perspective of 'board.turn'.

            # Construct state for evaluator
            state = Position(board, 1 if board.turn == chess.WHITE else -1, board.fen(), None)
            raw_score = self.evaluator.evaluate(state)

            # If board.turn is WHITE, raw_score is correct.
            # If board.turn is BLACK, raw_score is inverted (since raw_score is white advantage).
            # Wait, SunfishEvaluation returns "score".
            # If White is winning, score is positive.
            # If it's White's turn, we want positive score.
            # If it's Black's turn, we want positive score if Black is winning (so negative raw_score).

            perspective = 1 if board.turn == chess.WHITE else -1
            return raw_score * perspective

        legal_moves = list(board.legal_moves)
        best_score = -float("inf")

        for move in legal_moves:
            board.push(move)
            score = -self._alphabeta(board, depth - 1, -beta, -alpha, -color)
            board.pop()

            if score >= beta:
                return score  # Fail hard beta-cutoff
            if score > best_score:
                best_score = score
            alpha = max(alpha, score)

        return best_score
