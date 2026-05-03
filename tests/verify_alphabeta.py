import chess
import numpy as np

from src.training.evaluation import SunfishEvaluation
from src.training.game import ChessGame
from src.training.system import Factory


def verify_alphabeta():
    game = ChessGame()
    evaluator = SunfishEvaluation()
    # Depth 1 should be enough for mate in 1
    search = Factory.create_search("alphabeta", game, evaluator=evaluator)
    search.depth = 2  # Increase depth slightly

    # Setup a mate in 1 position
    # White to move: R@h1, K@e1. Black: K@a8.
    # Actually, let's use a simpler known fen.
    # White to move and mate in 1:
    # 7k/R7/8/8/8/8/8/7K w - - 0 1
    # Rook on a7, King on h1. Black King on h8.
    # Move Ra8# is mate.

    fen = "7k/R7/8/8/8/8/8/7K w - - 0 1"
    board = chess.Board(fen)
    state = (None, 1, fen, [], board)  # Mock state tuple

    print(f"Testing position: {fen}")
    policy, _ = search.search(state)

    # Find best move
    best_move_idx = np.argmax(policy)
    from src.training.game import MoveMapper

    best_move = MoveMapper.from_index(best_move_idx)

    print(f"Best move found: {best_move}")

    # Check if it is the mating move
    assert best_move == chess.Move.from_uci("a7a8"), f"Expected a7a8, got {best_move}"

    print("Verification successful: Alpha-Beta search found mate in 1.")


if __name__ == "__main__":
    verify_alphabeta()
