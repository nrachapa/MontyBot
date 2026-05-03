import chess
import numpy as np

from src.training.game import ChessGame, MoveMapper


def verify_game_logic():
    game = ChessGame()
    state = game.get_initial_state()

    # Check if state has 5 elements (including cached board)
    assert len(state) == 5, f"State should have 5 elements, got {len(state)}"
    assert isinstance(state[4], chess.Board), "State[4] should be a chess.Board object"

    # Check legal moves
    legal_moves = game.get_legal_moves(state)
    assert len(legal_moves) > 0, "Initial state should have legal moves"

    # Apply a move
    move_idx = legal_moves[0]
    next_state = game.apply_move(state, move_idx)

    # Check next state structure
    assert len(next_state) == 5, f"Next state should have 5 elements, got {len(next_state)}"
    assert isinstance(next_state[4], chess.Board), "Next state[4] should be a chess.Board object"
    assert next_state[4].turn == chess.BLACK, "Next turn should be Black"

    # Check terminal
    assert not game.is_terminal(next_state), "Game should not be terminal after 1 move"

    print("Verification successful: ChessGame state optimization works correctly.")


if __name__ == "__main__":
    verify_game_logic()
