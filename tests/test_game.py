import numpy as np
from training.game import Game

def test_game_initial_state():
    game = Game()
    state = game.get_initial_state()
    board, player = state
    assert board.shape == (3, 3)
    assert np.all(board == 0)
    assert player == 1

def test_legal_moves():
    game = Game()
    state = game.get_initial_state()
    moves = game.get_legal_moves(state)
    assert len(moves) == 9
    # apply one move and check
    state = game.apply_move(state, moves[0])
    moves = game.get_legal_moves(state)
    assert len(moves) == 8

def test_winner_and_terminal():
    game = Game()
    state = game.get_initial_state()
    # X wins on first row
    moves = [0, 3, 1, 4, 2]
    for m in moves:
        state = game.apply_move(state, m)
    assert game.is_terminal(state)
    assert game.get_winner(state) == 1