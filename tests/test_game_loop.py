"""Tests for play_game — the extracted game loop.

Red → green: these tests define the desired interface first,
then we implement play_game(game, policy_fn, opponent_fn=None)
to satisfy them.
"""
import chess
import numpy as np
import pytest

from src.training.game import ChessGame, Position


def _legal_move(game):
    """Return a policy that always picks the first legal action."""
    def policy(state):
        legal = list(game.get_legal_moves(state))
        if not legal:
            return 0
        return legal[0]
    return policy


class TestPlayGameBasic:
    """play_game(game, policy_fn, rng=None) → (trajectory, winner)"""

    def test_returns_trajectory_and_winner(self):
        """play_game returns a list of states and an int winner."""
        from src.training.game import play_game

        game = ChessGame()
        policy = _legal_move(game)

        traj, winner = play_game(game, policy)
        assert isinstance(traj, list)
        assert winner in [-1, 0, 1]

    def test_policy_called_once_per_trajectory_item(self):
        """Each state in the trajectory was produced by exactly one
        policy/opponent call."""
        from src.training.game import play_game

        game = ChessGame()
        policy = _legal_move(game)
        policy_count = [0]
        orig = policy

        def counting_policy(state):
            policy_count[0] += 1
            return orig(state)

        traj, winner = play_game(game, counting_policy, max_moves=5)
        assert policy_count[0] == len(traj) == 5


class TestPlayGameTermination:
    def test_game_advances_then_stops(self):
        """A few-move sequence should advance the game state."""
        from src.training.game import play_game

        game = ChessGame()
        policy = _legal_move(game)
        traj, winner = play_game(game, policy, max_moves=3)
        assert len(traj) == 3
        for s in traj:
            assert isinstance(s, Position)

    def test_max_moves_limit_stops_game(self):
        """When max_moves is reached, the game must stop even if
        not yet terminal."""
        from src.training.game import play_game

        game = ChessGame()
        policy = _legal_move(game)
        traj, winner = play_game(game, policy, max_moves=10)
        assert len(traj) == 10  # exactly 10 moves

    def test_policy_not_called_if_terminal(self):
        """If the game starts from a terminal state, policy should
        never be called."""
        from src.training.game import play_game

        game = ChessGame()

        def policy(state):
            raise RuntimeError("policy called on terminal state!")

        # Use a custom Game subclass that reports itself as terminal
        # immediately (so get_initial_state is never called — the
        # test doesn't depend on specific board positions).
        class AlwaysTerminalGame(ChessGame):
            def is_terminal(self, state):
                return True

            def get_winner(self, state):
                return 0  # draw

            def get_initial_state(self):
                # Return an empty position — won't be used since
                # is_terminal is always True.
                board = chess.Board()
                return Position(board, 1, board.fen(), None)

            def get_legal_moves(self, state):
                return []

            def apply_move(self, state, move_idx):
                return state

            def get_winner_always(self, state):
                return 0

        game = AlwaysTerminalGame()
        traj, winner = play_game(game, policy)
        assert len(traj) == 0
        assert winner == 0


class TestPlayGameOpponent:
    def test_opponent_fn_is_called_on_opponent_turn(self):
        """When opponent_fn is provided, it is called instead of
        policy_fn on odd-numbered plies (opponent's turn)."""
        from src.training.game import play_game

        game = ChessGame()
        policy = _legal_move(game)
        opp = _legal_move(game)

        policy_calls = [0]
        opponent_calls = [0]

        def counting_policy(state):
            policy_calls[0] += 1
            return policy(state)

        def counting_opponent(state):
            opponent_calls[0] += 1
            return opp(state)

        traj, winner = play_game(
            game, counting_policy, opponent_fn=counting_opponent, max_moves=4
        )
        assert policy_calls[0] >= 1
        assert opponent_calls[0] >= 1
        assert policy_calls[0] + opponent_calls[0] == len(traj)
