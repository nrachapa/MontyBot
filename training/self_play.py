import numpy as np
from typing import List

from .game import Game, State
from .mcts import MCTS
from .replay_buffer import ReplayBuffer
from .network import AlphaZeroNet

class SelfPlay:
    def __init__(self, game: Game, network: AlphaZeroNet, buffer: ReplayBuffer, simulations: int = 800, c_puct: float = 1.5):
        self.game = game
        self.network = network
        self.buffer = buffer
        self.simulations = simulations
        self.c_puct = c_puct

    def play_game(self):
        state = self.game.get_initial_state()
        trajectory = []
        while not self.game.is_terminal(state):
            mcts = MCTS(self.game, self.network, c_puct=self.c_puct, simulations=self.simulations)
            pi = mcts.run(state)
            move = np.random.choice(len(pi), p=pi)
            trajectory.append((state, pi))
            state = self.game.apply_move(state, move)
        winner = self.game.get_winner(state)
        for state, pi in trajectory:
            player = state[1]
            if winner == 0:
                z = 0.0
            else:
                z = 1.0 if winner == player else -1.0
            self.buffer.add(state, pi, z)

    def play_games(self, n_games: int):
        for _ in range(n_games):
            self.play_game()