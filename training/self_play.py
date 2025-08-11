from abc import ABC, abstractmethod
from multiprocessing import Pool
import os
import numpy as np
from typing import List
import torch

from .game import Game, State
from .mcts import MCTS
from .replay_buffer import ReplayBuffer
from .network import AlphaZeroNet
from .config import Config
from .device_manager import DeviceManager


class SelfPlayStrategy(ABC):
    """Abstract base class for self-play strategies"""
    
    @abstractmethod
    def play_games(self, n_games: int) -> List:
        pass


class ParallelSelfPlay(SelfPlayStrategy):
    """CPU-optimized multiprocessing self-play"""
    
    def __init__(self, game: Game, network: AlphaZeroNet, buffer: ReplayBuffer, config: Config):
        self.game = game
        self.network = network
        self.buffer = buffer
        self.config = config
        self.num_workers = min(os.cpu_count(), 8)
    
    def play_games(self, n_games: int) -> List:
        """Play games in parallel using multiprocessing"""
        net_state = self.network.state_dict()
        chunk_size = max(1, n_games // self.num_workers)
        args = [(self.game, net_state, self.config, i) for i in range(n_games)]
        
        with Pool(processes=self.num_workers) as pool:
            trajectories = pool.map(_play_single_game, args)
        
        # Add trajectories to buffer
        for trajectory in trajectories:
            for state, pi, z in trajectory:
                self.buffer.add(state, pi, z)
        
        return trajectories


class BatchedSelfPlay(SelfPlayStrategy):
    """GPU-optimized batched self-play"""
    
    def __init__(self, game: Game, network: AlphaZeroNet, buffer: ReplayBuffer, 
                 config: Config, device_manager: DeviceManager):
        self.game = game
        self.network = network
        self.buffer = buffer
        self.config = config
        self.device_manager = device_manager
    
    def play_games(self, n_games: int) -> List:
        """Play games with batched GPU evaluation"""
        trajectories = []
        
        for _ in range(n_games):
            trajectory = self._play_single_game()
            trajectories.append(trajectory)
            
            # Add to buffer
            for state, pi, z in trajectory:
                self.buffer.add(state, pi, z)
        
        return trajectories
    
    def _play_single_game(self) -> List:
        """Play a single game with GPU-optimized MCTS"""
        state = self.game.get_initial_state()
        trajectory = []
        
        while not self.game.is_terminal(state):
            mcts = MCTS(self.game, self.network, self.config, self.device_manager)
            pi = mcts.run(state)
            move = np.random.choice(len(pi), p=pi)
            trajectory.append((state, pi))
            state = self.game.apply_move(state, move)
        
        # Assign outcomes
        winner = self.game.get_winner(state)
        final_trajectory = []
        
        for state, pi in trajectory:
            player = state[1]
            if winner == 0:
                z = 0.0
            else:
                z = 1.0 if winner == player else -1.0
            final_trajectory.append((state, pi, z))
        
        return final_trajectory


def _play_single_game(args):
    """Worker function for multiprocessing"""
    game, net_state_dict, config, seed = args
    np.random.seed(seed)
    
    # Recreate network
    net = AlphaZeroNet(config)
    net.load_state_dict(net_state_dict)
    net.eval()
    
    # Create CPU device manager
    cpu_config = Config()
    cpu_config.device = 'cpu'
    device_manager = DeviceManager(cpu_config)
    
    state = game.get_initial_state()
    trajectory = []
    
    while not game.is_terminal(state):
        mcts = MCTS(game, net, cpu_config, device_manager)
        pi = mcts.run(state)
        move = np.random.choice(len(pi), p=pi)
        trajectory.append((state, pi))
        state = game.apply_move(state, move)
    
    # Assign outcomes
    winner = game.get_winner(state)
    final_trajectory = []
    
    for state, pi in trajectory:
        player = state[1]
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        final_trajectory.append((state, pi, z))
    
    return final_trajectory