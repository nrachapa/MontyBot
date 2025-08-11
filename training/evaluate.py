import numpy as np
from typing import Tuple

from .game import Game
from .network import AlphaZeroNet
from .mcts import MCTS
from .config import Config
from .device_manager import DeviceManager


def play_game(game: Game, net1: AlphaZeroNet, net2: AlphaZeroNet, config: Config, device_manager: DeviceManager) -> int:
    """Play one game between two networks. Returns winner: 1, -1, or 0 for draw."""
    state = game.get_initial_state()
    mcts1 = MCTS(game, net1, config, device_manager)
    mcts2 = MCTS(game, net2, config, device_manager)
    
    while not game.is_terminal(state):
        current_player = state[1]
        mcts = mcts1 if current_player == 1 else mcts2
        policy = mcts.run(state)
        legal_moves = game.get_legal_moves(state)
        
        # Secure move selection with input validation
        if len(legal_moves) == 0:
            break
        
        legal_policy = policy[legal_moves]
        policy_sum = legal_policy.sum()
        
        if policy_sum > 0:
            normalized_policy = legal_policy / policy_sum
            move_idx = np.random.choice(len(legal_moves), p=normalized_policy)
            move = legal_moves[move_idx]
        else:
            # Fallback to random legal move
            move = np.random.choice(legal_moves)
        
        state = game.apply_move(state, move)
    
    return game.get_winner(state)


def evaluate_networks(game: Game, net1: AlphaZeroNet, net2: AlphaZeroNet, 
                     config: Config, device_manager: DeviceManager, num_games: int = 10) -> Tuple[int, int, int]:
    """Evaluate net1 vs net2. Returns (wins, losses, draws) for net1."""
    wins = losses = draws = 0
    
    for _ in range(num_games):
        result = play_game(game, net1, net2, config, device_manager)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    return wins, losses, draws


def evaluate_training(game: Game, trained_net: AlphaZeroNet, baseline_net: AlphaZeroNet = None,
                     num_games: int = 20) -> dict:
    """Evaluate training progress with key metrics."""
    from .config import Config
    from .device_manager import DeviceManager
    
    config = Config(device='cpu', simulations=100)  # Use CPU for evaluation
    device_manager = DeviceManager(config)
    
    if baseline_net is None:
        baseline_net = AlphaZeroNet(config)
    
    wins, losses, draws = evaluate_networks(game, trained_net, baseline_net, config, device_manager, num_games)
    win_rate = wins / num_games
    
    return {
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games
    }