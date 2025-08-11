from math import sqrt
from typing import Dict, List, Optional
from numpy import random, stack, float32
import torch
from torch import from_numpy
import numpy as np

from .game import Game, State
from .network import AlphaZeroNet
from .config import Config
from .device_manager import DeviceManager

class Node:
    def __init__(self, state: State, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self._children: Optional[Dict[int, 'Node']] = None  # Lazy initialization
        self.visit_count = 0
        self.total_value = 0.0
        self._evaluated = False

    @property
    def Q(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0
    
    @property
    def children(self) -> Dict[int, 'Node']:
        """Lazy initialization of children dictionary."""
        if self._children is None:
            self._children = {}
        return self._children

    def expanded(self):
        return self._children is not None and len(self._children) > 0

class MCTS:
    def __init__(self, game: Game, network: AlphaZeroNet, config: Config, device_manager: DeviceManager, 
                 use_onnx: bool = False, onnx_engine=None):
        self.game = game
        self.network = network
        self.device_manager = device_manager
        self.c_puct = config.c_puct
        self.simulations = config.simulations
        self.temperature = config.temperature
        self.batch_evaluation = config.device != 'cpu'
        self.use_onnx = bool(config.use_onnx and onnx_engine is not None)
        self.onnx_engine = onnx_engine

    def state_to_tensor(self, state: State):
        board, player = state
        # two planes: current player stones, opponent stones
        current = (board == player).astype(np.float32)
        # amazonq-ignore-next-line
        opponent = (board == -player).astype(np.float32)
        x = np.stack([current, opponent], axis=0)
        return torch.from_numpy(x)
    
    def batch_state_to_tensor(self, states: List[State]) -> np.ndarray:
        """Vectorized conversion of multiple states to tensor batch."""
        batch_tensors = []
        for state in states:
            board, player = state
            current = (board == player).astype(np.float32)
            opponent = (board == -player).astype(np.float32)
            batch_tensors.append(np.stack([current, opponent], axis=0))
        return np.stack(batch_tensors, axis=0)

    def run(self, state: State):
        root = Node(state)
        for _ in range(self.simulations):
            node = root
            path = [node]
            # selection
            while node.expanded() and not self.game.is_terminal(node.state):
                total_N = sqrt(sum(child.visit_count for child in node.children.values()))
                best_score = -float('inf')
                best_action = None
                best_child = None
                for action, child in node.children.items():
                    u = self.c_puct * child.prior * total_N / (1 + child.visit_count)
                    score = child.Q + u
                    if score > best_score:
                        best_score = score
                        best_action = action
                        best_child = child
                node = best_child
                path.append(node)
            # expand or evaluate terminal
            if self.game.is_terminal(node.state):
                winner = self.game.get_winner(node.state)
                if winner == 0:
                    value = 0.0
                # amazonq-ignore-next-line
                else:
                    value = 1.0 if winner == node.state[1] else -1.0
            else:
                if self.use_onnx and self.onnx_engine:
                    # Use ONNX Runtime for faster inference
                    board_tensor = self.state_to_tensor(node.state).numpy()
                    log_p, v = self.onnx_engine.predict(board_tensor[np.newaxis, ...])
                    priors = np.exp(log_p[0])
                    value = float(v[0])
                else:
                    # Use PyTorch as fallback
                    board_tensor = self.state_to_tensor(node.state).unsqueeze(0)
                    board_tensor = self.device_manager.to_device(board_tensor)
                    
                    with torch.inference_mode():
                        with self.device_manager.autocast_context():
                            log_p, v = self.network(board_tensor)
                    
                    priors = torch.exp(log_p)[0].cpu().numpy()
                    value = float(v.item())
                
                legal = self.game.get_legal_moves(node.state)
                priors_masked = priors[legal]
                priors_masked = priors_masked / np.sum(priors_masked) if np.sum(priors_masked) > 0 else np.ones_like(priors_masked)/len(priors_masked)
                
                for a, p in zip(legal, priors_masked):
                    child_state = self.game.apply_move(node.state, a)
                    node.children[a] = Node(child_state, parent=node, prior=float(p))
            # backup
            for n in reversed(path):
                n.visit_count += 1
                n.total_value += value
                value = -value
        # create policy vector with temperature
        visits = np.zeros(self.game.board_size * self.game.board_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if visits.sum() > 0 and self.temperature > 0:
            visits = visits ** (1 / self.temperature)
            visits = visits / visits.sum()
        elif visits.sum() > 0:
            visits = visits / visits.sum()
        return visits