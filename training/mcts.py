import math
import numpy as np
import torch
from typing import Dict

from .game import Game, State
from .network import AlphaZeroNet

class Node:
    def __init__(self, state: State, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children: Dict[int, 'Node'] = {}
        self.visit_count = 0
        self.total_value = 0.0

    @property
    def Q(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0

    def expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, game: Game, network: AlphaZeroNet, c_puct: float = 1.5, simulations: int = 800):
        self.game = game
        self.network = network
        self.c_puct = c_puct
        self.simulations = simulations

    def state_to_tensor(self, state: State):
        board, player = state
        # two planes: current player stones, opponent stones
        current = (board == player).astype(np.float32)
        opponent = (board == -player).astype(np.float32)
        x = np.stack([current, opponent], axis=0)
        return torch.from_numpy(x)

    def run(self, state: State):
        root = Node(state)
        for _ in range(self.simulations):
            node = root
            path = [node]
            # selection
            while node.expanded() and not self.game.is_terminal(node.state):
                total_N = math.sqrt(sum(child.visit_count for child in node.children.values()))
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
                else:
                    value = 1.0 if winner == node.state[1] else -1.0
            else:
                board_tensor = self.state_to_tensor(node.state).unsqueeze(0).float()
                with torch.no_grad():
                    log_p, v = self.network(board_tensor)
                priors = torch.exp(log_p)[0].cpu().numpy()
                legal = self.game.get_legal_moves(node.state)
                priors_masked = priors[legal]
                priors_masked = priors_masked / np.sum(priors_masked) if np.sum(priors_masked) > 0 else np.ones_like(priors_masked)/len(priors_masked)
                for a, p in zip(legal, priors_masked):
                    child_state = self.game.apply_move(node.state, a)
                    node.children[a] = Node(child_state, parent=node, prior=float(p))
                value = float(v.item())
            # backup
            for n in reversed(path):
                n.visit_count += 1
                n.total_value += value
                value = -value
        # create policy vector
        visits = np.zeros(self.game.board_size * self.game.board_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if visits.sum() > 0:
            visits = visits / visits.sum()
        return visits