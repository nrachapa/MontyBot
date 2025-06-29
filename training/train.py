import os
import random
from typing import List
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim

from .game import Game
from .network import AlphaZeroNet
from .replay_buffer import ReplayBuffer
from .self_play import SelfPlay
from .mcts import MCTS


def state_to_tensor(game: Game, state):
    board, player = state
    current = (board == player).astype('float32')
    opponent = (board == -player).astype('float32')
    x = torch.from_numpy(np.stack([current, opponent], axis=0))
    return x


def train_network(game: Game, net: AlphaZeroNet, buffer: ReplayBuffer, optimizer):
    batch = buffer.sample(min(len(buffer), 32))
    states, policies, outcomes = zip(*batch)
    inputs = torch.stack([state_to_tensor(game, s) for s in states])
    target_p = torch.tensor(policies, dtype=torch.float32)
    target_v = torch.tensor(outcomes, dtype=torch.float32)
    log_p, v = net(inputs)
    value_loss = F.mse_loss(v, target_v)
    policy_loss = -torch.mean(torch.sum(target_p * log_p, dim=1))
    l2_loss = 1e-4 * sum(p.pow(2).sum() for p in net.parameters())
    loss = value_loss + policy_loss + l2_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    game = Game()
    net = AlphaZeroNet(board_size=game.board_size, action_size=game.board_size ** 2)
    buffer = ReplayBuffer(capacity=1000)
    optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4)

    num_iterations = 1
    games_per_iteration = 1
    train_steps = 1

    for _ in range(num_iterations):
        sp = SelfPlay(game, net, buffer)
        sp.play_games(games_per_iteration)
        for _ in range(train_steps):
            loss = train_network(game, net, buffer, optimizer)
            print(f"loss: {loss}")

if __name__ == "__main__":
    main()