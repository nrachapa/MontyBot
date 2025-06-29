import torch
from training.game import Game
from training.mcts import MCTS
from training.network import AlphaZeroNet

class DummyNet(torch.nn.Module):
    def forward(self, x):
        batch = x.size(0)
        log_p = torch.log_softmax(torch.ones(batch, 9), dim=1)
        v = torch.zeros(batch)
        return log_p, v

def test_mcts_runs():
    game = Game()
    net = DummyNet()
    mcts = MCTS(game, net, c_puct=1.5, simulations=10)
    state = game.get_initial_state()
    pi = mcts.run(state)
    assert abs(pi.sum() - 1.0) < 1e-5
    assert (pi > 0).sum() > 0