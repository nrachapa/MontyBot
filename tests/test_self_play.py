import torch
from training.game import Game
from training.self_play import SelfPlay
from training.replay_buffer import ReplayBuffer

class DummyNet(torch.nn.Module):
    def forward(self, x):
        batch = x.size(0)
        log_p = torch.log_softmax(torch.ones(batch, 9), dim=1)
        v = torch.zeros(batch)
        return log_p, v

def test_self_play_data_collection():
    game = Game()
    buffer = ReplayBuffer(capacity=10)
    net = DummyNet()
    sp = SelfPlay(game, net, buffer, simulations=5)
    sp.play_games(1)
    assert len(buffer) > 0
    state, pi, z = buffer.buffer[0]
    assert len(pi) == 9
    assert -1.0 <= z <= 1.0