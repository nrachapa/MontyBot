import torch
from training.network import AlphaZeroNet


def test_network_output_shapes():
    net = AlphaZeroNet(board_size=3, input_planes=2, filters=256, blocks=2, action_size=9)
    dummy = torch.zeros((1, 2, 3, 3))
    log_p, v = net(dummy)
    assert log_p.shape == (1, 9)
    assert v.shape == (1,)
    probs = torch.exp(log_p)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert v.item() >= -1 and v.item() <= 1