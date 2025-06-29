from training.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample():
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        buffer.add(i, i, i)
    assert len(buffer) == 3
    sample = buffer.sample(2)
    assert len(sample) == 2
    for entry in sample:
        assert len(entry) == 3