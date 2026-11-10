import numpy as np


class Buffer:
    def __init__(self, capacity: int = 250_000):
        self.data = []
        self.capacity = capacity

    def add_batch(self, trajectories):
        for t in trajectories:
            self.data.extend(t)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity :]

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.data))
        if n == 0:
            return []
        idx = np.random.choice(len(self.data), n, replace=False)
        return [self.data[i] for i in idx]
