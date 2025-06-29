import random
from typing import List, Tuple, Any

class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.buffer: List[Tuple[Any, Any, Any]] = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, policy, value):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, policy, value)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)