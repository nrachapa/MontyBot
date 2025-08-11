import random
import numpy as np
from typing import List, Tuple, Any

class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: List[Tuple[Any, Any, Any]] = []
        self.position = 0
        # Pre-allocate for better performance
        self._data = np.empty((capacity, 3), dtype=object)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, policy, value):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, policy, value)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) >= batch_size:
            return random.sample(self.buffer, batch_size)
        else:
            # Use numpy for faster sampling when buffer is large
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]


class MemoryMappedBuffer:
    """Memory-mapped buffer for large datasets"""
    
    def __init__(self, capacity: int, state_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Memory-mapped arrays
        self.states = np.memmap("buffer_states.dat", dtype="float32", mode="w+",
                               shape=(capacity, *state_shape))
        self.policies = np.memmap("buffer_policies.dat", dtype="float32", mode="w+",
                                 shape=(capacity, 9))
        self.values = np.memmap("buffer_values.dat", dtype="float32", mode="w+",
                               shape=(capacity,))
    
    def __len__(self):
        return self.size
    
    def add(self, state, policy, value):
        self.states[self.position] = state
        self.policies[self.position] = policy
        self.values[self.position] = value
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = self.states[indices]
        policies = self.policies[indices]
        values = self.values[indices]
        
        return [(states[i], policies[i], values[i]) for i in range(batch_size)]