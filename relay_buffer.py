from dataclasses import dataclass
import torch
from collections import deque
import numpy as np

@dataclass
class Buffer:
    state: torch.Tensor
    next_state: torch.Tensor
    action: int
    reward: float
    done: bool

class RelayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def push(self, item: Buffer) -> None:
        self.buffer.append(item)
    
    def sample(self, batch_size):
        state, next_state, action, reward, done = zip(*np.random.sample(self.buffer, batch_size))
        print(state.shape)
    
    def __len__(self):
        return len(self.buffer)