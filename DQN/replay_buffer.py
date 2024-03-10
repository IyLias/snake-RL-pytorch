
import random
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)


    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(torch.tensor, zip(*mini_batch))
        return states, actions, rewards, next_states, dones


    def size(self):
        return len(self.buffer)