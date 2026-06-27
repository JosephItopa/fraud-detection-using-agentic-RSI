# memory/replay_buffer.py

import random
from collections import deque


class ReplayBuffer:
    """
    Stores experiences:

    (
        state,
        action,
        reward,
        next_state
    )
    """

    def __init__(
        self,
        capacity=10000
    ):

        self.buffer = deque(
            maxlen=capacity
        )

    def push(
        self,
        state,
        action,
        reward,
        next_state
    ):

        experience = {

            "state":
                state,

            "action":
                action,

            "reward":
                reward,

            "next_state":
                next_state
        }

        self.buffer.append(
            experience
        )

    def sample(
        self,
        batch_size
    ):

        batch_size = min(
            batch_size,
            len(self.buffer)
        )

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):
        return len(self.buffer)