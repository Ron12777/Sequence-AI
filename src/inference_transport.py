"""One shared-memory inference slot per actor; queues carry only notifications.

Ownership transfers actor -> server on a request, then server -> actor on its
response. An actor must consume the response before publishing another request.
"""

import multiprocessing as mp
import numpy as np


class InferenceSlot:
    def __init__(self, capacity, channels=11):
        if capacity < 1 or channels < 1:
            raise ValueError("slot dimensions must be positive")
        self.capacity = capacity
        self.channels = channels
        # Store RawArray handles, not NumPy views: spawn must share the underlying
        # allocation rather than pickle a private copy of an ndarray.
        self.observations = mp.RawArray("f", capacity * channels * 100)
        self.policies = mp.RawArray("f", capacity * 100)
        self.values = mp.RawArray("f", capacity)

    def views(self):
        return (
            np.frombuffer(self.observations, np.float32).reshape(
                self.capacity, self.channels, 10, 10
            ),
            np.frombuffer(self.policies, np.float32).reshape(self.capacity, 100),
            np.frombuffer(self.values, np.float32),
        )
