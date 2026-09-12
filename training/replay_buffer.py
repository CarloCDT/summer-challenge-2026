"""Fixed-capacity replay buffer of self-play training examples."""
import random
from collections import deque
from typing import List, Tuple

import numpy as np

# (encoded_state, dense_policy_target, value_target, action_mask). The mask is stored
# explicitly (not reconstructed from the policy target's nonzero support) so training can
# re-run the model with exactly the same legal-action masking self-play used - a legal action
# with a tiny enough MCTS-derived prior could in principle underflow to exact 0.0 in float32,
# which would make a reconstructed mask silently wrong.
Example = Tuple[np.ndarray, np.ndarray, float, np.ndarray]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, example: Example):
        self.buffer.append(example)

    def push_many(self, examples: List[Example]):
        self.buffer.extend(examples)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values, masks = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.stack(policies).astype(np.float32),
            np.array(values, dtype=np.float32),
            np.stack(masks).astype(np.float32),
        )

    def __len__(self):
        return len(self.buffer)
