import numpy as np


# Device options
DEVICE_OPTIONS = [
    "/CPU:0",
    "/GPU:0",
    "/GPU:1",
]
NUM_DEVICE_CHOICES = len(DEVICE_OPTIONS)


# Model options
MODEL_NAME = 'facebook/opt-125m'

# - Adjacency: X->dense1->dense2
ADJ_FORWARD = np.array([
    [0, 1, 0],  # X feeds dense1
    [0, 0, 1],  # dense1 feeds dense2
    [0, 0, 0],  # dense2 feeds nothing
], dtype=np.float32)

ADJ_BACKWARD = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
], dtype=np.float32)

# Operation types (just for demonstration)
OP_TYPES = [0, 1, 1]  # X is type 0, dense1 is type 1, dense2 is type 1

# Output shapes (toy example)
#   X: (batch, 10)
#   dense1: (batch, D)
#   dense2: (batch, 1)
# We'll zero-pad them to length 4 for demonstration: e.g. [batch, 10, 0, 0]
SHAPES = [
    [1000, 10, 0, 0],  # X
    [1000, 300000, 0, 0],  # dense1
    [1000, 1, 0, 0],  # dense2
]
